"""TSF-TimesNet: cross-architecture audit object (plan v6, Stage 2b).

A TimesNet-style (Wu et al., ICLR 2023) TEMPORAL branch replacing the CNN
temporal branch of ATSF-DCG, combined with the SAME spectral front-end /
spectral branch, fusion ("adaptive" / "concat" / "fixed_global" /
"fixed_class" / "none", incl. R2 load-balanced routing) and gating
("dynamic" / "static" / "sparsemax" / "entmax" / "lstm" / "none") machinery
as :class:`atsf_dcg.model.ATSFDCG`, driven by the same ``ExpConfig`` fields.

Implementation: ``TSFTimesNet`` SUBCLASSES ``ATSFDCG`` — the entire
fusion/gating/head/OGM/diagnostics surface (``forward(x, y=None)``, cache
contract, ``_last_alpha``, ``balance_loss()``, ``temporal_params()`` /
``spectral_params()``, R1 aux heads, fixed-fusion validation) is inherited
unchanged; only the temporal branch module is swapped for a TimesNet-style
stack with an identical output shape ``(B, 128, T/2)`` so every downstream
dimension is the same as ATSF-DCG.

Temporal branch (``TimesNetTemporalBranch``):

- embed: Conv1d(N -> d_model=64, k=3) -> ReLU -> BatchNorm;
- 2 ``TimesBlock``s: FFT top-k period detection (top_k=2) along time on the
  batch/channel-mean spectrum, per-period zero-padding + reshape to 2-D
  (period, T/period), Inception-style 2-D conv (bottleneck 64->32, parallel
  kernels {1,3,5}, project back to 64), softmax-weighted aggregation over
  the k periods, residual connection;
- projection to the ATSF-DCG interface: Conv1d(64 -> 128, k=3) -> ReLU ->
  BatchNorm -> MaxPool1d(2) -> ``(B, 128, T/2)``.

Vanilla baseline (Cell D): ``ExpConfig(use_spectral=False, fusion="none",
gating="none")`` gives the plain TimesNet classifier (temporal branch +
shared BiLSTM+attention head, no fusion/gating modules).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .model import ATSFDCG, _conv_relu_bn
from .utils import ExpConfig

D_MODEL = 64
TOP_K = 2
KERNELS = (1, 3, 5)
BOTTLENECK = 32
N_BLOCKS = 2


class InceptionBottleneck(nn.Module):
    """Small Inception-style 2-D conv: bottleneck -> parallel kernels -> project.

    ``(B, d_model, rows, cols) -> (B, d_model, rows, cols)``.
    """

    def __init__(self, d_model: int = D_MODEL, bottleneck: int = BOTTLENECK,
                 kernels: tuple[int, ...] = KERNELS):
        super().__init__()
        self.reduce = nn.Conv2d(d_model, bottleneck, kernel_size=1)
        self.convs = nn.ModuleList(
            nn.Conv2d(bottleneck, bottleneck, kernel_size=k, padding=k // 2)
            for k in kernels
        )
        self.expand = nn.Conv2d(bottleneck * len(kernels), d_model,
                                kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.reduce(x))
        h = torch.cat([torch.relu(c(h)) for c in self.convs], dim=1)
        return self.expand(h)


class TimesBlock(nn.Module):
    """One TimesNet-style block on ``(B, d_model, T)`` sequences.

    Detects the top-k dominant periods via the FFT of the
    batch/channel-mean amplitude spectrum, folds each period into a 2-D
    tensor ``(B, d_model, period, T/period)`` (zero-padding when T is not a
    multiple), applies the Inception bottleneck, unfolds and aggregates with
    softmax weights over the k period amplitudes, then adds the residual.
    """

    def __init__(self, d_model: int = D_MODEL, top_k: int = TOP_K):
        super().__init__()
        self.top_k = top_k
        self.conv = InceptionBottleneck(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, T = x.shape
        # ---- FFT top-k period detection --------------------------------
        amp = torch.fft.rfft(x, dim=-1).abs().mean(dim=(0, 1))  # (T//2+1,)
        amp = amp.clone()
        amp[0] = 0.0  # DC carries no period
        k = min(self.top_k, amp.numel() - 1)
        top = torch.topk(amp, k)
        freqs = top.indices  # frequency bins (>= 1 since DC is zeroed)
        weights = torch.softmax(top.values, dim=0)  # (k,)

        # ---- per-period 2-D folding + inception conv -------------------
        outs = []
        for f in freqs:
            period = max(2, int(math.ceil(T / max(int(f), 1))))
            pad = (-T) % period
            xp = torch.nn.functional.pad(x, (0, pad)) if pad else x
            L = xp.shape[-1]
            folded = xp.reshape(B, D, L // period, period)
            folded = folded + self.conv(folded)  # inception + local residual
            outs.append(folded.reshape(B, D, L)[..., :T])
        agg = torch.stack(outs, dim=0)  # (k, B, D, T)
        agg = (weights.view(k, 1, 1, 1) * agg).sum(dim=0)
        return x + agg


class TimesNetTemporalBranch(nn.Module):
    """TimesNet-style temporal branch: ``(B, N, T) -> (B, 128, T/2)``.

    Same output contract as ``ATSFDCG.temporal_branch`` so the shared
    fusion/gating/head dimensions are identical.
    """

    def __init__(self, n_channels: int, d_model: int = D_MODEL,
                 out_channels: int = 128, n_blocks: int = N_BLOCKS,
                 top_k: int = TOP_K):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv1d(n_channels, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(d_model),
        )
        self.blocks = nn.ModuleList(
            TimesBlock(d_model, top_k=top_k) for _ in range(n_blocks))
        self.proj = nn.Sequential(
            _conv_relu_bn(d_model, out_channels),
            nn.MaxPool1d(2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.proj(h)


class TSFTimesNet(ATSFDCG):
    """TimesNet temporal branch + ATSF-DCG spectral/fusion/gating/head.

    Constructor signature and runtime contract are exactly those of
    :class:`ATSFDCG` (which this class subclasses): everything except the
    temporal branch module — validation, fusion, gating, classifier head,
    R1 aux heads, R2 routing, ``_last_alpha``, parameter groupers — is
    inherited unchanged.
    """

    def __init__(self, n_channels: int, n_classes: int, cfg: ExpConfig):
        super().__init__(n_channels, n_classes, cfg)
        if cfg.use_temporal:
            # swap the CNN temporal branch for the TimesNet-style branch
            # (module reassignment drops the CNN from the parameter set)
            self.temporal_branch = TimesNetTemporalBranch(n_channels)
