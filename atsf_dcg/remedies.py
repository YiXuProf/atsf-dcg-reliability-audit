"""Remedy modules for ATSF-DCG (SPEC.md §remedies.py, Agent-2).

Alternative spectral front-ends (remedy R3) and an OGM-GE-style gradient
balancer (remedy R1).

Shape conventions (all time-series tensors are ``(B, N, T)`` = batch,
channels, time; T=128 in our protocol):

- ``STFTFrontend``: ``(B, N, T) -> (B, N, F_groups * L)`` where
  ``F_groups = 8`` frequency groups and ``L = T // hop + 1`` STFT frames
  (L=9 for T=128, n_fft=64, hop=16, center padding), i.e. ``(B, N, 72)``.
  The last axis is group-major: each frequency group's per-frame log-power
  time trajectory is contiguous (index = group * L + frame). Unlike a
  whole-window ``|FFT|``, this keeps time localization: each frame only
  summarizes a 64-sample neighbourhood. ``model.ATSFDCG`` aligns the last
  axis to 64 with ``nn.AdaptiveAvgPool1d(64)`` before the spectral CNN.

- ``SincFrontend``: ``(B, N, T) -> (B, n_filters, T)``. A SincNet-style
  bank of learnable band-pass filters (parametrized by cutoffs) is applied
  per channel and aggregated by the channel mean. Because filtering is a
  linear operator and the filter bank is shared across channels, this is
  exactly equivalent to filtering the channel-mean signal once, which is
  what the implementation does (96x cheaper, identical result).
  ``model.ATSFDCG`` aligns T -> 64 with ``nn.AdaptiveAvgPool1d(64)``.

- ``OGMBalancer``: simplified adaptation of OGM-GE (Peng et al., 2022) to
  the two-branch temporal/spectral setting. It tracks each branch's
  auxiliary-head accuracy with an EMA and returns gradient coefficients in
  (0, 2) that sum to 2: the lagging branch (lower smoothed accuracy) gets a
  coefficient > 1, the leading branch < 1. Used by train.py to rescale the
  R1 auxiliary losses.

SPEC v2 (R4) adds sparse, non-saturating gating mappings and gate modules:

- ``sparsemax(z, dim)``: standard sparsemax (Martins & Astudillo, 2016) --
  Euclidean projection of ``z`` onto the probability simplex along ``dim``;
  the output distribution contains exact zeros.
- ``entmax15(z, dim)``: alpha-entmax with alpha=1.5 (Peters et al., 2019),
  ``p = [(alpha-1) z - tau]_+^(1/(alpha-1)) = [0.5 z - tau]_+^2`` with the
  threshold ``tau`` found by bisection so that ``sum(p) = 1``; also sparse
  (exact zeros) but smoother than sparsemax.
- ``SparsemaxGate`` / ``EntmaxGate``: ``(B, C, T') -> (B, C, T')``;
  gate logits = Conv1d(C->C, k=1)(H), mapped per time step along the
  channel dim (``dim=1``) by sparsemax / entmax15 respectively.
- ``LSTMGate``: ``(B, C, T') -> (B, C, T')``; per-time-step channel
  statistics (mean, std) ``(B, T', 2)`` -> LSTM(hidden=64, batch_first)
  -> Linear(64->C) -> sigmoid, giving a time-varying gate ``(B, C, T')``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTFrontend(nn.Module):
    """Log-power STFT front-end -> ``(B, N, 8 * L)`` (see module docstring).

    Computes the STFT (Hann window, center padding), takes the log power
    spectrogram, adaptively average-pools the frequency bins into 8 groups
    (keeping every frame), and flattens group-major so that each group's
    per-frame log-power trajectory is contiguous along the last axis.
    """

    FREQ_GROUPS = 8

    def __init__(self, n_fft: int = 64, hop: int = 16):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, T)
        B, N, T = x.shape
        spec = torch.stft(
            x.reshape(B * N, T),
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )  # (B*N, n_fft//2+1, L)
        power = spec.real.pow(2) + spec.imag.pow(2)
        logp = torch.log(power + 1e-8)
        n_freq, L = logp.shape[-2], logp.shape[-1]
        logp = logp.reshape(B, N, n_freq, L)
        # pool frequency bins into FREQ_GROUPS groups, keep all frames
        pooled = F.adaptive_avg_pool2d(logp, (self.FREQ_GROUPS, L))
        return pooled.reshape(B, N, self.FREQ_GROUPS * L)  # group-major


class SincFrontend(nn.Module):
    """SincNet-style learnable band-pass filter bank -> ``(B, n_filters, T)``.

    Each filter is a windowed ideal band-pass:
    ``h[m] = (f2*sinc(f2*m) - f1*sinc(f1*m)) * hamming[m]`` with cutoffs
    ``0 <= f1 < f2 <= 1`` normalized to the Nyquist frequency (f=1). ``f1``
    and the bandwidth ``f2 - f1`` are learnable (via absolute values, so the
    parametrization stays valid). The bank is applied per channel and
    aggregated by the channel mean (implemented as filtering the channel
    mean, which is mathematically identical; see module docstring).
    """

    def __init__(self, n_channels: int, n_filters: int = 64, kernel: int = 65):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("kernel must be odd so padding preserves T")
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.kernel = kernel
        # cutoffs spread over the low/mid band at init; bandwidth 0.1 Nyquist
        self.f1 = nn.Parameter(torch.linspace(0.02, 0.60, n_filters))
        self.bandwidth = nn.Parameter(torch.full((n_filters,), 0.10))
        half = kernel // 2
        self.register_buffer("m", torch.arange(-half, half + 1, dtype=torch.float32))
        self.register_buffer("win", torch.hamming_window(kernel))

    def filters(self) -> torch.Tensor:
        """Current filter bank, shape (n_filters, kernel)."""
        f1 = self.f1.abs().clamp(max=1.0)
        f2 = (f1 + self.bandwidth.abs()).clamp(max=1.0)
        m = self.m.unsqueeze(0)  # (1, kernel)
        low2 = f2.unsqueeze(1) * torch.sinc(f2.unsqueeze(1) * m)
        low1 = f1.unsqueeze(1) * torch.sinc(f1.unsqueeze(1) * m)
        return (low2 - low1) * self.win.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, T) -> (B, n_filters, T)
        xm = x.mean(dim=1, keepdim=True)  # == per-channel filter + mean
        w = self.filters().unsqueeze(1)  # (n_filters, 1, kernel)
        return F.conv1d(xm, w, padding=self.kernel // 2)


class OGMBalancer:
    """Simplified OGM-GE adaptation for the two-branch ATSF-DCG (remedy R1).

    Tracks EMAs of both branches' auxiliary-head accuracies and returns
    gradient coefficients ``(c_t, c_f)`` with ``c_t + c_f = 2``:
    the lagging branch (lower smoothed accuracy) receives a coefficient
    > 1 so its auxiliary loss gradient is boosted, the leading branch is
    damped (< 1). train.py multiplies the R1 auxiliary losses by these
    coefficients each epoch.
    """

    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self.ema_t: float | None = None
        self.ema_f: float | None = None

    def coefficients(self, acc_t: float, acc_f: float) -> tuple[float, float]:
        if self.ema_t is None:
            self.ema_t, self.ema_f = float(acc_t), float(acc_f)
        else:
            m = self.momentum
            self.ema_t = m * self.ema_t + (1 - m) * float(acc_t)
            self.ema_f = m * self.ema_f + (1 - m) * float(acc_f)
        total = self.ema_t + self.ema_f + 1e-8
        c_t = 2.0 * self.ema_f / total  # temporal lags (low acc) -> c_t > 1
        c_f = 2.0 * self.ema_t / total
        return float(c_t), float(c_f)


def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax: Euclidean projection of ``z`` onto the simplex along ``dim``.

    Standard implementation (Martins & Astudillo, ICML 2016): sort, find the
    support size ``k(z)`` and threshold ``tau(z) = (cumsum_k - 1) / k``, then
    return ``max(z - tau, 0)``.  The output is a probability distribution
    along ``dim`` and contains exact zeros for coordinates outside the
    support.  Differentiable almost everywhere (like ReLU).
    """
    zs, _ = torch.sort(z, descending=True, dim=dim)
    cssv = zs.cumsum(dim)
    n = z.size(dim)
    view = [1] * z.dim()
    view[dim] = n
    k_range = torch.arange(1, n + 1, device=z.device, dtype=z.dtype).view(view)
    support = 1.0 + k_range * zs > cssv  # at least the max element qualifies
    k = support.sum(dim=dim, keepdim=True)  # (.., 1, ..), >= 1
    tau = (cssv.gather(dim, (k.long() - 1).clamp(min=0)) - 1.0) / k.to(z.dtype)
    return torch.clamp(z - tau, min=0.0)


def entmax15(z: torch.Tensor, dim: int = -1, n_iter: int = 50) -> torch.Tensor:
    """alpha-entmax with alpha=1.5 along ``dim`` (Peters et al., ACL 2019).

    ``p = [0.5 * z - tau]_+ ** 2`` where the threshold ``tau`` is the unique
    value such that ``sum(p) = 1``; found by ``n_iter`` bisection steps
    between the standard bounds (``x.min - 1`` gives ``sum(p) >= 1``,
    ``x.max`` gives ``sum(p) == 0``, with ``x = 0.5 * z``).  50 bisection
    iterations are far beyond float32 precision.  Like sparsemax the output
    contains exact zeros, but it is smoother (C1) inside the support.
    """
    x = 0.5 * z
    lo = x.min(dim=dim, keepdim=True).values - 1.0
    hi = x.max(dim=dim, keepdim=True).values
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        s = torch.clamp(x - mid, min=0.0).pow(2).sum(dim=dim, keepdim=True)
        above = s > 1.0  # sum too large -> raise the threshold
        lo = torch.where(above, mid, lo)
        hi = torch.where(above, hi, mid)
    tau = 0.5 * (lo + hi)
    return torch.clamp(x - tau, min=0.0).pow(2)


class SparsemaxGate(nn.Module):
    """R4 sparsemax gate: ``(B, C, T') -> (B, C, T')``.

    Gate logits = Conv1d(C->C, k=1)(H); each time step's channel vector is
    mapped to a sparse distribution along the channel dim (``dim=1``) by
    :func:`sparsemax`.  Output rows sum to 1 per time step and contain exact
    zeros (non-saturating alternative to per-channel sigmoid).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return sparsemax(self.conv(H), dim=1)


class EntmaxGate(nn.Module):
    """R4 entmax(1.5) gate: ``(B, C, T') -> (B, C, T')``.

    Same structure as :class:`SparsemaxGate` but uses :func:`entmax15` along
    the channel dim (``dim=1``) per time step.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return entmax15(self.conv(H), dim=1)


class LSTMGate(nn.Module):
    """R4 LSTM-conditioned gate: ``(B, C, T') -> (B, C, T')``.

    Per-time-step channel statistics of ``H`` (channel mean and std,
    ``(B, T', 2)``) are fed to a small LSTM (hidden=64, batch_first); a
    Linear(64 -> C) + sigmoid head then emits a gate value per channel and
    time step.
    """

    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(2, hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, channels)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        stats = torch.stack([H.mean(dim=1), H.std(dim=1)], dim=-1)  # (B,T',2)
        out, _ = self.lstm(stats)  # (B, T', hidden)
        return torch.sigmoid(self.fc(out)).transpose(1, 2)  # (B, C, T')
