"""ATSF-DCG model (SPEC.md §model.py, Agent-2).

Adaptive Temporal-Spectral fusion with Dynamic Channel/time-step Gating.

Input: ``x`` of shape ``(B, N, T)`` (batch, sensor channels, time; T=128 in
our protocol). Both branches emit ``(B, 128, T')`` with T' = T/2 = 64.

- Temporal branch: Conv1d(N->64, k=3, pad=1) -> ReLU -> BN ->
  Conv1d(64->128, k=3, pad=1) -> ReLU -> BN -> MaxPool1d(2,2).
- Spectral branch: front-end chosen by ``cfg.spectral_frontend``:
    * "fft": |rfft(x)| along time, DC dropped -> (B, N, T/2=64);
    * "stft": ``remedies.STFTFrontend`` -> (B, N, 8*L), adaptively pooled to
      (B, N, 64);
    * "sinc": ``remedies.SincFrontend`` -> (B, n_filters=64, T), adaptively
      pooled to (B, 64, 64);
  then a CNN mirroring the temporal branch without MaxPool -> H_f (B,128,64).
- Fusion ("adaptive"): alpha = sigmoid(Linear(256->1)) on the channel-concat
  of [H_t, H_f], giving (B, 1, T') broadcast:
  ``H = alpha * H_t + (1 - alpha) * H_f``.
  "concat": 1x1 Conv1d(256->128) on the concat. "none" (both branches
  active): unweighted mean, no learnable fusion. Single-branch configs skip
  fusion entirely.  "fixed_global" / "fixed_class" (reviewer control
  experiments, both branches only): non-learnable fusion with a constant
  alpha — a global scalar ``cfg.fusion_fixed_alpha``, or a per-class oracle
  ``cfg.fusion_fixed_class_alpha`` indexed by the TRUE label passed as
  ``forward(x, y)`` (y is required; it is an intentional diagnostic upper
  bound).  ``cache["alpha"]`` carries the constant expanded to (B,1,T').
- Gating: "dynamic" G = sigmoid(Conv1d(128->128, k=1)) applied per time
  step; "static" SE block (GAP -> FC(128->32) -> ReLU -> FC(32->128) ->
  sigmoid) broadcast from (B,128,1); "none" identity (cache["G"] = None).
  SPEC v2 R4 adds non-saturating gates: "sparsemax" / "entmax"
  (Conv1d(128->128, k=1) logits mapped along the channel dim per time step
  by ``remedies.sparsemax`` / ``remedies.entmax15``) and "lstm"
  (``remedies.LSTMGate``: per-time-step channel mean/std stats -> LSTM(64)
  -> Linear(->128) -> sigmoid). All three return G (B,128,T').
- SPEC v2 R2 (``cfg.r2_load_balanced`` with fusion "adaptive" and both
  branches): alpha becomes per-channel/per-time-step, (B,128,T'), via
  sigmoid(Conv1d(256->128, k=1)/tau); ``cfg.r2_gumbel`` switches to
  two-expert Gumbel-softmax hard routing (cache["alpha"] keeps the soft
  probability of H_t). ``balance_loss()`` returns the two-expert MoE
  load-balancing term (scalar 0 when R2 is inactive).
- Head: BiLSTM(128 -> 64/direction, 1 layer, batch_first) -> additive
  attention (Linear(128->64) -> tanh -> Linear(64->1) -> softmax over T' ->
  weighted sum) -> FC(128 -> K).

``forward(x, y=None)`` returns ``(logits, cache)`` with ``logits`` of shape
``(B, K)`` and cache keys exactly:
``"H_t"`` / ``"H_f"`` ((B,128,T') or None if the branch is disabled),
``"alpha"`` ((B,1,T') by default, (B,C,T') under R2; values in [0,1];
present when fusion is "adaptive"/"fixed_global"/"fixed_class" with both
branches, else None),
``"G"`` ((B,128,T') in (0,1), or None if gating == "none"),
``"z"`` (attention-pooled feature (B,128), always present),
``"logits_t"`` / ``"logits_f"`` (auxiliary-head outputs when
``cfg.r1_balanced`` and the branch is active, else None).

``temporal_params()`` / ``spectral_params()`` return the parameter lists of
the respective branch (branch CNN + front-end + R1 auxiliary head); a
disabled branch yields an empty list.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .remedies import EntmaxGate, LSTMGate, SincFrontend, SparsemaxGate, STFTFrontend
from .utils import ExpConfig

# length the spectral branch works on after alignment (T=128 -> T'=64)
SPEC_LEN = 64


def _conv_relu_bn(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(cin, cout, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm1d(cout),
    )


class ATSFDCG(nn.Module):
    """Adaptive Temporal-Spectral Fusion with Dynamic Gating (see module docstring)."""

    def __init__(self, n_channels: int, n_classes: int, cfg: ExpConfig):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.cfg = cfg
        if not (cfg.use_temporal or cfg.use_spectral):
            raise ValueError("at least one of use_temporal/use_spectral must be True")
        # fixed-fusion controls (reviewer control experiments): both branches
        # are required and the constant alpha value(s) must be provided
        if cfg.fusion in ("fixed_global", "fixed_class"):
            if not (cfg.use_temporal and cfg.use_spectral):
                raise ValueError(
                    f"fusion={cfg.fusion!r} requires use_temporal and use_spectral")
            if cfg.fusion == "fixed_global":
                if cfg.fusion_fixed_alpha is None:
                    raise ValueError(
                        "fusion='fixed_global' requires cfg.fusion_fixed_alpha "
                        "(inject from an eval-dump alpha_means.json)")
                if not 0.0 <= float(cfg.fusion_fixed_alpha) <= 1.0:
                    raise ValueError(
                        f"fusion_fixed_alpha must be in [0, 1], got "
                        f"{cfg.fusion_fixed_alpha}")
            else:
                if cfg.fusion_fixed_class_alpha is None:
                    raise ValueError(
                        "fusion='fixed_class' requires cfg.fusion_fixed_class_alpha "
                        "(per-class oracle alphas, length n_classes)")
                if len(cfg.fusion_fixed_class_alpha) != n_classes:
                    raise ValueError(
                        f"fusion_fixed_class_alpha length "
                        f"{len(cfg.fusion_fixed_class_alpha)} != n_classes "
                        f"{n_classes}")
                if any(not 0.0 <= float(a) <= 1.0
                       for a in cfg.fusion_fixed_class_alpha):
                    raise ValueError(
                        "fusion_fixed_class_alpha values must be in [0, 1]")

        # ---- temporal branch: (B,N,T) -> (B,128,T/2) ----
        if cfg.use_temporal:
            self.temporal_branch = nn.Sequential(
                _conv_relu_bn(n_channels, 64),
                _conv_relu_bn(64, 128),
                nn.MaxPool1d(2, 2),
            )

        # ---- spectral front-end + branch: -> (B,128,T/2) ----
        if cfg.use_spectral:
            if cfg.spectral_frontend == "fft":
                self.frontend = None
                spec_in = n_channels
            elif cfg.spectral_frontend == "stft":
                self.frontend = STFTFrontend()
                spec_in = n_channels
            elif cfg.spectral_frontend == "sinc":
                self.frontend = SincFrontend(n_channels)
                spec_in = self.frontend.n_filters
            else:
                raise ValueError(f"unknown spectral_frontend: {cfg.spectral_frontend}")
            self.spec_align = (
                None if cfg.spectral_frontend == "fft" else nn.AdaptiveAvgPool1d(SPEC_LEN)
            )
            self.spectral_branch = nn.Sequential(  # mirror of temporal, no MaxPool
                _conv_relu_bn(spec_in, 64),
                _conv_relu_bn(64, 128),
            )

        # ---- fusion ----
        self.fusion = cfg.fusion
        # R2 only applies to adaptive fusion with both branches active
        self.r2_load_balanced = bool(
            cfg.r2_load_balanced
            and cfg.use_temporal and cfg.use_spectral
            and cfg.fusion == "adaptive"
        )
        self._last_alpha: torch.Tensor | None = None  # (B,C,T') from last forward
        if cfg.use_temporal and cfg.use_spectral:
            if cfg.fusion == "adaptive":
                if self.r2_load_balanced:
                    self.alpha_conv = nn.Conv1d(256, 128, kernel_size=1)
                else:
                    self.alpha_head = nn.Linear(256, 1)
            elif cfg.fusion == "concat":
                self.fuse_conv = nn.Conv1d(256, 128, kernel_size=1)
            elif cfg.fusion in ("fixed_global", "fixed_class"):
                # no learnable fusion parameters; alpha is a fixed constant
                if cfg.fusion == "fixed_class":
                    self.register_buffer(
                        "fixed_class_alpha",
                        torch.tensor(list(cfg.fusion_fixed_class_alpha),
                                     dtype=torch.float32))
            elif cfg.fusion != "none":
                raise ValueError(f"unknown fusion: {cfg.fusion}")

        # ---- gating ----
        self.gating = cfg.gating
        if cfg.gating == "dynamic":
            self.gate_conv = nn.Conv1d(128, 128, kernel_size=1)
        elif cfg.gating == "static":
            self.se_fc1 = nn.Linear(128, 32)
            self.se_fc2 = nn.Linear(32, 128)
        elif cfg.gating == "sparsemax":
            self.gate = SparsemaxGate(128)
        elif cfg.gating == "entmax":
            self.gate = EntmaxGate(128)
        elif cfg.gating == "lstm":
            self.gate = LSTMGate(128, hidden=64)
        elif cfg.gating != "none":
            raise ValueError(f"unknown gating: {cfg.gating}")

        # ---- classifier head ----
        self.lstm = nn.LSTM(128, 64, num_layers=1, batch_first=True, bidirectional=True)
        self.attn_proj = nn.Linear(128, 64)
        self.attn_score = nn.Linear(64, 1)
        self.classifier = nn.Linear(128, n_classes)

        # ---- R1 auxiliary heads ----
        self.r1_balanced = cfg.r1_balanced
        if cfg.r1_balanced:
            if cfg.use_temporal:
                self.aux_head_t = nn.Linear(128, n_classes)
            if cfg.use_spectral:
                self.aux_head_f = nn.Linear(128, n_classes)

    def _spectral_input(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, T) -> (B, C_spec, 64) for the spectral CNN."""
        if self.frontend is None:  # fft: |rfft| magnitude, DC dropped
            return torch.fft.rfft(x, dim=-1).abs()[..., 1:]
        return self.spec_align(self.frontend(x))

    def forward(self, x: torch.Tensor,
                y: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        # x: (B, N, T);  y: true labels (B,) — used ONLY by fusion
        # "fixed_class" (oracle control); every other fusion mode ignores it.
        H_t = self.temporal_branch(x) if self.cfg.use_temporal else None
        H_f = (
            self.spectral_branch(self._spectral_input(x))
            if self.cfg.use_spectral
            else None
        )

        # fusion
        alpha = None
        if H_t is not None and H_f is not None:
            if self.fusion == "adaptive":
                cat = torch.cat([H_t, H_f], dim=1)  # (B, 256, T')
                if self.r2_load_balanced:
                    alpha, H = self._r2_fuse(cat, H_t, H_f)  # alpha: (B,C,T')
                else:
                    alpha = torch.sigmoid(self.alpha_head(cat.transpose(1, 2))).transpose(1, 2)
                    H = alpha * H_t + (1.0 - alpha) * H_f  # alpha: (B,1,T') broadcast
            elif self.fusion == "concat":
                H = self.fuse_conv(torch.cat([H_t, H_f], dim=1))
            elif self.fusion == "fixed_global":
                # control: constant scalar alpha, no learnable fusion
                alpha = H_t.new_full((H_t.shape[0], 1, H_t.shape[-1]),
                                     float(self.cfg.fusion_fixed_alpha))
                H = alpha * H_t + (1.0 - alpha) * H_f
            elif self.fusion == "fixed_class":
                # ORACLE control: per-sample alpha = class alpha of the TRUE
                # label (deliberate diagnostic upper bound), (B,1,1) broadcast
                if y is None:
                    raise ValueError(
                        "fusion='fixed_class' requires true labels: call "
                        "model(x, y) — this is an oracle control and indexes "
                        "the per-class alpha vector by the true class")
                a = self.fixed_class_alpha[y.long().view(-1)].view(-1, 1, 1)
                H = a * H_t + (1.0 - a) * H_f
                alpha = a.expand(-1, -1, H_t.shape[-1])  # cache: (B,1,T')
            else:  # "none": unweighted mean, no learnable fusion module
                H = 0.5 * (H_t + H_f)
        else:
            H = H_t if H_t is not None else H_f

        # gating
        G = None
        if self.gating == "dynamic":
            G = torch.sigmoid(self.gate_conv(H))  # (B,128,T')
            H = G * H
        elif self.gating == "static":
            g = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(H.mean(dim=2)))))
            G = g.unsqueeze(-1).expand(-1, -1, H.shape[-1])  # (B,128,T') for cache
            H = g.unsqueeze(-1) * H
        elif self.gating in ("sparsemax", "entmax", "lstm"):  # R4 gates
            G = self.gate(H)  # (B,128,T')
            H = G * H

        # BiLSTM + additive attention + FC
        seq = H.transpose(1, 2)  # (B, T', 128)
        out, _ = self.lstm(seq)  # (B, T', 128)
        score = self.attn_score(torch.tanh(self.attn_proj(out)))  # (B, T', 1)
        w = torch.softmax(score, dim=1)
        z = (w * out).sum(dim=1)  # (B, 128)
        logits = self.classifier(z)

        # R1 auxiliary heads
        logits_t = logits_f = None
        if self.r1_balanced:
            if H_t is not None:
                logits_t = self.aux_head_t(H_t.mean(dim=2))
            if H_f is not None:
                logits_f = self.aux_head_f(H_f.mean(dim=2))

        self._last_alpha = alpha
        cache = {
            "H_t": H_t,
            "H_f": H_f,
            "alpha": alpha,
            "G": G,
            "z": z,
            "logits_t": logits_t,
            "logits_f": logits_f,
        }
        return logits, cache

    def _r2_fuse(self, cat: torch.Tensor, H_t: torch.Tensor, H_f: torch.Tensor):
        """R2 per-channel/per-time-step routing (SPEC v2).

        Default: ``alpha = sigmoid(Conv1d(256->128, k=1)([H_t;H_f]) / tau)``,
        shape ``(B, C, T')``; ``H = alpha * H_t + (1 - alpha) * H_f``.
        Gumbel variant (``cfg.r2_gumbel``): two-expert Gumbel-softmax along a
        new expert dim (temperature ``tau``, hard straight-through during
        training) routes each element to H_t or H_f; the returned alpha is
        the *soft* probability of choosing H_t.  Returns ``(alpha, H)``.
        """
        tau = max(float(self.cfg.r2_temperature), 1e-6)
        logits_c = self.alpha_conv(cat)  # (B, C, T')
        if self.cfg.r2_gumbel:
            two = torch.stack([logits_c, -logits_c], dim=1)  # (B, 2, C, T')
            alpha = torch.softmax(two / tau, dim=1)[:, 0]  # soft P(H_t), (B,C,T')
            g = F.gumbel_softmax(two, tau=tau, hard=self.training, dim=1)
            H = g[:, 0] * H_t + g[:, 1] * H_f
        else:
            alpha = torch.sigmoid(logits_c / tau)
            H = alpha * H_t + (1.0 - alpha) * H_f
        return alpha, H

    def balance_loss(self) -> torch.Tensor:
        """MoE-style load-balancing loss for R2 (SPEC v2).

        Two-expert form ``L = 2 * sum_e(P_e * U_e)`` where ``P_e`` is the
        batch mean of the routing probability of expert e (alpha for e=0 /
        1-alpha for e=1) and ``U_e`` is the argmax assignment fraction of
        expert e.  Uses ``alpha`` from the most recent ``forward`` (the soft
        probability in the Gumbel variant); the gradient flows through P.
        Returns a scalar 0 tensor when R2 is not enabled or no forward has
        run yet.
        """
        if not self.r2_load_balanced or self._last_alpha is None:
            device = next(self.parameters()).device
            return torch.zeros((), device=device)
        a = self._last_alpha.float()  # (B, C, T'), P(expert 0)
        probs = torch.stack([a, 1.0 - a], dim=1)  # (B, 2, C, T')
        P = probs.mean(dim=(0, 2, 3))  # (2,)
        assign = probs.argmax(dim=1)  # (B, C, T'), non-differentiable by design
        U = torch.stack([(assign == e).float().mean() for e in range(2)])  # (2,)
        return 2.0 * (P * U).sum()

    def temporal_params(self) -> list:
        """Parameters of the temporal branch (CNN + R1 aux head), [] if disabled."""
        if not self.cfg.use_temporal:
            return []
        params = list(self.temporal_branch.parameters())
        if self.r1_balanced:
            params += list(self.aux_head_t.parameters())
        return params

    def spectral_params(self) -> list:
        """Parameters of the spectral branch (front-end + CNN + R1 aux head), [] if disabled."""
        if not self.cfg.use_spectral:
            return []
        params = list(self.spectral_branch.parameters())
        if self.frontend is not None:
            params += list(self.frontend.parameters())
        if self.r1_balanced:
            params += list(self.aux_head_f.parameters())
        return params
