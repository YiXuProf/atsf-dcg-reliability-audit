"""Diagnostic metrics for the ATSF-DCG failure analysis (SPEC.md §diagnostics.py).

Shapes
------
- ``alpha``: routing weight tensor ``(B, 1, T')`` or ``(B, C, T')``, values in (0, 1).
- ``G``: gating tensor ``(B, C, T')``, values in (0, 1).
- All functions are pure (no dependency on model internals) and return plain
  Python floats / dicts, except :func:`permutation_null_alpha`, which only
  relies on the public forward contract ``model(x) -> (logits, cache)`` with
  ``cache["alpha"]``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

_EPS = 1e-12
_LOG2 = math.log(2.0)

# Chunk size (windows) for diagnostic forwards over an ENTIRE split.  A
# single forward over e.g. the 64k-window Paderborn test set allocates tens
# of GiB of LSTM intermediates and OOMs a 22 GiB card; 2048 windows need
# ~1 GiB for the 96-channel NPPAD model.  Chunk-then-concat is bit-identical
# to one big forward in eval mode (no batch-size-dependent ops; verified on
# CPU/GPU), and datasets smaller than DIAG_BATCH take the exact single-call
# code path as before.
DIAG_BATCH = 2048


def routing_entropy(alpha: torch.Tensor) -> float:
    """H(α): mean binary entropy of the routing weight, normalised to [0, 1].

    H(α) = mean( -[α·log α + (1−α)·log(1−α)] / log 2 ), computed over batch,
    channel and time.  Values near 1 indicate uncertain (0.5) routing; values
    near 0 indicate collapsed/committed routing.

    SPEC v2: accepts both ``(B, 1, T')`` and ``(B, C, T')`` alpha; the
    statistic is computed elementwise (all channels and time steps enter the
    mean equally), so no channel-dim pre-averaging is needed.

    Computed in float64: in float32 ``1 - 1e-12`` rounds to exactly 1.0, so
    a saturated sigmoid output (α == 1.0) survives the clamp and
    ``(1-a)·log(1-a)`` becomes ``0·(-inf) = NaN``.  float64 keeps the clamp
    bounds distinct from 0/1 and the entropy finite.
    """
    a = alpha.detach().double().clamp(_EPS, 1.0 - _EPS)
    h = -(a * torch.log(a) + (1.0 - a) * torch.log(1.0 - a)) / _LOG2
    return float(h.mean())


def alpha_temporal_variance(alpha: torch.Tensor) -> float:
    """Mean over batch/channels of Var_t(α) — how much routing varies in time.

    SPEC v2: accepts both ``(B, 1, T')`` and ``(B, C, T')`` alpha; the
    variance is taken along the last (time) dim and then averaged over batch
    and channels (per-channel temporal variances, then the grand mean).
    """
    return float(alpha.detach().float().var(dim=-1, unbiased=False).mean())


def per_class_alpha(alpha: torch.Tensor, y: torch.Tensor, K: int) -> dict[int, float]:
    """Mean routing weight per true class; only classes present in ``y`` appear.

    SPEC v2: accepts both ``(B, 1, T')`` and ``(B, C, T')`` alpha; each
    sample's routing weight is the mean over all of its channel/time
    elements before the per-class average.
    """
    a = alpha.detach().float()
    per_sample = a.reshape(a.shape[0], -1).mean(dim=1)  # (B,)
    y = y.detach().long().view(-1)
    out: dict[int, float] = {}
    for k in range(int(K)):
        mask = y == k
        if bool(mask.any()):
            out[k] = float(per_sample[mask].mean())
    return out


def gate_saturation(G: torch.Tensor, tau: float = 0.9) -> float:
    """S(τ) = P(G > τ): fraction of gate activations saturated above ``tau``."""
    return float((G.detach().float() > tau).float().mean())


@torch.no_grad()
def _forward_alpha_batched(model, X: torch.Tensor,
                           y_labels: torch.Tensor | None = None,
                           batch_size: int | None = None) -> torch.Tensor:
    """Eval-mode forward of ``X`` in chunks; returns concatenated ``alpha``.

    Numerics are identical to a single full forward: the model is in eval
    mode (BatchNorm uses running stats, no dropout), every op is independent
    across the batch dim, and the chunk outputs are concatenated in order.
    When ``X`` fits in one chunk the call is exactly the historical single
    forward.  Consumes no extra RNG draws beyond the forwards themselves.

    ``batch_size=None`` reads the module-level ``DIAG_BATCH`` at call time
    (so tests can monkeypatch ``diagnostics.DIAG_BATCH``).
    """
    if batch_size is None:
        batch_size = DIAG_BATCH
    batch_size = int(batch_size)
    if X.shape[0] <= batch_size:
        _, cache = model(X, y_labels) if y_labels is not None else model(X)
        return cache["alpha"]
    chunks: list[torch.Tensor] = []
    for i in range(0, X.shape[0], batch_size):
        Xc = X[i:i + batch_size]
        yc = y_labels[i:i + batch_size] if y_labels is not None else None
        _, cache = model(Xc, yc) if yc is not None else model(Xc)
        chunks.append(cache["alpha"])
    return torch.cat(chunks, dim=0)


def permutation_null_alpha(model, X: torch.Tensor, y: torch.Tensor,
                           n_perm: int = 20) -> dict:
    """Permutation null for the temporal variance of α.

    ``observed_var`` is :func:`alpha_temporal_variance` on the real input.
    For each of ``n_perm`` permutations the time axis of ``X`` is shuffled
    independently per sample (destroying temporal structure while keeping
    marginal statistics), the model is re-run, and the same statistic is
    recomputed from ``cache["alpha"]``.  All forwards (observed and every
    permutation) run in chunks of :data:`DIAG_BATCH` windows via
    :func:`_forward_alpha_batched` so a large test set cannot OOM the GPU;
    the permutation draws (one ``torch.rand(B, T)`` per iteration) and the
    numpy null statistics are unchanged, so the seed -> z-score mapping is
    identical to an unbatched run.  Returns::

        {"observed_var": float, "null_mean": float, "null_std": float,
         "z": float | None, "null_values": list[float]}

    ``y`` is forwarded to the model only when it runs the label-aware
    ``fixed_class`` oracle fusion (``model.fusion == "fixed_class"``); it does
    not otherwise enter the temporal-variance statistic, and time-axis
    permutations leave the label of each window unchanged.  Randomness uses
    the global torch RNG, so it is governed by ``utils.set_seed``.

    SPEC v2: works unchanged when ``cache["alpha"]`` is ``(B, C, T')`` (R2),
    since :func:`alpha_temporal_variance` accepts both shapes.

    Constant-alpha control configs (``fixed_global`` / ``fixed_class``) have
    Var_t(alpha) == 0 for the observed AND every permuted run, so the z
    statistic is a degenerate 0/0: it is recorded as ``None`` (JSON null)
    instead of a meaningless 0.0 or a crash, while ``observed_var`` /
    ``null_mean`` / ``null_std`` (all 0.0, meaningful by construction) are
    still reported.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    X = X.to(device)
    B, N, T = X.shape
    pass_labels = getattr(model, "fusion", None) == "fixed_class"
    y_labels = y.to(device) if pass_labels else None
    with torch.no_grad():
        alpha_obs = _forward_alpha_batched(model, X, y_labels)
        observed = alpha_temporal_variance(alpha_obs)
        nulls: list[float] = []
        for _ in range(int(n_perm)):
            perm = torch.argsort(torch.rand(B, T, device=device), dim=1)  # (B, T)
            Xp = torch.gather(X, 2, perm.unsqueeze(1).expand(-1, N, -1))
            alpha_p = _forward_alpha_batched(model, Xp, y_labels)
            nulls.append(alpha_temporal_variance(alpha_p))
    if was_training:
        model.train()

    arr = np.asarray(nulls, dtype=np.float64)
    null_mean = float(arr.mean())
    null_std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    z: float | None
    if null_std < _EPS and abs(observed - null_mean) < _EPS:
        z = None  # degenerate 0/0 (constant alpha): z is undefined
    else:
        z = float((observed - null_mean) / (null_std + _EPS))
        if not math.isfinite(z):
            z = None
    return {
        "observed_var": float(observed),
        "null_mean": null_mean,
        "null_std": null_std,
        "z": z,
        "null_values": arr.tolist(),
    }


def _global_grad_norm(params) -> float:
    sq = 0.0
    for p in params:
        if p is not None and getattr(p, "grad", None) is not None:
            sq += float(p.grad.detach().float().norm()) ** 2
    return math.sqrt(sq)


def grad_norm_ratio(param_group_a: list, param_group_b: list) -> float:
    """ρ = ||∇θ_a|| / ||∇θ_b||.  Call **after** ``backward()`` (reads ``.grad``).

    Parameters without a gradient are ignored; a small epsilon guards the
    denominator (returns 0.0 when both norms are ~0).
    """
    na = _global_grad_norm(param_group_a)
    nb = _global_grad_norm(param_group_b)
    return na / (nb + _EPS)


def _to_jsonable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().float().item() if obj.numel() == 1 else obj.detach().float().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


class DiagnosticsLogger:
    """Append one JSON line per epoch to ``out_path`` (JSONL format).

    The file is truncated on creation; parent directories are created as
    needed.  Tensor/numpy values in the record are converted automatically.
    """

    def __init__(self, out_path: str):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.out_path, "w", encoding="utf-8")

    def log_epoch(self, epoch: int, record: dict) -> None:
        row = {"epoch": int(epoch)}
        row.update(_to_jsonable(record))
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "DiagnosticsLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
