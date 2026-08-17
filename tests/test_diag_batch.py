"""Regression test for batched diagnostic forwards (CUDA OOM fix).

Bug: ``diagnostics.permutation_null_alpha`` forwarded the ENTIRE test set in
one ``model(X)`` call (observed alpha + every permutation iteration).  On the
Paderborn cell (63,987 test windows) the LSTM intermediates of that single
forward need ~31.5 GiB -> CUDA OOM on a 22 GiB card.  Fix: all diagnostic
full-split forwards run in chunks of ``diagnostics.DIAG_BATCH`` (default
2048) windows and the alpha tensors are concatenated.

Covered here:
1. BIG-TEST END-TO-END: synthetic bundle with >2048 test windows, both
   opt-in flags on (``log_epoch_indicators``, ``degradation``) — the run
   completes (CPU is fine; the chunking is what prevents the GPU OOM) and
   all final diagnostics are present and valid.
2. CHUNK-SIZE INDEPENDENCE: the same run with ``DIAG_BATCH`` monkeypatched
   to 64 (35 chunks) vs 100000 (single chunk) yields identical final dicts
   (H_alpha / alpha_tvar / S_tau / per_class_alpha / perm_null z /
   degradation) up to a tight floating-point tolerance — chunk-then-concat
   is numerics-preserving and the permutation draws (one
   ``torch.rand(B, T)`` per iteration) are unchanged, so same seed ->
   same permutations -> same z.  On CPU the match is exact; on GPU the
   batch-size-dependent reduction order of the kernels makes exact
   equality impossible (~1e-6 level), so floats are compared with
   rel tol 1e-5 / abs tol 1e-9 (tight enough that a real chunking bug,
   e.g. wrong alpha concatenation giving diffs >> 1e-3, still FAILS),
   while ints/strings/None must match exactly.
3. BIT-IDENTITY (unit): ``_forward_alpha_batched`` concatenated output is
   ``torch.equal`` to the single full forward on the trained model (CPU).
   On GPU, kernel-level reduction-order differences make exact equality
   impossible, so the check falls back to ``torch.allclose`` with
   rtol=1e-5 / atol=1e-8 (alpha/G are float32).

The test set is built directly as a ``DatasetBundle`` (not via
``make_synthetic`` with thousands of runs) purely to keep the test fast;
windows are class-frequency sinusoids + noise, same convention as
``make_synthetic``.

Run from the project root:  ``python tests/test_diag_batch.py``
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from atsf_dcg import diagnostics as diag  # noqa: E402
from atsf_dcg.data import DatasetBundle  # noqa: E402
from atsf_dcg.train import train_one  # noqa: E402
from atsf_dcg.utils import ExpConfig  # noqa: E402

N_TEST = 2200  # > DIAG_BATCH default (2048): exercises the multi-chunk path
N_CLASSES = 3
N_CHANNELS = 4
T = 128


def _big_test_bundle(n_test: int = N_TEST, seed: int = 0) -> DatasetBundle:
    """Small train/val, >2048-window test split (see module docstring)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float64)
    freqs = np.array([0.05, 0.11, 0.17])

    def windows(n: int):
        y = (np.arange(n) % N_CLASSES).astype(np.int64)
        f = freqs[y][:, None, None]
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(n, N_CHANNELS, 1))
        X = np.sin(2.0 * np.pi * f * t[None, None, :] + phase)
        X = X + 0.3 * rng.standard_normal((n, N_CHANNELS, T))
        return X.astype(np.float32), y

    X_train, y_train = windows(96)
    X_val, y_val = windows(48)
    X_test, y_test = windows(n_test)
    return DatasetBundle(
        X_train, y_train, X_val, y_val, X_test, y_test,
        class_names=[f"c{k}" for k in range(N_CLASSES)], report={})


def _check_final(final: dict) -> None:
    assert isinstance(final["H_alpha"], float)
    assert isinstance(final["alpha_tvar"], float)
    assert isinstance(final["S_tau"], float)
    assert 0.0 <= final["H_alpha"] <= 1.0
    assert 0.0 <= final["S_tau"] <= 1.0
    assert set(final["per_class_alpha"]) == set(range(N_CLASSES))
    pn = final["perm_null"]
    assert set(pn) == {"observed_var", "null_mean", "null_std", "z",
                       "null_values"}, sorted(pn)
    assert len(pn["null_values"]) == 20  # n_perm=20 from train_one
    assert pn["z"] is None or math.isfinite(pn["z"])
    assert set(final["degradation"]) == set(__import__(
        "atsf_dcg.degradation", fromlist=["DEGRADATION_NAMES"]).DEGRADATION_NAMES)


def _run(cfg: ExpConfig, bundle: DatasetBundle, diag_batch: int) -> dict:
    old = diag.DIAG_BATCH
    diag.DIAG_BATCH = diag_batch  # monkeypatch the module-level constant
    try:
        return train_one(cfg, bundle, return_model=True,
                         log_epoch_indicators=True, degradation=True)
    finally:
        diag.DIAG_BATCH = old


# Tolerances for the DIAG_BATCH=64 vs 100000 comparison.  Chunking changes
# only the ORDER of floating-point reductions on GPU (cuBLAS/cuDNN kernels
# reduce differently per batch size), so exact bit-identity only holds on
# CPU.  rel tol 1e-5 / abs tol 1e-9 absorbs that (~1e-6 observed) while
# still being ~100x tighter than any diff a genuine chunking bug would
# produce (wrong alpha concatenation, misaligned windows -> >> 1e-3).
REL_TOL = 1e-5
ABS_TOL = 1e-9


def _max_abs_diff(a, b) -> float:
    """Max |a-b| over a pair of numeric leaves (used for the diff report)."""
    if isinstance(a, dict):
        return max(_max_abs_diff(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return max(_max_abs_diff(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, np.integer)) and not isinstance(a, bool):
        return float(abs(int(a) - int(b)))
    if a is None:
        return 0.0
    return abs(float(a) - float(b))


def _assert_close(a, b, path: str = "result") -> None:
    """Recursive comparison: floats within tolerance, everything else exact."""
    if isinstance(a, dict):
        assert isinstance(b, dict), f"{path}: type mismatch"
        assert set(a) == set(b), \
            f"{path}: key mismatch {sorted(a)} vs {sorted(b)}"
        for k in a:
            _assert_close(a[k], b[k], f"{path}.{k}")
        return
    if isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)), f"{path}: type mismatch"
        assert len(a) == len(b), f"{path}: len {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_close(x, y, f"{path}[{i}]")
        return
    if a is None or b is None:
        assert a is None and b is None, f"{path}: {a!r} vs {b!r}"
        return
    if isinstance(a, bool) or isinstance(b, bool) or \
            isinstance(a, (int, np.integer)) or isinstance(b, (int, np.integer)):
        assert a == b, f"{path}: int/bool mismatch {a!r} vs {b!r}"
        return
    if isinstance(a, str) or isinstance(b, str):
        assert a == b, f"{path}: str mismatch {a!r} vs {b!r}"
        return
    fa, fb = float(a), float(b)
    assert math.isclose(fa, fb, rel_tol=REL_TOL, abs_tol=ABS_TOL), \
        f"{path}: float mismatch {fa!r} vs {fb!r} (diff {abs(fa - fb):.3e}, " \
        f"tol rel={REL_TOL} abs={ABS_TOL})"


def main() -> None:
    t0 = time.time()
    bundle = _big_test_bundle()
    assert len(bundle.y_test) == N_TEST > 2048, len(bundle.y_test)
    cfg = replace(ExpConfig(name="full"), epochs=1, patience=1, seed=42)

    # ---- 1+2. flags-on end-to-end, DIAG_BATCH 64 (multi-chunk) vs 100000 --
    res_small = _run(cfg, bundle, diag_batch=64)
    _check_final(res_small["final"])
    for rec in res_small["diagnostics"]:
        assert 0.0 <= rec["h_alpha"] <= 1.0
        assert 0.0 <= rec["s_tau"] <= 1.0
    print(f"[test] OK: {N_TEST}-window test set, DIAG_BATCH=64, flags on — "
          "run completes, final diagnostics present and valid.")

    res_single = _run(cfg, bundle, diag_batch=100000)  # single chunk
    _check_final(res_single["final"])
    # Whole result (final dict, epoch diagnostics, accuracy, macro_f1,
    # epochs_run ...) except the returned model: floats within tolerance,
    # ints/strings/None exact.
    small_cmp = {k: v for k, v in res_small.items() if k != "model"}
    single_cmp = {k: v for k, v in res_single.items() if k != "model"}
    _assert_close(small_cmp, single_cmp)
    z = res_small["final"]["perm_null"]["z"]
    print(f"[test] OK: chunk-size independence — DIAG_BATCH=64 vs 100000 "
          f"give matching final dicts within rel tol {REL_TOL} "
          f"(perm_null z={z}).")
    dH = _max_abs_diff(res_small["final"]["H_alpha"],
                       res_single["final"]["H_alpha"])
    dS = _max_abs_diff(res_small["final"]["S_tau"],
                       res_single["final"]["S_tau"])
    dz = (_max_abs_diff(res_small["final"]["perm_null"]["z"],
                        res_single["final"]["perm_null"]["z"])
          if z is not None else float("nan"))
    print(f"[test] NOTE: max abs diff DIAG_BATCH=64 vs 100000 — "
          f"H_alpha={dH:.3e}, S_tau={dS:.3e}, perm_null z={dz:.3e} "
          "(exact 0 on CPU; small nonzero values are expected on GPU due "
          "to batch-size-dependent kernel reduction order).")

    # ---- 3. bit-identity of the batched helper itself ----------------------
    model = res_small["model"]
    device = next(model.parameters()).device
    X = torch.from_numpy(bundle.X_test).float().to(device)
    model.eval()
    with torch.no_grad():
        _, cache = model(X)
        alpha_full = cache["alpha"]
    for bs in (64, 2048, 100000):
        alpha_b = diag._forward_alpha_batched(model, X, batch_size=bs)
        if torch.equal(alpha_b, alpha_full):
            continue  # exact match (always the case on CPU)
        # GPU: batch-size-dependent reduction order of the float32 kernels
        # makes exact equality impossible; kernel-level differences are
        # acceptable, so fall back to a tight allclose.
        assert torch.allclose(alpha_b, alpha_full, rtol=1e-5, atol=1e-8), \
            (f"batch_size={bs} alpha mismatch: max abs diff "
             f"{(alpha_b - alpha_full).abs().max().item():.3e} exceeds "
             "rtol=1e-5/atol=1e-8")
        print(f"[test] NOTE: batch_size={bs} not bit-identical on this "
              f"device (max abs diff "
              f"{(alpha_b - alpha_full).abs().max().item():.3e}); "
              "within torch.allclose rtol=1e-5/atol=1e-8 — expected on GPU.")
    print("[test] OK: _forward_alpha_batched matches the single full "
          "forward (torch.equal on CPU; allclose rtol=1e-5/atol=1e-8 "
          "fallback) for batch_size 64 / 2048 / 100000.")

    print(f"[test_diag_batch] ALL OK ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
