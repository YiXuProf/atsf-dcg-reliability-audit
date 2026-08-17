"""Tests for the two opt-in experiment-pipeline features:

- ``--log-epoch-indicators`` / ``train_one(..., log_epoch_indicators=True)``:
  per-epoch ``h_alpha`` (routing entropy) and ``s_tau`` (gate saturation at
  tau=0.9) on a fixed deterministic first-256-window validation subset;
- ``--degradation`` / ``train_one(..., degradation=True)``: test-time
  sensor-degradation robustness eval (7 degradations applied
  post-normalization, in per-channel-std units, seeded per run seed),
  stored as ``final["degradation"]`` and aggregated by the merge into
  ``degradation_table.csv``.

Covers:
1. DEFAULT-OFF REGRESSION: smoke run without the flags -> epoch records
   have NO h_alpha/s_tau keys, final rows have NO "degradation" key, no
   degradation_table.csv is written, and the build_configs baseline guards
   still pass byte-identically;
2. FLAGS ON (synthetic micro data, config full, 2 epochs): epoch records
   contain h_alpha/s_tau floats in [0,1]; final["degradation"] has all 7
   keys with accuracies in [0,1]; configs without fusion/gating get nulls
   exactly like the final block; determinism: same seed twice -> identical
   degradation accuracies; degradation_table.csv written on merge with the
   expected shape (clean + 7 rows);
3. DEGRADATION SANITY (unit): every degradation preserves the (B, N, T)
   shape (downsample output length correct), is deterministic under its
   seed, and a trained tiny model yields valid accuracy floats in [0,1]
   (no monotone clean-vs-degraded assertion: too strict for tiny models).

Run from the project root:  ``python tests/test_degradation.py``
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from atsf_dcg import degradation as deg  # noqa: E402
from atsf_dcg.data import make_synthetic  # noqa: E402
from atsf_dcg.run_experiments import build_configs  # noqa: E402
from atsf_dcg.train import train_one  # noqa: E402
from atsf_dcg.utils import ExpConfig  # noqa: E402

DEG_KEYS = list(deg.DEGRADATION_NAMES)
assert len(DEG_KEYS) == 7


def _run(cmd: list[str], cwd: Path = ROOT) -> subprocess.CompletedProcess:
    print(f"[test] running: {' '.join(cmd)}")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          env=env)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().strip().splitlines() if l.strip()]


# ---- 3. degradation unit sanity -------------------------------------------

def _unit_sanity() -> None:
    rng_np = np.random.RandomState(0)
    X = rng_np.normal(size=(8, 16, 128)).astype(np.float32)
    for i, name in enumerate(DEG_KEYS):
        rng = np.random.RandomState(42 + deg.SEED_OFFSET + i)
        out = deg.apply_degradation(X, name, rng)
        assert out.shape == X.shape, (name, out.shape)  # downsample length too
        assert np.isfinite(out).all(), name
        assert not np.shares_memory(out, X), name
        # same seed -> identical output
        rng2 = np.random.RandomState(42 + deg.SEED_OFFSET + i)
        out2 = deg.apply_degradation(X, name, rng2)
        assert np.array_equal(out, out2), f"{name} not deterministic"
        # downsample is the only lossless-rng-free one; others must differ
        if name != "downsample":
            assert not np.array_equal(out, X.astype(np.float64)), name
    # downsample really smooths: stride-2 subsample + interpolate back to T
    d = deg.apply_degradation(X, "downsample", np.random.RandomState(0))
    assert d.shape[-1] == X.shape[-1] == 128
    try:
        deg.apply_degradation(X, "nope", np.random.RandomState(0))
        raise AssertionError("unknown degradation name accepted")
    except ValueError:
        pass
    print("[test] OK: 7 degradations preserve (B,N,T), finite, deterministic, "
          "downsample length correct, unknown name rejected.")


# ---- 1. default-off regression ---------------------------------------------

def _default_off_regression() -> None:
    # baseline build_configs guards (byte-identical default grids)
    base = Path(__file__).with_name("baseline_build_configs.txt").read_text()
    smoke = Path(__file__).with_name("baseline_build_configs_smoke.txt").read_text()
    assert repr(build_configs()) == base, "default build_configs() changed"
    assert repr(build_configs(smoke=True)) == smoke, \
        "default build_configs(smoke=True) changed"
    print("[test] OK: baseline build_configs guards pass (byte-identical).")

    tmp = Path(tempfile.mkdtemp(prefix="atsf_dcg_deg_off_"))
    out_dir = tmp / "results"
    _run([sys.executable, "-m", "atsf_dcg.run_experiments",
          "--synthetic", "--smoke", "--out-dir", str(out_dir)], cwd=tmp)
    for f in sorted((out_dir / "diagnostics").glob("*_seed*.jsonl")):
        for rec in _jsonl(f):
            if "epoch" in rec:
                assert "h_alpha" not in rec and "s_tau" not in rec, \
                    (f.name, rec)
            if "final" in rec:
                assert "degradation" not in rec["final"], f.name
    assert not (out_dir / "degradation_table.csv").exists(), \
        "degradation_table.csv must not be written without degradation data"
    print("[test] OK: default-off smoke run — no h_alpha/s_tau epoch keys, "
          "no final['degradation'], no degradation_table.csv.")


# ---- 2. flags on: records, determinism, merge table -------------------------

def _check_degradation_dict(d: dict) -> None:
    assert set(d) == set(DEG_KEYS), sorted(d)
    for k, v in d.items():
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)


def _determinism_direct() -> None:
    """Same seed twice -> identical degradation accuracies + h_alpha trail."""
    bundle = make_synthetic(n_classes=6, runs_per_class=3, n_channels=16,
                            seed=0)
    cfg = replace(ExpConfig(name="full"), epochs=2, patience=2, seed=42)
    r1 = train_one(cfg, bundle, log_epoch_indicators=True, degradation=True)
    r2 = train_one(cfg, bundle, log_epoch_indicators=True, degradation=True)
    _check_degradation_dict(r1["final"]["degradation"])
    assert r1["final"]["degradation"] == r2["final"]["degradation"]
    h1 = [rec["h_alpha"] for rec in r1["diagnostics"]]
    h2 = [rec["h_alpha"] for rec in r2["diagnostics"]]
    assert h1 == h2
    for rec in r1["diagnostics"]:
        assert 0.0 <= rec["h_alpha"] <= 1.0
        assert 0.0 <= rec["s_tau"] <= 1.0
    print("[test] OK: determinism — identical degradation accuracies and "
          f"h_alpha trail across two seed-42 runs {h1}.")


def _null_indicator_guards() -> None:
    """Configs without fusion/gate get nulls exactly like the final block."""
    bundle = make_synthetic(n_classes=6, runs_per_class=3, n_channels=16,
                            seed=0)
    # w/o_gating: alpha present (h_alpha float) but no gate (s_tau null)
    cfg = replace(ExpConfig(name="w/o_gating", gating="none"),
                  epochs=1, patience=1, seed=42)
    res = train_one(cfg, bundle, log_epoch_indicators=True, degradation=True)
    rec = res["diagnostics"][0]
    assert isinstance(rec["h_alpha"], float) and 0.0 <= rec["h_alpha"] <= 1.0
    assert rec["s_tau"] is None
    assert res["final"]["H_alpha"] is not None and res["final"]["S_tau"] is None
    _check_degradation_dict(res["final"]["degradation"])
    # tsnet_vanilla: single branch, no fusion alpha and no gate -> both null
    cfg_v = replace(ExpConfig(name="tsnet_vanilla", use_spectral=False,
                              fusion="none", gating="none"),
                    epochs=1, patience=1, seed=42)
    res_v = train_one(cfg_v, bundle, arch="tsnet", log_epoch_indicators=True)
    rec_v = res_v["diagnostics"][0]
    assert rec_v["h_alpha"] is None and rec_v["s_tau"] is None
    assert "degradation" not in res_v["final"]  # flag off here
    print("[test] OK: null guards mirror the final block (w/o_gating -> "
          "s_tau null; tsnet_vanilla -> both null).")


def _flags_on_end_to_end() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="atsf_dcg_deg_on_"))
    out_dir = tmp / "results"
    _run([sys.executable, "-m", "atsf_dcg.run_experiments",
          "--synthetic", "--smoke", "--configs", "full",
          "--degradation", "--log-epoch-indicators",
          "--out-dir", str(out_dir)], cwd=tmp)
    diag_dir = out_dir / "diagnostics"
    files = sorted(diag_dir.glob("full_seed*.jsonl"))
    assert len(files) == 2, files  # smoke seeds 42, 43
    for f in files:
        epochs = finals = 0
        for rec in _jsonl(f):
            if "epoch" in rec:
                epochs += 1
                assert isinstance(rec["h_alpha"], float)
                assert isinstance(rec["s_tau"], float)
                assert 0.0 <= rec["h_alpha"] <= 1.0
                assert 0.0 <= rec["s_tau"] <= 1.0
            if "final" in rec:
                finals += 1
                _check_degradation_dict(rec["final"]["degradation"])
        assert epochs == 2 and finals == 1, (f.name, epochs, finals)
    print("[test] OK: flags-on smoke run — h_alpha/s_tau in [0,1] per epoch, "
          "final['degradation'] has all 7 accuracies in [0,1].")

    # degradation_table.csv written by the fresh run
    dt_path = out_dir / "degradation_table.csv"
    assert dt_path.is_file(), "degradation_table.csv missing after --degradation"
    dt = pd.read_csv(dt_path)
    assert list(dt["degradation"]) == ["clean", *DEG_KEYS], list(dt["degradation"])
    assert "full" in dt.columns and "delta_vs_clean_pp" in dt.columns
    assert float(dt.loc[dt["degradation"] == "clean", "delta_vs_clean_pp"][0]) == 0.0
    print(f"[test] OK: degradation_table.csv shape {dt.shape} "
          f"(clean + 7 rows x configs + delta_vs_clean_pp).")

    # merge-only rebuilds it from the jsonl final rows
    _run([sys.executable, "-m", "atsf_dcg.run_experiments", "--merge-only",
          "--out-dir", str(out_dir)], cwd=tmp)
    dt_m = pd.read_csv(dt_path)
    assert list(dt_m["degradation"]) == ["clean", *DEG_KEYS]
    assert (dt_m["full"] == dt["full"]).all(), "merge changed degradation table"
    print("[test] OK: --merge-only rebuilds degradation_table.csv from "
          "diagnostics (identical cells).")


def main() -> None:
    t0 = time.time()
    _unit_sanity()
    _default_off_regression()
    _determinism_direct()
    _null_indicator_guards()
    _flags_on_end_to_end()
    print(f"[test_degradation] ALL OK ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
