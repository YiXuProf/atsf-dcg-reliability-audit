"""Smoke tests for the reviewer-mandated fixed-fusion control experiments.

Covers (synthetic data only, tiny epochs):

1. The three new configs (``r1_w/o_spectral``, ``full_fixed_global``,
   ``full_fixed_class``) run end-to-end via the real CLI chain
   (``eval_dump --split train`` -> ``run_experiments --fixed-alpha-file``),
   producing diagnostics jsonl + merged tables.
2. Regression guard: config ``full`` (synthetic micro data, seed 42,
   2 epochs) must be BIT-IDENTICAL to the pre-change baseline captured from
   the unmodified code (``set_seed`` makes it deterministic):
   accuracy 0.16666666666666666, macro-F1 0.047619047619047616.
3. ``alpha_means.json`` is written next to ``alpha.npy`` and its global /
   per-class means match a recomputation from ``alpha.npy`` +
   ``features.npz`` labels (rtol 1e-5).
4. ``--merge-only`` (no alpha file) picks up the new configs in
   ``results_table.csv``.

Unit-level checks first: fixed-fusion init validation, the y=None oracle
error, constant cache alpha, and degenerate perm-null z -> None.

Run from the project root:  ``python tests/test_fixed_fusion.py``
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NEW_CONFIGS = ["r1_w/o_spectral", "full_fixed_global", "full_fixed_class"]
SLUGS = {"r1_w/o_spectral": "r1_wo_spectral",
         "full_fixed_global": "full_fixed_global",
         "full_fixed_class": "full_fixed_class"}
# pre-change baseline (config "full", micro synthetic, seed 42, 2 epochs)
BASELINE_FULL_ACC = 0.16666666666666666
BASELINE_FULL_F1 = 0.047619047619047616


def _unit_tests() -> None:
    """Model-level fixed-fusion contract + bit-identical regression guard."""
    from dataclasses import replace

    import torch

    from atsf_dcg.data import make_synthetic
    from atsf_dcg.model import ATSFDCG
    from atsf_dcg.train import train_one
    from atsf_dcg.utils import ExpConfig, set_seed

    set_seed(0)
    x = torch.randn(4, 16, 128)
    y = torch.tensor([0, 1, 2, 3])

    # fixed_global: constant (B,1,T') alpha, no learnable fusion params
    m = ATSFDCG(16, 6, ExpConfig(name="fg", fusion="fixed_global",
                                 fusion_fixed_alpha=0.3))
    _, cache = m(x)
    assert cache["alpha"].shape == (4, 1, 64)
    assert torch.allclose(cache["alpha"], torch.full((4, 1, 64), 0.3))
    assert not hasattr(m, "alpha_head") and not hasattr(m, "fuse_conv")

    # fixed_class: per-sample oracle alpha from the TRUE labels
    m = ATSFDCG(16, 6, ExpConfig(
        name="fc", fusion="fixed_class",
        fusion_fixed_class_alpha=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)))
    _, cache = m(x, y)
    expect = torch.tensor([0.1, 0.2, 0.3, 0.4]).view(4, 1, 1).expand(4, 1, 64)
    assert torch.allclose(cache["alpha"], expect)
    try:
        m(x)  # oracle without labels must fail loudly
        raise AssertionError("fixed_class accepted y=None")
    except ValueError as e:
        assert "true labels" in str(e)

    # init validation of the fixed modes
    for kw in (dict(fusion="fixed_global"),
               dict(fusion="fixed_global", fusion_fixed_alpha=1.5),
               dict(fusion="fixed_global", fusion_fixed_alpha=0.5,
                    use_spectral=False),
               dict(fusion="fixed_class"),
               dict(fusion="fixed_class", fusion_fixed_class_alpha=(0.1, 0.2)),
               dict(fusion="fixed_class",
                    fusion_fixed_class_alpha=(0.1,) * 5 + (-0.1,))):
        try:
            ATSFDCG(16, 6, ExpConfig(name="bad", **kw))
            raise AssertionError(f"init accepted invalid fixed config: {kw}")
        except ValueError:
            pass

    # existing fusion modes ignore y entirely
    set_seed(0)
    m = ATSFDCG(16, 6, ExpConfig(name="full")).eval()
    with torch.no_grad():
        l1, _ = m(x)
        l2, _ = m(x, y)
    assert torch.equal(l1, l2)

    # ---- regression guard ---------------------------------------------------
    bundle = make_synthetic(n_classes=6, runs_per_class=3, n_channels=16,
                            seed=0)
    cfg = replace(ExpConfig(name="full"), epochs=2, patience=2, seed=42)
    res = train_one(cfg, bundle)
    assert res["accuracy"] == BASELINE_FULL_ACC, (
        f"regression: full/seed42 accuracy {res['accuracy']!r} != pre-change "
        f"baseline {BASELINE_FULL_ACC!r}")
    assert res["macro_f1"] == BASELINE_FULL_F1, (
        f"regression: full/seed42 macro-F1 {res['macro_f1']!r} != pre-change "
        f"baseline {BASELINE_FULL_F1!r}")

    # degenerate perm-null z for constant-alpha controls -> None (not crash)
    cfg_fg = replace(ExpConfig(name="full_fixed_global", fusion="fixed_global",
                               fusion_fixed_alpha=0.5),
                     epochs=2, patience=2, seed=42)
    res_fg = train_one(cfg_fg, bundle)
    final = res_fg["final"]
    assert final["alpha_tvar"] == 0.0
    assert final["perm_null"]["z"] is None
    assert final["perm_null"]["observed_var"] == 0.0
    assert math.isfinite(final["H_alpha"])  # still computable / meaningful
    print("[test] unit OK: fixed-fusion contract, init validation, y=None "
          "error, regression guard (full == pre-change baseline), "
          "degenerate perm-null z -> None.")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    print(f"[test] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


def main() -> None:
    _unit_tests()
    tmp = Path(tempfile.mkdtemp(prefix="atsf_dcg_fixedfusion_"))
    dump_root = tmp / "eval_dump"
    out_dir = tmp / "results"

    # ---- (a) train-split alpha-mean dump (18-class synthetic) --------------
    _run([sys.executable, "-m", "atsf_dcg.eval_dump", "--synthetic",
          "--epochs", "2", "--seed", "42", "--split", "train",
          "--out-dir", str(dump_root)])
    dump_dir = dump_root / "full_seed42_train"
    am_path = dump_dir / "alpha_means.json"
    assert am_path.is_file(), "alpha_means.json missing"
    alpha_means = json.loads(am_path.read_text())
    assert alpha_means["split"] == "train"
    assert set(alpha_means) >= {"global", "per_class", "n_windows"}
    feats = np.load(dump_dir / "features.npz")
    assert list(alpha_means["per_class"]) == list(feats["class_names"]), (
        "per_class keys must cover every bundle class (train split)")

    # (3) recompute means from alpha.npy + true labels, rtol 1e-5
    alpha = np.load(dump_dir / "alpha.npy").astype(np.float64)
    labels = feats["labels"]
    assert np.isclose(alpha_means["global"], alpha.mean(), rtol=1e-5)
    for k, name in enumerate(feats["class_names"]):
        mask = labels == k
        if mask.any():
            assert np.isclose(alpha_means["per_class"][name],
                              alpha[mask].mean(), rtol=1e-5), name
    print("[test] OK: alpha_means.json consistent with alpha.npy (rtol 1e-5).")

    # ---- error path: fixed config without the alpha file --------------------
    proc = subprocess.run(
        [sys.executable, "-m", "atsf_dcg.run_experiments", "--synthetic",
         "--epochs", "2", "--seeds", "42",
         "--configs", "full_fixed_global", "--out-dir", str(tmp / "nope")],
        cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0 and "--fixed-alpha-file" in proc.stderr, (
        proc.stderr[-500:])
    print("[test] OK: fixed config without --fixed-alpha-file errors clearly.")

    # ---- (1) the three new configs end-to-end ------------------------------
    _run([sys.executable, "-m", "atsf_dcg.run_experiments", "--synthetic",
          "--epochs", "2", "--seeds", "42", "43",
          "--configs", *NEW_CONFIGS,
          "--fixed-alpha-file", str(am_path), "--out-dir", str(out_dir)])

    diag_dir = out_dir / "diagnostics"
    for name in NEW_CONFIGS:
        for seed in (42, 43):
            f = diag_dir / f"{SLUGS[name]}_seed{seed}.jsonl"
            assert f.is_file(), f"missing {f}"
            finals = [json.loads(l)["final"] for l in
                      f.read_text().strip().splitlines()
                      if "final" in json.loads(l)]
            assert len(finals) == 1, f"{f.name}: expected 1 final row"
            final = finals[0]
            assert math.isfinite(float(final["accuracy"]))
            assert math.isfinite(float(final["macro_f1"]))
            if name == "r1_w/o_spectral":  # single-branch: no fusion alpha
                assert final["H_alpha"] is None and final["perm_null"] is None
            else:  # constant alpha: computable metrics present, z is null
                assert math.isfinite(float(final["H_alpha"]))
                assert float(final["alpha_tvar"]) == 0.0
                assert final["perm_null"]["z"] is None
    rt = pd.read_csv(out_dir / "results_table.csv")
    assert list(rt["config"]) == NEW_CONFIGS, list(rt["config"])
    rem = pd.read_csv(out_dir / "remedy_table.csv")
    assert len(rem) == 3 and not rem["accuracy_mean"].isna().any()
    assert (out_dir / "protocol_report.json").is_file()
    print("[test] OK: 3 new configs end-to-end, diag jsonl + tables valid.")

    # ---- (4) merge-only includes the new configs (no alpha file needed) ----
    _run([sys.executable, "-m", "atsf_dcg.run_experiments", "--merge-only",
          "--out-dir", str(out_dir)])
    rt_m = pd.read_csv(out_dir / "results_table.csv")
    assert list(rt_m["config"]) == NEW_CONFIGS, list(rt_m["config"])
    for col in ("accuracy", "macro_f1"):
        assert (rt_m[col] == rt[col]).all(), "merge-only table drift"
    pn = pd.read_csv(out_dir / "perm_null_summary.csv")
    assert set(pn["config"]) == {"full_fixed_global", "full_fixed_class"}
    print("[test] OK: merge-only (no alpha file) includes the new configs.")
    print(f"[test] outputs at {tmp}")


if __name__ == "__main__":
    main()
