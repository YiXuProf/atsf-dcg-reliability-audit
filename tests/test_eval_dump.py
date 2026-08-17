"""Smoke test for the eval-dump tooling and run-level figure regeneration.

Runs ``python -m atsf_dcg.eval_dump --synthetic --smoke`` into a temporary
directory, asserts all dump files exist with the expected shapes, then runs
``python scripts/make_fig_runlevel.py`` on the dump and asserts the 5 PNG + 5 SVG
figure files and the stdout stats (per-class alpha band, S(0.9) rates,
silhouette, confusion-matrix accuracy).

Also verifies at unit level that ``train_one(..., return_model=True)``
returns a best-weights model whose test accuracy matches the dict's, and
that the model cache exposes the attention-pooled feature ``z``.

Run from the project root:  ``python tests/test_eval_dump.py``
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# output filenames use the v4 numbering (renamed in commit "figures: rename
# outputs to v4 numbering"); the stdout stat tags below keep the paper's
# [fig1]-[fig5] numbering
FIG_STEMS = ["fig2_fusion_weight_dist", "fig3_fusion_weight_by_class",
             "fig4_gate_heatmap", "fig5_tsne", "fig6_confusion_matrix"]


def _unit_tests() -> None:
    """cache['z'] exists and return_model=True hands back the trained model."""
    from dataclasses import replace

    from atsf_dcg.data import make_synthetic
    from atsf_dcg.model import ATSFDCG
    from atsf_dcg.train import train_one
    from atsf_dcg.utils import ExpConfig

    bundle = make_synthetic(n_classes=6, runs_per_class=3, n_channels=16,
                            seed=0)
    cfg = replace(ExpConfig(name="full"), epochs=2, patience=2, seed=42)
    res = train_one(cfg, bundle, return_model=True)
    assert "model" in res and isinstance(res["model"], ATSFDCG)

    # default call (no return_model) keeps the historical contract
    res_default = train_one(cfg, bundle)
    assert "model" not in res_default
    assert abs(res_default["accuracy"] - res["accuracy"]) < 1e-9, (
        "return_model must not change training behaviour: "
        f"{res_default['accuracy']} vs {res['accuracy']}")

    # z in cache: (B, 128), eval mode, matches dump inference shape
    import torch
    model = res["model"].eval()
    x = torch.from_numpy(bundle.X_test[:4]).float()
    with torch.no_grad():
        logits, cache = model(x)
    assert cache["z"].shape == (4, 128), cache["z"].shape
    assert torch.isfinite(cache["z"]).all()
    assert logits.shape == (4, len(bundle.class_names))
    print("[test] unit assertions OK: cache['z'], return_model round-trip, "
          "default contract unchanged.")


def main() -> None:
    _unit_tests()
    tmp = Path(tempfile.mkdtemp(prefix="atsf_dcg_evaldump_"))
    dump_root = tmp / "eval_dump"
    fig_dir = tmp / "figures"

    # ---- 1. eval_dump smoke path -------------------------------------------
    cmd = [sys.executable, "-m", "atsf_dcg.eval_dump",
           "--synthetic", "--smoke", "--seed", "42",
           "--out-dir", str(dump_root)]
    print(f"[test] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"eval_dump failed with code {proc.returncode}")
    assert "[cross-check]" in proc.stdout, "metric cross-check line missing"

    dump_dir = dump_root / "full_seed42"
    for fname in ("predictions.csv", "alpha.npy", "gates.npy",
                  "features.npz", "meta.json"):
        assert (dump_dir / fname).is_file(), f"missing {fname}"

    meta = json.loads((dump_dir / "meta.json").read_text())
    assert meta["config"] == "full" and meta["seed"] == 42
    assert "run-level leakage-free" in meta["protocol"]
    n_test = int(meta["n_test_windows"])

    preds = pd.read_csv(dump_dir / "predictions.csv")
    assert list(preds.columns) == ["window_idx", "class_name", "y_true",
                                   "y_pred", "correct"], preds.columns
    assert len(preds) == n_test
    assert set(preds["correct"].unique()) <= {0, 1}

    alpha = np.load(dump_dir / "alpha.npy")
    gates = np.load(dump_dir / "gates.npy")
    feats = np.load(dump_dir / "features.npz")
    assert alpha.shape == (n_test, 64), alpha.shape  # (n, T'), channel squeezed
    assert alpha.dtype == np.float32 and (alpha >= 0).all() and (alpha <= 1).all()
    assert gates.shape == (n_test, 128, 64), gates.shape  # (n, C, T')
    assert feats["z"].shape == (n_test, 128), feats["z"].shape
    assert feats["labels"].dtype == np.int64
    assert len(feats["class_names"]) == 6

    # recomputed metrics in meta agree with predictions.csv
    acc_csv = float(preds["correct"].mean())
    assert abs(meta["test_accuracy"] - acc_csv) < 1e-9
    assert abs(meta["test_accuracy"] - meta["train_one_accuracy"]) < 1e-6
    print("[test] eval_dump outputs OK: shapes, dtypes, metric cross-check.")

    # ---- 2. figure script end-to-end ----------------------------------------
    cmd = [sys.executable, str(ROOT / "scripts" / "make_fig_runlevel.py"),
           "--dump-dir", str(dump_dir), "--out-dir", str(fig_dir)]
    print(f"[test] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"make_fig_runlevel failed with code {proc.returncode}")

    for stem in FIG_STEMS:
        for ext in ("png", "svg"):
            f = fig_dir / f"{stem}.{ext}"
            assert f.is_file() and f.stat().st_size > 0, f"missing/empty {f}"

    # stdout stats needed for the paper text must be present and parse
    m = re.search(r"\[fig2\] per-class mean alpha band: \[([\d.]+), ([\d.]+)\]",
                  proc.stdout)
    assert m, proc.stdout
    assert all(0.0 <= float(v) <= 1.0 for v in m.groups())
    m = re.search(r"\[fig4\] silhouette \(high-dim\): raw=(-?[\d.]+) -> "
                  r"z=(-?[\d.]+)", proc.stdout)
    assert m, proc.stdout
    assert all(-1.0 <= float(v) <= 1.0 for v in m.groups())
    assert "[fig3] gate saturation S(0.9)" in proc.stdout
    m = re.search(r"\[fig5\] accuracy implied by confusion matrix: ([\d.]+)",
                  proc.stdout)
    assert m and abs(float(m.group(1)) - acc_csv) < 5e-4

    print("[test] OK: 5 PNG + 5 SVG figures, all stdout stats present.")
    print(f"[test] outputs at {tmp}")


if __name__ == "__main__":
    main()
