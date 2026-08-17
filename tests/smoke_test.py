"""Smoke test: run the full experiment pipeline on synthetic micro data.

Runs ``python -m atsf_dcg.run_experiments --synthetic --smoke``
(3 configs x 2 seeds x 2 epochs) into a temporary results directory and
asserts that all four output families exist, that ``results_table.csv`` has
the right shape, and that all reported numbers are finite.

SPEC v2 additions (unit-level, before the pipeline run):
- R2 configs produce alpha of shape (B, C, T') and all diagnostics
  functions accept it without error; ``balance_loss()`` backpropagates.
- ``gating="sparsemax"`` (R4) produces a gate G containing exact zeros
  (sparsity verification); entmax/lstm gates forward correctly.

Run from the project root:  ``python tests/smoke_test.py``
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MEAN_STD = re.compile(r"^(-?\d+\.\d+)±(-?\d+\.\d+)$")
SMOKE_CONFIGS = ["full", "full_r2", "full_r4_lstm"]


def _unit_tests() -> None:
    """SPEC v2 unit assertions (R2 diagnostics compat, R4 gate sparsity)."""
    import torch

    from atsf_dcg import diagnostics as diag
    from atsf_dcg.model import ATSFDCG
    from atsf_dcg.utils import ExpConfig, set_seed

    set_seed(0)
    x = torch.randn(4, 16, 128)
    y = torch.randint(0, 6, (4,))

    # ---- R2: (B, C, T') alpha + diagnostics compatibility ------------------
    cfg_r2 = ExpConfig(name="full_r2", r2_load_balanced=True)
    model = ATSFDCG(16, 6, cfg_r2)
    model.train()
    logits, cache = model(x)
    alpha = cache["alpha"]
    assert alpha is not None and alpha.shape == (4, 128, 64), alpha.shape
    assert math.isfinite(diag.routing_entropy(alpha))
    assert math.isfinite(diag.alpha_temporal_variance(alpha))
    assert diag.per_class_alpha(alpha, y, 6)
    perm = diag.permutation_null_alpha(model, x, y, n_perm=2)
    assert math.isfinite(perm["z"]), perm
    # balance_loss is a scalar and differentiable through the alpha head
    loss = logits.sum() + cfg_r2.r2_lambda_balance * model.balance_loss()
    loss.backward()
    assert model.alpha_conv.weight.grad is not None
    # Gumbel variant forwards with (B, C, T') soft alpha as well
    cfg_g = ExpConfig(name="full_r2_gumbel", r2_load_balanced=True, r2_gumbel=True)
    _, cache_g = ATSFDCG(16, 6, cfg_g)(x)
    assert cache_g["alpha"].shape == (4, 128, 64), cache_g["alpha"].shape

    # ---- R4: sparsemax gate is exactly sparse; entmax/lstm forward --------
    cfg_s = ExpConfig(name="full_r4_sparsemax", gating="sparsemax")
    _, cache_s = ATSFDCG(16, 6, cfg_s)(x)
    G = cache_s["G"]
    assert G is not None and G.shape == (4, 128, 64), G.shape
    assert bool((G <= 1e-6).any()), "sparsemax gate G should contain near-0 elements"
    for gname in ("entmax", "lstm"):
        _, cache_r4 = ATSFDCG(16, 6, ExpConfig(name=gname, gating=gname))(x)
        assert cache_r4["G"].shape == (4, 128, 64), (gname, cache_r4["G"].shape)

    print("[smoke] unit assertions OK: R2 (B,C,T') alpha + diagnostics, "
          "balance_loss backward, R4 gates (sparsemax sparsity verified).")


def _check_mean_std(s: str) -> None:
    m = MEAN_STD.match(str(s).strip())
    assert m, f"bad mean±std cell: {s!r}"
    for part in m.groups():
        assert math.isfinite(float(part)), f"non-finite value in {s!r}"


def main() -> None:
    _unit_tests()
    tmp = Path(tempfile.mkdtemp(prefix="atsf_dcg_smoke_"))
    out_dir = tmp / "results"
    cmd = [sys.executable, "-m", "atsf_dcg.run_experiments",
           "--synthetic", "--smoke", "--out-dir", str(out_dir)]
    print(f"[smoke] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"run_experiments failed with code {proc.returncode}")

    # ---- 1. results_table.csv ---------------------------------------------
    rt_path = out_dir / "results_table.csv"
    assert rt_path.is_file(), "results_table.csv missing"
    rt = pd.read_csv(rt_path)
    assert list(rt.columns) == ["config", "accuracy", "macro_f1"], rt.columns
    assert len(rt) == 3, f"expected 3 config rows, got {len(rt)}"
    assert list(rt["config"]) == SMOKE_CONFIGS, list(rt["config"])
    for col in ("accuracy", "macro_f1"):
        for cell in rt[col]:
            _check_mean_std(cell)
    assert not rt.isna().any().any(), "NaN in results_table.csv"

    # ---- 2. remedy_table.csv ----------------------------------------------
    rem_path = out_dir / "remedy_table.csv"
    assert rem_path.is_file(), "remedy_table.csv missing"
    rem = pd.read_csv(rem_path)
    assert len(rem) == 3, f"expected 3 rows in remedy_table.csv, got {len(rem)}"
    for col in ("H_alpha", "alpha_tvar", "S_tau", "rho_last", "perm_null_z"):
        assert col in rem.columns, f"missing column {col}"
        vals = rem[col].to_numpy(dtype=float)
        assert all(math.isfinite(v) for v in vals), f"non-finite in {col}: {vals}"

    # ---- 3. significance.csv ----------------------------------------------
    sig_path = out_dir / "significance.csv"
    assert sig_path.is_file(), "significance.csv missing"
    sig = pd.read_csv(sig_path)
    assert list(sig["config"]) == ["full_r2", "full_r4_lstm"], sig
    for p in sig["t_paired_p"]:
        p = float(p)
        assert math.isnan(p) or 0.0 <= p <= 1.0, f"t-test p out of range: {p}"

    # ---- 4. diagnostics/*.jsonl + protocol_report.json ---------------------
    diag_dir = out_dir / "diagnostics"
    for name in SMOKE_CONFIGS:
        for seed in (42, 43):
            f = diag_dir / f"{name}_seed{seed}.jsonl"
            assert f.is_file(), f"missing {f}"
            lines = f.read_text().strip().splitlines()
            recs = [json.loads(line) for line in lines]
            epochs = [r for r in recs if "epoch" in r]
            finals = [r for r in recs if "final" in r]
            assert len(epochs) == 2, f"{f.name}: expected 2 epoch lines, got {len(epochs)}"
            assert len(finals) == 1, f"{f.name}: expected 1 final line, got {len(finals)}"
            assert math.isfinite(float(finals[0]["final"]["accuracy"])), finals[0]
            for rec in epochs:
                assert "rho" in rec and "val_acc" in rec, rec
                assert math.isfinite(float(rec["train_loss"])), rec
                if name == "full_r2":  # SPEC v2: R2 epochs log L_balance
                    assert "L_balance" in rec, rec
                    assert math.isfinite(float(rec["L_balance"])), rec
    proto_path = out_dir / "protocol_report.json"
    assert proto_path.is_file(), "protocol_report.json missing"
    proto = json.loads(proto_path.read_text())
    assert proto["seeds"] == [42, 43] and proto["epochs"] == 2, proto["seeds"]
    assert "dataset_report" in proto and "per_class" in proto["dataset_report"]

    print("[smoke] OK: all 4 output families present, shapes and values valid.")
    print(f"[smoke] outputs at {out_dir}")


if __name__ == "__main__":
    main()
