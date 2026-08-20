#!/usr/bin/env python3
"""Cell D bearing-level vs run-level summary (work-order A.3).

Reads existing run-level ``CellD_Paderborn_ATSF`` and the new bearing-level
``CellD_Paderborn_ATSF_bearing``. Never writes into those directories.

Usage (from the repository root)::

    python scripts/summary_bearing_vs_run.py
    python scripts/summary_bearing_vs_run.py --smoke

Output: ``output/experiments/CellD_Paderborn_ATSF_bearing/summary_bearing_vs_run.md``
when the bearing dir exists; otherwise
``output/analysis_n30/summary_bearing_vs_run.md`` with a "not run yet" note.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atsf_dcg.paths import EXPERIMENTS_ROOT, OUTPUT_ROOT  # noqa: E402
from atsf_dcg.run_experiments import _slug  # noqa: E402


def _per_seed(diag: Path, config: str) -> dict[int, dict]:
    out = {}
    slug = _slug(config)
    for p in sorted(diag.glob(f"{slug}_seed*.jsonl")):
        seed = int(p.stem.rsplit("_seed", 1)[-1])
        final = None
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "final" in row:
                    final = row["final"]
        if not isinstance(final, dict):
            continue
        out[seed] = {
            "accuracy": float(final["accuracy"]),
            "macro_f1": float(final["macro_f1"]),
            "S_tau": final.get("S_tau"),
            "rho_last": final.get("rho_last"),
        }
    return out


def _pm(vals) -> str:
    a = np.asarray(list(vals), dtype=np.float64)
    if a.size == 0:
        return "n/a"
    if a.size == 1:
        return f"{a[0]:.4f}±0.0000"
    return f"{a.mean():.4f}±{a.std(ddof=1):.4f}"


def _paired(a, b) -> dict:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    n = d.size
    md = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    dz = md / sd if sd and sd > 0 else float("nan")
    t_p = float(stats.ttest_rel(a, b).pvalue) if n > 1 else float("nan")
    try:
        w_p = float(stats.wilcoxon(d).pvalue) if n > 1 else float("nan")
    except Exception:
        w_p = float("nan")
    return {"n": n, "mean_diff": md, "cohens_dz": dz, "t_p": t_p,
            "wilcoxon_p": w_p}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--bearing-dir", type=str, default=None)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir) if args.run_dir else (
        EXPERIMENTS_ROOT / "CellD_Paderborn_ATSF")
    bear_dir = Path(args.bearing_dir) if args.bearing_dir else (
        EXPERIMENTS_ROOT / "CellD_Paderborn_ATSF_bearing")

    if bear_dir.is_dir() and (bear_dir / "diagnostics").is_dir():
        out_md = bear_dir / "summary_bearing_vs_run.md"
    else:
        out_md = OUTPUT_ROOT / "analysis_n30" / "summary_bearing_vs_run.md"
        out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Cell D: bearing-level split vs run-level split",
        "",
        f"Run-level dir: `{run_dir}`",
        f"Bearing-level dir: `{bear_dir}`",
        "",
        "Code labels KB as **BothRings**. The supplement may say outer-ring "
        "for KB; do not change code labels.",
        "",
    ]
    if not bear_dir.is_dir() or not (bear_dir / "diagnostics").is_dir():
        lines += [
            "Bearing-level experiment has not been run yet. From `code01/`:",
            "",
            "```bash",
            "python -m atsf_dcg.run_experiments --synthetic --smoke "
            "--cell replication --dataset paderborn --arch atsf "
            "--split-unit bearing --configs full",
            "",
            "python -m atsf_dcg.run_experiments --cell replication "
            "--dataset paderborn --arch atsf --split-unit bearing \\",
            "  --seeds 42 43 44 45 46 47 48 49 50 51 \\",
            "  --degradation --log-epoch-indicators \\",
            "  --out-dir output/experiments/CellD_Paderborn_ATSF_bearing",
            "```",
            "",
            "Do **not** point `--out-dir` at `CellD_Paderborn_ATSF`.",
            "",
        ]
        out_md.write_text("\n".join(lines), encoding="utf-8")
        print(f"[summary_bearing_vs_run] wrote stub {out_md}")
        if args.smoke:
            assert out_md.is_file()
        return

    proto = {}
    ppath = bear_dir / "protocol_report.json"
    if ppath.is_file():
        proto = json.loads(ppath.read_text(encoding="utf-8"))
    ds = proto.get("dataset_report") or proto
    part = ds.get("bearing_partition") or {}
    lines += [
        "## Protocol (bearing split)",
        "",
        f"- split_unit: `{ds.get('split_unit', proto.get('split_unit'))}`",
        f"- train bearings (n={len(part.get('train', []))}): "
        f"{part.get('train')}",
        f"- val bearings (n={len(part.get('val', []))}): {part.get('val')}",
        f"- test bearings (n={len(part.get('test', []))}): {part.get('test')}",
        f"- KB 1/1/1: `{ds.get('kb_degenerate_111')}`",
        f"- excluded files: `{ds.get('excluded_files')}`",
        "",
    ]

    configs = []
    rt = bear_dir / "results_table.csv"
    if rt.is_file():
        configs = list(pd.read_csv(rt)["config"])
    if not configs:
        configs = ["full", "w/o_spectral", "w/o_temporal", "w/o_fusion",
                   "w/o_gating", "full_r1", "full_r2_gumbel"]

    lines += ["## Clean accuracy / F1 (bearing split)", "",
              "| config | acc mean±SD | F1 mean±SD | n |",
              "|---|---|---|---|"]
    bear_diag = bear_dir / "diagnostics"
    run_diag = run_dir / "diagnostics" if run_dir.is_dir() else None
    f1_rev = None
    for cfg in configs:
        seeds = _per_seed(bear_diag, cfg)
        acc = [v["accuracy"] for v in seeds.values()]
        f1 = [v["macro_f1"] for v in seeds.values()]
        lines.append(f"| {cfg} | {_pm(acc)} | {_pm(f1)} | {len(seeds)} |")
    lines.append("")

    wo = _per_seed(bear_diag, "w/o_spectral")
    fu = _per_seed(bear_diag, "full")
    common = sorted(set(wo) & set(fu))
    if len(common) >= 2:
        acc_st = _paired([wo[s]["accuracy"] for s in common],
                         [fu[s]["accuracy"] for s in common])
        f1_st = _paired([wo[s]["macro_f1"] for s in common],
                        [fu[s]["macro_f1"] for s in common])
        f1_rev = f1_st["mean_diff"] < 0
        lines += [
            "## Paired w/o_spectral − full (bearing split)",
            "",
            f"- n={acc_st['n']}",
            f"- accuracy: Δ={acc_st['mean_diff']*100:+.3f} pp, "
            f"t p={acc_st['t_p']:.4g}, Wilcoxon p={acc_st['wilcoxon_p']:.4g}, "
            f"dz={acc_st['cohens_dz']:+.3f}",
            f"- macro-F1: Δ={f1_st['mean_diff']*100:+.3f} pp, "
            f"t p={f1_st['t_p']:.4g}, Wilcoxon p={f1_st['wilcoxon_p']:.4g}, "
            f"dz={f1_st['cohens_dz']:+.3f}",
            "",
            "**F1 reversal (w/o_spectral < full) persists:** "
            f"{'yes' if f1_rev else 'no'}.",
            "",
        ]

    deg = bear_dir / "degradation_table.csv"
    if deg.is_file():
        lines += ["## Degradation matrix (bearing split)", "",
                  "```", deg.read_text(encoding="utf-8").strip(), "```", ""]
    else:
        lines += ["## Degradation matrix", "",
                  "Not present (`degradation_table.csv` missing).", ""]

    lines += ["## S(0.9) / ρ (bearing split, config `full`)", ""]
    full_s = _per_seed(bear_diag, "full")
    s_tau = [v["S_tau"] for v in full_s.values() if v["S_tau"] is not None]
    rho = [v["rho_last"] for v in full_s.values() if v["rho_last"] is not None]
    lines += [
        f"- S(0.9) mean±SD: {_pm(s_tau)} (n={len(s_tau)})",
        f"- ρ mean±SD: {_pm(rho)} (n={len(rho)})",
        "",
    ]
    if run_diag and run_diag.is_dir():
        rfull = _per_seed(run_diag, "full")
        rwo = _per_seed(run_diag, "w/o_spectral")
        rc = sorted(set(rwo) & set(rfull))
        if rc:
            rf1 = _paired([rwo[s]["macro_f1"] for s in rc],
                          [rfull[s]["macro_f1"] for s in rc])
            lines += [
                "## Run-level reference (existing Cell D, not overwritten)",
                "",
                f"- w/o_spectral − full F1 Δ={rf1['mean_diff']*100:+.3f} pp "
                f"(n={rf1['n']})",
                "",
            ]

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[summary_bearing_vs_run] wrote {out_md}")
    if args.smoke:
        assert out_md.is_file()


if __name__ == "__main__":
    main()
