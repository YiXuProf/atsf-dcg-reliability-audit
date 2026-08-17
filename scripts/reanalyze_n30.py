#!/usr/bin/env python3
"""Merge n=10 (seeds 42-51) with n=30 extension (seeds 52-71) and recompute
paired tests / Holm / TOST / D1-D2 rates.

Does not train. Does not write into existing Cell* / nppad_atsf_full dirs.

Usage (from the repository root)::

    python scripts/reanalyze_n30.py --smoke
    python scripts/reanalyze_n30.py

Outputs (new directory only)::

    output/analysis_n30/tableII_n30.csv
    output/analysis_n30/tableIII_n30.csv
    output/analysis_n30/tableIV_n30.csv
    output/analysis_n30/tableVI_cellO_n30.csv
    output/analysis_n30/tableVI_cellA_n30.csv
    output/analysis_n30/significance_family19_n30.csv
    output/analysis_n30/d1_d2_recalibration.csv
    output/analysis_n30/tier_changes.md
    output/analysis_n30/DEGRADATION_N10_ONLY.md   (if new seeds have no deg)

n=30 rows exist only after the n30ext_* experiment dirs are populated.
Until then the script still runs on seeds 42-51 and records realized n.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atsf_dcg.paths import EXPERIMENTS_ROOT, OUTPUT_ROOT  # noqa: E402
from atsf_dcg.manuscript_tables import ABLATION_CONFIGS, REMEDY_CONFIGS  # noqa: E402
from atsf_dcg.run_experiments import (  # noqa: E402
    _bh_adjust, _holm_adjust, _slug, build_configs,
)

PAT = re.compile(r"^(?P<slug>.+)_seed(?P<seed>\d+)\.jsonl$")

N10_SEEDS = list(range(42, 52))
N30_NEW_SEEDS = list(range(52, 72))
PRIORITY = ["full", "w/o_spectral", "w/o_gating", "full_r2_gumbel"]
OPTIONAL_P1 = ["full_r3_sinc", "full_r1_r3"]
D1_S_TAU = 0.15
D2_RHO = 1.5

# Pre-registered 19-comparison family = full-grid configs vs "full"
FAMILY19 = [c.name for c in build_configs() if c.name != "full"]


def _load_jsonl_dir(diag: Path) -> dict[str, dict[int, dict]]:
    """{config_name: {seed: {accuracy, macro_f1, S_tau, rho_last, ...}}}."""
    slug2name = {_slug(c.name): c.name for c in build_configs()}
    # replication-only names
    slug2name.setdefault("tsnet_vanilla", "tsnet_vanilla")
    out: dict[str, dict[int, dict]] = {}
    if not diag.is_dir():
        return out
    for p in sorted(diag.glob("*_seed*.jsonl")):
        m = PAT.match(p.name)
        if not m:
            continue
        seed = int(m.group("seed"))
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
        name = slug2name.get(m.group("slug"), m.group("slug"))
        rec = {
            "accuracy": float(final["accuracy"]) if final.get("accuracy") is not None else float("nan"),
            "macro_f1": float(final["macro_f1"]) if final.get("macro_f1") is not None else float("nan"),
            "S_tau": _maybe_float(final.get("S_tau")),
            "rho_last": _maybe_float(final.get("rho_last")),
            "H_alpha": _maybe_float(final.get("H_alpha")),
            "alpha_tvar": _maybe_float(final.get("alpha_tvar")),
            "perm_null_z": _maybe_float((final.get("perm_null") or {}).get("z")
                                        if isinstance(final.get("perm_null"), dict)
                                        else None),
            "has_degradation": isinstance(final.get("degradation"), dict)
            and bool(final.get("degradation")),
        }
        out.setdefault(name, {})[seed] = rec
    return out


def _maybe_float(v) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def merge_dirs(primary: Path, extra: Path | None) -> dict[str, dict[int, dict]]:
    """Union of diagnostics; extra seeds win on collision (should not collide)."""
    data = _load_jsonl_dir(primary / "diagnostics")
    if extra is not None and extra.is_dir():
        more = _load_jsonl_dir(extra / "diagnostics")
        for name, seeds in more.items():
            data.setdefault(name, {}).update(seeds)
    return data


def mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    if a.size == 1:
        return float(a[0]), 0.0
    return float(a.mean()), float(a.std(ddof=1))


def fmt_pm(vals: list[float]) -> str:
    m, s = mean_std(vals)
    if not np.isfinite(m):
        return ""
    return f"{m:.4f}±{s:.4f}"


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    n = int(d.size)
    md = float(d.mean()) if n else float("nan")
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    if n > 1 and sd > 0:
        half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
        dz = md / sd
        half90 = float(stats.t.ppf(0.95, n - 1)) * sd / math.sqrt(n)
    else:
        half = half90 = dz = float("nan")
    try:
        t_p = float(stats.ttest_rel(a, b).pvalue) if n > 1 else float("nan")
    except Exception:
        t_p = float("nan")
    try:
        w_p = float(stats.wilcoxon(d).pvalue) if n > 1 else float("nan")
    except Exception:
        w_p = float("nan")
    shapiro = float("nan")
    if n >= 3:
        try:
            shapiro = float(stats.shapiro(d).pvalue)
        except Exception:
            shapiro = float("nan")
    return {
        "n": n, "mean_diff": md, "sd_diff": sd,
        "ci95_lo": md - half, "ci95_hi": md + half,
        "ci90_lo": md - half90, "ci90_hi": md + half90,
        "cohens_dz": dz, "t_p": t_p, "wilcoxon_p": w_p, "shapiro_p": shapiro,
    }


def tost(a: np.ndarray, b: np.ndarray, eps: float) -> dict:
    d = a - b
    n = int(d.size)
    if n < 2:
        return {"eps": eps, "tost_p": float("nan"), "equivalent": False,
                "ci90_inside": False}
    md, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / math.sqrt(n) if sd > 0 else float("nan")
    if not np.isfinite(se) or se == 0:
        return {"eps": eps, "tost_p": float("nan"), "equivalent": False,
                "ci90_inside": abs(md) <= eps}
    t1 = (md + eps) / se
    t2 = (eps - md) / se
    p1 = 1.0 - float(stats.t.cdf(t1, n - 1))
    p2 = 1.0 - float(stats.t.cdf(t2, n - 1))
    p = max(p1, p2)
    half90 = float(stats.t.ppf(0.95, n - 1)) * se
    inside = (md - half90) >= -eps and (md + half90) <= eps
    return {"eps": eps, "tost_p": p, "equivalent": bool(p < 0.05),
            "ci90_inside": bool(inside)}


def tier(t_p: float, holm_p: float) -> str:
    if np.isfinite(holm_p) and holm_p < 0.05:
        return "robust"
    if np.isfinite(t_p) and t_p < 0.05:
        return "suggestive"
    if np.isfinite(t_p) and t_p < 0.10:
        return "marginal"
    return "n.s."


def results_table(data: dict[str, dict[int, dict]], names: list[str]
                  ) -> pd.DataFrame:
    rows = []
    for name in names:
        seeds = data.get(name, {})
        if not seeds:
            continue
        acc = [seeds[s]["accuracy"] for s in sorted(seeds)]
        f1 = [seeds[s]["macro_f1"] for s in sorted(seeds)]
        rows.append({
            "config": name, "n": len(seeds),
            "seeds": ",".join(str(s) for s in sorted(seeds)),
            "accuracy": fmt_pm(acc), "macro_f1": fmt_pm(f1),
            "accuracy_mean": mean_std(acc)[0],
            "accuracy_std": mean_std(acc)[1],
            "macro_f1_mean": mean_std(f1)[0],
            "macro_f1_std": mean_std(f1)[1],
        })
    return pd.DataFrame(rows)


def remedy_table(data: dict[str, dict[int, dict]], names: list[str]
                 ) -> pd.DataFrame:
    rows = []
    for name in names:
        seeds = data.get(name, {})
        if not seeds:
            continue
        acc = [seeds[s]["accuracy"] for s in sorted(seeds)]
        row = {"config": name, "n": len(seeds),
               "accuracy_mean": mean_std(acc)[0]}
        for m in ("H_alpha", "alpha_tvar", "S_tau", "rho_last", "perm_null_z"):
            vals = [seeds[s][m] for s in sorted(seeds)]
            row[m] = mean_std(vals)[0]
        rows.append(row)
    return pd.DataFrame(rows)


def vs_full_rows(data: dict[str, dict[int, dict]], names: list[str],
                 metric: str) -> list[dict]:
    full = data.get("full", {})
    rows = []
    for name in names:
        if name == "full":
            continue
        cfg = data.get(name, {})
        common = sorted(set(full) & set(cfg))
        if len(common) < 2:
            continue
        a = np.asarray([cfg[s][metric] for s in common], dtype=np.float64)
        b = np.asarray([full[s][metric] for s in common], dtype=np.float64)
        st = paired(a, b)
        t05 = tost(a, b, 0.005)
        t10 = tost(a, b, 0.010)
        rows.append({
            "config": name, "metric": metric, "n_paired": st["n"],
            "cfg_mean": float(a.mean()), "full_mean": float(b.mean()),
            **{k: st[k] for k in ("mean_diff", "ci95_lo", "ci95_hi",
                                  "ci90_lo", "ci90_hi", "cohens_dz",
                                  "t_p", "wilcoxon_p", "shapiro_p")},
            "tost_p_0.5pp": t05["tost_p"],
            "tost_eq_0.5pp": t05["equivalent"],
            "tost_p_1.0pp": t10["tost_p"],
            "tost_eq_1.0pp": t10["equivalent"],
        })
    return rows


def holm_family(rows: list[dict], p_col: str, out_col: str) -> None:
    pv = np.asarray([r[p_col] for r in rows], dtype=np.float64)
    ok = np.isfinite(pv)
    adj = np.full(len(pv), np.nan)
    if ok.any():
        adj[ok] = _holm_adjust(pv[ok])
    for r, h in zip(rows, adj):
        r[out_col] = float(h) if np.isfinite(h) else float("nan")


def d1_d2_table(by_cell: dict[str, dict[str, dict[int, dict]]]) -> pd.DataFrame:
    """D1 (S_tau > 0.15) and D2 (rho > 1.5) on config 'full'.

    (i) balanced: seeds 42-51, five cells (50 runs).
    (ii) all available seeds in each cell (O/A may be 30).
    """
    rows = []
    for mode, seed_ok in (
        ("balanced_50", lambda s, cell: s in N10_SEEDS),
        ("all_available", lambda s, cell: True),
    ):
        s_vals, r_vals, cells_used = [], [], []
        for cell, data in by_cell.items():
            full = data.get("full", {})
            for seed, rec in full.items():
                if not seed_ok(seed, cell):
                    continue
                s_vals.append(rec["S_tau"])
                r_vals.append(rec["rho_last"])
                cells_used.append(cell)
        s = np.asarray(s_vals, dtype=np.float64)
        r = np.asarray(r_vals, dtype=np.float64)
        ok = np.isfinite(s) & np.isfinite(r)
        n = int(ok.sum())
        d1 = int((s[ok] > D1_S_TAU).sum()) if n else 0
        d2 = int((r[ok] > D2_RHO).sum()) if n else 0
        both = int(((s[ok] > D1_S_TAU) & (r[ok] > D2_RHO)).sum()) if n else 0
        rows.append({
            "subset": mode,
            "n_runs": n,
            "cells": ",".join(sorted(set(cells_used))),
            "n_D1_S_gt_0.15": d1,
            "rate_D1": (d1 / n) if n else float("nan"),
            "n_D2_rho_gt_1.5": d2,
            "rate_D2": (d2 / n) if n else float("nan"),
            "n_D1_and_D2": both,
            "rate_D1_and_D2": (both / n) if n else float("nan"),
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="run on existing n=10 diagnostics only; still writes "
                         "analysis_n30/ (realized n recorded per row)")
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args(argv)
    out = Path(args.out_dir) if args.out_dir else (
        OUTPUT_ROOT / "analysis_n30")
    out.mkdir(parents=True, exist_ok=True)

    full_grid = EXPERIMENTS_ROOT / "nppad_atsf_full"
    cell_o = EXPERIMENTS_ROOT / "CellO_NPPAD_ATSF"
    cell_a = EXPERIMENTS_ROOT / "CellA_NPPAD_TimesNet"
    n30_o = EXPERIMENTS_ROOT / "n30ext_nppad_atsf"
    n30_a = EXPERIMENTS_ROOT / "n30ext_nppad_tsnet"

    data_full = merge_dirs(full_grid, n30_o if n30_o.is_dir() else None)
    data_o = merge_dirs(cell_o, n30_o if n30_o.is_dir() else None)
    # Cell O folder has no full_r2_gumbel; borrow n=10 from the full grid
    if "full_r2_gumbel" in data_full:
        data_o.setdefault("full_r2_gumbel", {}).update(
            {s: rec for s, rec in data_full["full_r2_gumbel"].items()
             if s in N10_SEEDS and s not in data_o.get("full_r2_gumbel", {})})
        if n30_o.is_dir() and "full_r2_gumbel" in data_full:
            data_o.setdefault("full_r2_gumbel", {}).update(
                {s: rec for s, rec in data_full["full_r2_gumbel"].items()
                 if s in N30_NEW_SEEDS})
    data_a = merge_dirs(cell_a, n30_a if n30_a.is_dir() else None)

    names_full = [c.name for c in build_configs()]
    names_o = ["full", "w/o_spectral", "w/o_temporal", "w/o_fusion",
               "w/o_dynamic_gating", "w/o_gating", "full_r2_gumbel"]
    names_a = ["full", "w/o_spectral", "w/o_temporal", "w/o_fusion",
               "w/o_gating", "full_r2_gumbel", "tsnet_vanilla"]

    t2 = results_table(data_full, list(ABLATION_CONFIGS))
    t3 = results_table(data_full, list(REMEDY_CONFIGS))
    t4 = remedy_table(data_full, names_full)
    t6o = results_table(data_o, names_o)
    t6a = results_table(data_a, names_a)
    t2.to_csv(out / "tableII_n30.csv", index=False)
    t3.to_csv(out / "tableIII_n30.csv", index=False)
    t4.to_csv(out / "tableIV_n30.csv", index=False)
    t6o.to_csv(out / "tableVI_cellO_n30.csv", index=False)
    t6a.to_csv(out / "tableVI_cellA_n30.csv", index=False)

    # Holm family of 19: n=30 p for extended configs, n=10 p otherwise
    family_rows = []
    n10_only = merge_dirs(full_grid, None)
    for name in FAMILY19:
        src = data_full if name in PRIORITY + OPTIONAL_P1 else n10_only
        family_rows.extend(vs_full_rows(src, [name], "accuracy"))
        family_rows.extend(vs_full_rows(src, [name], "macro_f1"))
    acc_rows = [r for r in family_rows if r["metric"] == "accuracy"]
    f1_rows = [r for r in family_rows if r["metric"] == "macro_f1"]
    holm_family(acc_rows, "t_p", "t_holm_p")
    holm_family(acc_rows, "wilcoxon_p", "wilcoxon_holm_p")
    holm_family(f1_rows, "t_p", "t_holm_p")
    holm_family(f1_rows, "wilcoxon_p", "wilcoxon_holm_p")
    for group in (acc_rows, f1_rows):
        pv = np.asarray([r["t_p"] for r in group], dtype=np.float64)
        ok = np.isfinite(pv)
        bh = np.full(len(pv), np.nan)
        if ok.any():
            bh[ok] = _bh_adjust(pv[ok])
        for r, b in zip(group, bh):
            r["t_bh_p"] = float(b) if np.isfinite(b) else float("nan")
            r["tier"] = tier(r["t_p"], r["t_holm_p"])
    fam = pd.DataFrame(acc_rows + f1_rows)
    fam.to_csv(out / "significance_family19_n30.csv", index=False)

    # n=10 baseline tiers (accuracy family) for tier_changes.md
    n10_rows = vs_full_rows(n10_only, FAMILY19, "accuracy")
    holm_family(n10_rows, "t_p", "t_holm_p")
    for r in n10_rows:
        r["tier"] = tier(r["t_p"], r["t_holm_p"])
    n10_map = {r["config"]: r for r in n10_rows}
    n30_map = {r["config"]: r for r in acc_rows}

    lines = [
        "# Tier changes (n=10 vs merged n)",
        "",
        "Tiers: **robust** = Holm p < 0.05; **suggestive** = raw t p < 0.05 "
        "but Holm n.s.; **marginal** = 0.05 ≤ t p < 0.10; **n.s.** otherwise.",
        "Holm is the pre-registered 19-comparison accuracy family vs `full`.",
        "Comparisons not in the n=30 expansion keep their n=10 p-values; "
        "Holm uses each row's own p (realized n is in the CSV).",
        "",
    ]
    changes = []
    for name in FAMILY19:
        a, b = n10_map.get(name), n30_map.get(name)
        if not a or not b:
            continue
        if a["tier"] != b["tier"]:
            changes.append(
                f"- `{name}`: {a['tier']} (n={a['n_paired']}, t_p="
                f"{a['t_p']:.4g}, holm={a['t_holm_p']:.4g}) → {b['tier']} "
                f"(n={b['n_paired']}, t_p={b['t_p']:.4g}, "
                f"holm={b['t_holm_p']:.4g})")
    if changes:
        lines.append("## Changes")
        lines.extend(changes)
    else:
        lines.append("## Changes")
        lines.append("None (including the case where n=30 dirs are still empty "
                     "and every row is still n=10).")
    lines.append("")
    lines.append("## Realized n (accuracy vs full)")
    for name in FAMILY19:
        b = n30_map.get(name)
        if b:
            lines.append(f"- `{name}`: n={b['n_paired']}  tier={b['tier']}")
    (out / "tier_changes.md").write_text("\n".join(lines) + "\n",
                                         encoding="utf-8")

    by_cell = {
        "O": data_o,
        "A": data_a,
        "B": merge_dirs(EXPERIMENTS_ROOT / "CellB_TEP_ATSF", None),
        "C": merge_dirs(EXPERIMENTS_ROOT / "CellC_TEP_TimesNet", None),
        "D": merge_dirs(EXPERIMENTS_ROOT / "CellD_Paderborn_ATSF", None),
    }
    d1d2 = d1_d2_table(by_cell)
    d1d2.to_csv(out / "d1_d2_recalibration.csv", index=False)

    # degradation note
    new_has_deg = False
    for folder in (n30_o, n30_a):
        if not folder.is_dir():
            continue
        blob = _load_jsonl_dir(folder / "diagnostics")
        for seeds in blob.values():
            if any(rec.get("has_degradation") for rec in seeds.values()):
                new_has_deg = True
    if not n30_o.is_dir() and not n30_a.is_dir():
        (out / "DEGRADATION_N10_ONLY.md").write_text(
            "n30ext_* directories are not present yet. Degradation tables "
            "remain those already computed on seeds 42–51 (n=10).\n",
            encoding="utf-8")
    elif not new_has_deg:
        (out / "DEGRADATION_N10_ONLY.md").write_text(
            "The n=30 extension runs have no `final.degradation` records. "
            "Degradation tables remain n=10 (seeds 42–51).\n",
            encoding="utf-8")

    missing = []
    if not n30_o.is_dir():
        missing.append(str(n30_o))
    if not n30_a.is_dir():
        missing.append(str(n30_a))
    print(f"[reanalyze_n30] wrote {out}")
    if missing:
        print("[reanalyze_n30] n=30 dirs not found (realized n stays 10):")
        for p in missing:
            print(f"  missing {p}")
    if args.smoke:
        assert (out / "tableII_n30.csv").is_file()
        assert (out / "tableIII_n30.csv").is_file()
        assert (out / "tableIV_n30.csv").is_file()
        assert (out / "tier_changes.md").is_file()
        print("[reanalyze_n30] smoke OK")


if __name__ == "__main__":
    main()
