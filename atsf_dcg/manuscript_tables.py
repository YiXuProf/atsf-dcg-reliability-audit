"""Write CSVs numbered as in the RESS manuscript + supplement.

Manuscript (RESS_manuscript.docx)
    I     NPPAD per-class run-level split
    II    ablation-as-diagnosis (acc / F1 + tests vs full)
    III   remedy evaluation (same stats; last rows = controls)
    IV    mechanism indicators H(α), Var_t(α), S(τ), ρ, perm-null z
    V     planned control contrasts (Experiments A/B) + TOST
    VI    five-cell recurrence (per-cell results + significance)
    VII   D1–D5 rules (prose; not a CSV)

Supplement (RESS_supplementary.docx)
    S1–S5   degradation matrices, Cells O / A / B / C / D
            (Δ vs clean recomputed from per-seed finals; see S.1 table note)
    S6      Cell O v7 rerun vs original-paper grid
    S7      Spearman: epoch-3 indicators vs clean→snr10 drop
    S8a     Cell O per-seed Acc / F1 (%)
    S8–S11  Cells A / B / C / D per-seed Acc / F1 (%)
    S12–S15 full-model per-seed ρ / H(α) / Var_t(α) / perm-null z
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .degradation import DEGRADATION_NAMES
from .paths import (
    EXPERIMENTS_ROOT,
    PAPER_TABLES,
    TABLES_ROOT,
)

# manuscript Table II rows (ablations + full baseline)
ABLATION_CONFIGS = [
    "full", "w/o_spectral", "w/o_temporal", "w/o_fusion",
    "w/o_dynamic_gating", "w/o_gating",
]
# manuscript Table III: remedies, then three controls
REMEDY_CONFIGS = [
    "full_r1", "full_r3_stft", "full_r3_sinc", "full_r1_r3",
    "full_r2", "full_r2_gumbel", "full_r4_sparsemax", "full_r4_entmax",
    "full_r4_lstm", "full_r1_r2_r3", "full_all",
    "r1_w/o_spectral", "full_fixed_global", "full_fixed_class",
]

_CELL_TAGS = [
    ("O", "cellO_key"),
    ("A", "nppad_tsnet"),
    ("B", "tep_atsf"),
    ("C", "tep_tsnet"),
    ("D", "paderborn_atsf"),
]

_S_DEGRADATION = {
    "S1": ("O", "cellO_key"),
    "S2": ("A", "nppad_tsnet"),
    "S3": ("B", "tep_atsf"),
    "S4": ("C", "tep_tsnet"),
    "S5": ("D", "paderborn_atsf"),
}

_S_PERSEED_ACC = {
    "S8a": "O",
    "S8": "A",
    "S9": "B",
    "S10": "C",
    "S11": "D",
}

_RETIRE = (
    "Table_S1_per_class.csv",
    "Table_S2_perm_null.csv",
    "Table_S3_rho_curve.csv",
    "Table_S4_nppad_degradation.csv",
    "Table_S5_tost_1pp.csv",
    "Table_II_nppad_atsf_results.csv",
    "Table_III_diagnostics.csv",
    "Table_IV_significance.csv",
    "Table_S15_per_seed_raw.csv",
)


def _unslug_config(name: str) -> str:
    s = str(name)
    if s.startswith("wo_"):
        return "w/o_" + s[3:]
    return s


def _copy(src: Path, dest: Path) -> Path | None:
    if not src.is_file():
        print(f"  [skip] missing {src}", flush=True)
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())
    print(f"  paper <- {dest.name}", flush=True)
    return dest


def retire_misnamed(paper_dir: Path) -> None:
    """Remove CSVs that used the pre-alignment S/III/IV numbering."""
    gone = 0
    for name in _RETIRE:
        p = paper_dir / name
        if p.is_file():
            p.unlink()
            gone += 1
    for pat in (
        "Table_S8_cellO_*.csv", "Table_S9_cellA_*.csv",
        "Table_S10_cellB_*.csv", "Table_S11_cellC_*.csv",
        "Table_S12_cellD_*.csv", "Table_S13_epoch_indicators_*.csv",
        "Table_S14_finals_*.csv",
    ):
        for p in paper_dir.glob(pat):
            p.unlink()
            gone += 1
    if gone:
        print(f"  [retire] removed {gone} misnumbered CSV(s)", flush=True)


def _join_results_significance(results: pd.DataFrame, sig: pd.DataFrame,
                               names: list[str]) -> pd.DataFrame:
    sig = sig.copy()
    if "config" in sig.columns:
        sig = sig.set_index("config")
    rows = []
    for name in names:
        hit = results[results["config"] == name]
        if hit.empty:
            continue
        row = {"config": name,
               "accuracy": hit.iloc[0].get("accuracy", ""),
               "macro_f1": hit.iloc[0].get("macro_f1", "")}
        if name != "full" and name in sig.index:
            s = sig.loc[name]
            for col in s.index:
                if col == "config":
                    continue
                row[col] = s[col]
        rows.append(row)
    return pd.DataFrame(rows)


def write_table_ii_iii(full_dir: Path, paper: Path) -> None:
    rt = full_dir / "results_table.csv"
    sg = full_dir / "significance.csv"
    if not rt.is_file():
        print(f"  [skip] missing {rt}", flush=True)
        return
    results = pd.read_csv(rt)
    sig = pd.read_csv(sg) if sg.is_file() else pd.DataFrame()
    _join_results_significance(results, sig, ABLATION_CONFIGS).to_csv(
        paper / "Table_II_ablation.csv", index=False)
    print("  paper <- Table_II_ablation.csv", flush=True)
    _join_results_significance(results, sig, REMEDY_CONFIGS).to_csv(
        paper / "Table_III_remedies.csv", index=False)
    print("  paper <- Table_III_remedies.csv", flush=True)
    if sg.is_file():
        _copy(sg, paper / "Table_II_III_significance.csv")


def write_table_iv(full_dir: Path, paper: Path) -> None:
    _copy(full_dir / "remedy_table.csv",
          paper / "Table_IV_mechanism_indicators.csv")
    _copy(full_dir / "perm_null_summary.csv",
          paper / "Table_IV_perm_null_summary.csv")


def write_table_vi(paper: Path) -> None:
    summary = TABLES_ROOT / "five_cell_summary"
    for lab, tag in _CELL_TAGS:
        _copy(summary / f"results_table_{tag}.csv",
              paper / f"Table_VI_cell{lab}_results.csv")
        _copy(summary / f"significance_{tag}.csv",
              paper / f"Table_VI_cell{lab}_significance.csv")


def _recompute_delta(deg: pd.DataFrame, finals: Path) -> pd.DataFrame:
    """Replace delta_vs_clean_pp using per-seed full-config means (S.1 note)."""
    out = deg.copy()
    if "delta_vs_clean_pp" not in out.columns:
        out["delta_vs_clean_pp"] = np.nan
    if not finals.is_file():
        print(f"  [warn] no finals at {finals}; leaving pipeline Δ", flush=True)
        return out
    f = pd.read_csv(finals)
    cfg = f["config"].map(_unslug_config)
    sub = f[cfg == "full"]
    if sub.empty or "final.accuracy" not in sub.columns:
        return out
    clean = float(sub["final.accuracy"].mean())
    deltas = {"clean": 0.0}
    for name in DEGRADATION_NAMES:
        col = f"final.degradation.{name}"
        if col not in sub.columns:
            continue
        deltas[name] = (float(sub[col].mean()) - clean) * 100.0
    out["delta_vs_clean_pp"] = [
        round(deltas.get(str(r["degradation"]), float("nan")), 2)
        for _, r in out.iterrows()
    ]
    return out


def write_s1_s5(sid: str, paper: Path) -> None:
    lab, tag = _S_DEGRADATION[sid]
    src = TABLES_ROOT / "five_cell_summary" / f"degradation_table_{tag}.csv"
    if not src.is_file():
        print(f"  [skip] missing {src}", flush=True)
        return
    finals = TABLES_ROOT / "per_seed_finals" / f"finals_{tag}.csv"
    if not finals.is_file():
        finals = TABLES_ROOT / "per_seed_finals" / f"finals_replication_{tag}.csv"
    deg = _recompute_delta(pd.read_csv(src), finals)
    dest = paper / f"Table_{sid}_cell{lab}_degradation.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    deg.to_csv(dest, index=False)
    print(f"  paper <- {dest.name} (Δ vs clean from per-seed finals)", flush=True)


def write_s6(paper: Path) -> None:
    """Original 20-config grid vs Cell O key rerun (supplement Table S6)."""
    orig = EXPERIMENTS_ROOT / "nppad_atsf_full" / "results_table.csv"
    rerun = EXPERIMENTS_ROOT / "CellO_NPPAD_ATSF" / "results_table.csv"
    orig_r = EXPERIMENTS_ROOT / "nppad_atsf_full" / "remedy_table.csv"
    rerun_r = EXPERIMENTS_ROOT / "CellO_NPPAD_ATSF" / "remedy_table.csv"
    orig_s = EXPERIMENTS_ROOT / "nppad_atsf_full" / "significance.csv"
    rerun_s = EXPERIMENTS_ROOT / "CellO_NPPAD_ATSF" / "significance.csv"
    if not orig.is_file() or not rerun.is_file():
        print("  [skip] Table S6 needs nppad_atsf_full and CellO results",
              flush=True)
        return

    def _acc_map(path: Path) -> dict[str, str]:
        df = pd.read_csv(path)
        return {str(r["config"]): str(r["accuracy"]) for _, r in df.iterrows()}

    def _metric(path: Path, cfg: str, col: str) -> float:
        if not path.is_file():
            return float("nan")
        df = pd.read_csv(path)
        hit = df[df["config"] == cfg]
        if hit.empty or col not in hit.columns:
            return float("nan")
        v = hit.iloc[0][col]
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    def _pp(a: str, b: str) -> str:
        try:
            d = (float(str(b).split("±")[0]) - float(str(a).split("±")[0])) * 100
            return f"{d:+.2f}pp"
        except (TypeError, ValueError):
            return ""

    oa, ra = _acc_map(orig), _acc_map(rerun)
    rows = []
    for cfg in ABLATION_CONFIGS:
        if cfg not in oa or cfg not in ra:
            continue
        rows.append({
            "metric": f"{cfg} acc",
            "original_paper": oa[cfg],
            "v7_rerun": ra[cfg],
            "deviation": _pp(oa[cfg], ra[cfg]),
        })
    for col, label in (("rho_last", "rho"), ("H_alpha", "H(alpha)"),
                       ("alpha_tvar", "Var_t(alpha)"), ("S_tau", "S(0.9)"),
                       ("perm_null_z", "perm-null z")):
        o = _metric(orig_r, "full", col)
        r = _metric(rerun_r, "full", col)
        rows.append({
            "metric": label,
            "original_paper": o, "v7_rerun": r,
            "deviation": (r - o) if np.isfinite(o) and np.isfinite(r) else "",
        })
    if orig_s.is_file() and rerun_s.is_file():
        osig, rsig = pd.read_csv(orig_s), pd.read_csv(rerun_s)
        for cfg in ("w/o_spectral", "w/o_temporal", "w/o_fusion", "w/o_gating"):
            o = osig[osig["config"] == cfg]
            r = rsig[rsig["config"] == cfg]
            if o.empty or r.empty:
                continue
            rows.append({
                "metric": f"{cfg} Holm p (acc vs full)",
                "original_paper": float(o.iloc[0]["t_holm_p"]),
                "v7_rerun": float(r.iloc[0]["t_holm_p"]),
                "deviation": "",
            })
    dest = paper / "Table_S6_cellO_rerun_vs_original.csv"
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"  paper <- {dest.name}", flush=True)


def write_s7(paper: Path) -> None:
    """Within-cell Spearman of epoch-3 indicators vs clean→snr10 drop (pp).

    Drop is (clean − snr10)×100 so that a larger positive value means a
    larger accuracy loss, matching supplement Table S7.
    """
    from scipy import stats
    rows = []
    for lab, tag in _CELL_TAGS:
        epath = TABLES_ROOT / "epoch_indicators" / f"epoch_indicators_{tag}.csv"
        if not epath.is_file():
            epath = TABLES_ROOT / "epoch_indicators" / f"epoch_indicators_replication_{tag}.csv"
        fpath = TABLES_ROOT / "per_seed_finals" / f"finals_{tag}.csv"
        if not fpath.is_file():
            fpath = TABLES_ROOT / "per_seed_finals" / f"finals_replication_{tag}.csv"
        if not epath.is_file() or not fpath.is_file():
            print(f"  [skip] S7 cell {lab}: missing epoch/finals CSV", flush=True)
            continue
        ep = pd.read_csv(epath)
        fn = pd.read_csv(fpath)
        ep["config"] = ep["config"].map(_unslug_config)
        fn["config"] = fn["config"].map(_unslug_config)
        e3 = ep[(ep["config"] == "full") & (ep["epoch"] == 3)]
        full = fn[fn["config"] == "full"]
        if "final.accuracy" not in full.columns:
            print(f"  [skip] S7 cell {lab}: no final.accuracy", flush=True)
            continue
        drop_col = "final.degradation.gaussian_noise_snr10"
        if drop_col not in full.columns:
            print(f"  [skip] S7 cell {lab}: no snr10 degradation", flush=True)
            continue
        merged = e3.merge(full, on="seed", suffixes=("_e", "_f"))
        drop = (merged["final.accuracy"] - merged[drop_col]) * 100.0
        row = {"cell": lab}
        for src, key in (("rho", "rho@3"), ("s_tau", "s_tau@3"),
                         ("h_alpha", "h@3"), ("val_acc", "val@3")):
            if src not in merged.columns:
                row[key] = float("nan")
                continue
            x = merged[src].astype(float)
            ok = np.isfinite(x) & np.isfinite(drop)
            if ok.sum() < 3:
                row[key] = float("nan")
                continue
            r, p = stats.spearmanr(x[ok], drop[ok])
            row[key] = round(float(r), 2)
            row[f"{key}_p"] = float(p)
        rows.append(row)
    dest = paper / "Table_S7_spearman_epoch3_vs_snr10.csv"
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"  paper <- {dest.name}", flush=True)


def _per_seed_raw() -> pd.DataFrame | None:
    p = TABLES_ROOT / "per_seed_finals" / "per_seed_raw.csv"
    if not p.is_file():
        print(f"  [skip] missing {p}", flush=True)
        return None
    df = pd.read_csv(p)
    df["config"] = df["config"].map(_unslug_config)
    df["cell"] = df["cell"].astype(str)
    return df


def write_s8_s11(sid: str, paper: Path) -> None:
    raw = _per_seed_raw()
    if raw is None:
        return
    cell = _S_PERSEED_ACC[sid]
    sub = raw[raw["cell"] == cell]
    if sub.empty:
        print(f"  [skip] {sid}: no per-seed rows for cell {cell}", flush=True)
        return
    seeds = sorted(int(s) for s in sub["seed"].unique())
    preferred = list(ABLATION_CONFIGS) + list(REMEDY_CONFIGS)
    present = set(sub["config"])
    configs = [c for c in preferred if c in present]
    for c in sub["config"]:
        if c not in configs:
            configs.append(c)
    rows = []
    for cfg in configs:
        block = sub[sub["config"] == cfg]
        for metric, col in (("Acc", "accuracy"), ("F1", "macro_f1")):
            row = {"config": cfg, "metric": metric}
            vals = []
            for s in seeds:
                hit = block[block["seed"] == s]
                if hit.empty or pd.isna(hit.iloc[0][col]):
                    row[str(s)] = ""
                    continue
                v = float(hit.iloc[0][col]) * 100.0
                row[str(s)] = round(v, 2)
                vals.append(v)
            if vals:
                m = float(np.mean(vals))
                sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                row["Mean ± SD"] = f"{m:.2f} ± {sd:.2f}"
            else:
                row["Mean ± SD"] = ""
            rows.append(row)
    dest = paper / f"Table_{sid}_cell{cell}_per_seed_acc_f1.csv"
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"  paper <- {dest.name}", flush=True)


def write_s12_s15(sid: str, paper: Path) -> None:
    raw = _per_seed_raw()
    if raw is None:
        return
    col, fname = {
        "S12": ("rho_last", "Table_S12_rho_per_seed.csv"),
        "S13": ("H_alpha", "Table_S13_H_alpha_per_seed.csv"),
        "S14": ("alpha_tvar", "Table_S14_alpha_tvar_per_seed.csv"),
        "S15": ("perm_null_z", "Table_S15_perm_null_z_per_seed.csv"),
    }[sid]
    full = raw[raw["config"] == "full"]
    seeds = sorted(int(s) for s in full["seed"].unique())
    cells = [lab for lab, _ in _CELL_TAGS]
    rows = []
    for s in seeds:
        row = {"seed": s}
        for lab in cells:
            hit = full[(full["cell"] == lab) & (full["seed"] == s)]
            if hit.empty or col not in hit.columns or pd.isna(hit.iloc[0][col]):
                row[lab] = ""
            else:
                row[lab] = float(hit.iloc[0][col])
        rows.append(row)
    means = {"seed": "Mean ± SD"}
    for lab in cells:
        vals = [float(r[lab]) for r in rows if r[lab] != ""]
        if vals:
            m, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
            means[lab] = f"{m:.4g} ± {sd:.4g}"
        else:
            means[lab] = ""
    rows.append(means)
    dest = paper / fname
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"  paper <- {dest.name}", flush=True)


def emit(sid: str, *, paper: Path | None = None,
         full_dir: Path | None = None) -> None:
    paper = paper or PAPER_TABLES
    full_dir = full_dir or (EXPERIMENTS_ROOT / "nppad_atsf_full")
    paper.mkdir(parents=True, exist_ok=True)
    if sid == "I":
        # written by main._write_protocol_csv
        return
    if sid == "II" or sid == "III":
        write_table_ii_iii(full_dir, paper)
        return
    if sid == "IV":
        write_table_iv(full_dir, paper)
        return
    if sid == "V":
        review = TABLES_ROOT / "reviewer_controls"
        _copy(review / "custom_paired.csv", paper / "Table_V_reviewer_controls.csv")
        _copy(review / "custom_tost.csv", paper / "Table_V_tost.csv")
        # ±1pp TOST (same schema as custom_tost.csv); NOT custom_paired_eps1pp.csv
        _copy(review / "custom_tost_eps1pp.csv", paper / "Table_V_tost_1pp.csv")
        return
    if sid == "VI":
        write_table_vi(paper)
        return
    if sid in _S_DEGRADATION:
        write_s1_s5(sid, paper)
        return
    if sid == "S6":
        write_s6(paper)
        return
    if sid == "S7":
        write_s7(paper)
        return
    if sid in _S_PERSEED_ACC:
        write_s8_s11(sid, paper)
        return
    if sid in ("S12", "S13", "S14", "S15"):
        write_s12_s15(sid, paper)
        return
    raise KeyError(sid)
