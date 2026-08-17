#!/usr/bin/env python3
"""Paper figure/table entry for ATSF-DCG.

Numbers match the RESS manuscript and supplement: Fig. 1–13, Table I–VII,
Tables S1–S15 (S8a is the Cell O per-seed matrix).

One-click (uses existing ``output/experiments`` artefacts; no training)::

    python main.py all

One figure or table::

    python main.py fig 1
    python main.py fig 1 12 13
    python main.py fig 3-7
    python main.py table I
    python main.py table VI S1
    python main.py table all

List the catalog::

    python main.py
    python main.py list
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SCRIPTS = REPO / "scripts"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from atsf_dcg.paths import (  # noqa: E402
    EXPERIMENTS_ROOT,
    FIGURES_ROOT,
    INTERMEDIATE_ROOT,
    PAPER_FIGS as PAPER_FIG_DIR,
    PAPER_TABLES as PAPER_TABLE_DIR,
    TABLES_ROOT,
    nppad_root,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Catalog (RESS manuscript numbering)
# ---------------------------------------------------------------------------

FIG_INFO = {
    "1": "Architecture / failure-mode map (no data)",
    "2": "rho training trajectories (needs nppad_atsf_full/rho_curve.csv)",
    "3": "fusion-weight histogram (needs eval_dump; generated with 4–7)",
    "4": "per-class fusion weights (eval_dump)",
    "5": "gate heatmap (eval_dump)",
    "6": "t-SNE raw vs features (eval_dump + NPPAD for the raw panel)",
    "7": "confusion matrix (eval_dump)",
    "8": "mechanism four-panel (needs nppad_atsf_full results/remedy tables)",
    "9": "five-cell ablation forest (needs output/tables)",
    "10": "early-stop indicator trajectories (needs output/tables)",
    "11": "failure regime map (needs output/tables)",
    "12": "D1–D5 screening workflow (no data)",
    "13": "sensor-degradation heatmap (needs output/tables)",
}

TABLE_INFO = {
    "I": "NPPAD per-class run-level split (protocol_report.json)",
    "II": "Ablation-as-diagnosis on NPPAD (results + tests vs full)",
    "III": "Remedy evaluation on NPPAD (results + tests; last rows = controls)",
    "IV": "Mechanism indicators H(alpha), rho, S(tau), perm-null z",
    "V": "Planned control contrasts / TOST (Experiments A and B)",
    "VI": "Five-cell recurrence (per-cell results + significance)",
    "VII": "D1–D5 rules (text in the manuscript; not a CSV)",
    "S1": "Cell O degradation matrix (supplement S.1)",
    "S2": "Cell A degradation matrix",
    "S3": "Cell B degradation matrix",
    "S4": "Cell C degradation matrix",
    "S5": "Cell D degradation matrix",
    "S6": "Cell O v7 rerun vs original-paper grid",
    "S7": "Spearman: epoch-3 indicators vs snr10 drop",
    "S8a": "Cell O per-seed Acc / F1 (%)",
    "S8": "Cell A per-seed Acc / F1 (%)",
    "S9": "Cell B per-seed Acc / F1 (%)",
    "S10": "Cell C per-seed Acc / F1 (%)",
    "S11": "Cell D per-seed Acc / F1 (%)",
    "S12": "Full-model per-seed rho (all cells)",
    "S13": "Full-model per-seed H(alpha)",
    "S14": "Full-model per-seed Var_t(alpha)",
    "S15": "Full-model per-seed perm-null z",
}

_SKIP_TABLES = {"VII"}

_FIG_V7_ONLY = {"9": "9", "10": "8", "11": "10", "13": "7"}

_DUMP_DIR = INTERMEDIATE_ROOT / "eval_dump" / "full_seed42"
_NPPAD_FULL = EXPERIMENTS_ROOT / "nppad_atsf_full"


def catalog() -> str:
    figs = "\n".join(f"  fig {k:<4} {v}" for k, v in FIG_INFO.items())
    tabs = "\n".join(f"  table {k:<4} {v}" for k, v in TABLE_INFO.items())
    return f"""ATSF-DCG paper outputs  ({REPO})
Numbers match the RESS manuscript / supplement: Fig. 1–13 / Table I–VII / Tables S1–S15.

Usage:
  python main.py                 print this catalog
  python main.py all             all figures + all tables (no training)
  python main.py fig             all figures
  python main.py fig 1           one figure
  python main.py fig 1 12 13     several figures
  python main.py fig 3-7         a range
  python main.py table           all tables
  python main.py table I VI S1   selected tables

Outputs land in output/paper/figures and output/paper/tables
(generators also refresh output/figures and output/tables).

Figures
{figs}

Tables
{tabs}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(SCRIPTS / script), *args]
    print(">", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0:
        raise SystemExit(f"{script} failed (rc={r.returncode})")


def _copy_pair(src_stem: Path, dest_stem: Path) -> list[Path]:
    dest_stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in (".png", ".svg"):
        src = src_stem.with_suffix(ext)
        if not src.is_file():
            continue
        dst = dest_stem.with_suffix(ext)
        shutil.copy2(src, dst)
        written.append(dst)
        print(f"  paper <- {dst.relative_to(REPO)}", flush=True)
    if not written:
        raise FileNotFoundError(f"missing {src_stem.with_suffix('.png')}")
    return written


def _copy_file(src: Path, dest: Path) -> Path | None:
    if not src.is_file():
        print(f"  [skip] missing {src}", flush=True)
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"  paper <- {dest.relative_to(REPO)}", flush=True)
    return dest


def _expand_fig_ids(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        t = tok.lower().replace("fig.", "").replace("fig", "").strip()
        if "-" in t:
            a, b = t.split("-", 1)
            out.extend(str(i) for i in range(int(a), int(b) + 1))
        elif t in ("all", "*"):
            out.extend(FIG_INFO)
        else:
            out.append(str(int(t)))
    seen: set[str] = set()
    ordered: list[str] = []
    for i in out:
        if i not in FIG_INFO:
            raise SystemExit(f"unknown figure {i!r}; expected 1–13")
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    return ordered


_ROMAN = {
    "1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "6": "VI", "7": "VII",
    "I": "I", "II": "II", "III": "III", "IV": "IV", "V": "V", "VI": "VI",
    "VII": "VII",
}


def _expand_table_ids(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        t = tok.upper().replace("TABLE", "").replace("TAB.", "").replace("TAB", "")
        t = t.replace(".", "").strip()
        if t in ("ALL", "*"):
            out.extend(TABLE_INFO)
            continue
        if t.startswith("S"):
            rest = t[1:]
            if rest.upper() == "8A":
                key = "S8a"
            else:
                key = f"S{int(rest)}"
        else:
            key = _ROMAN.get(t, t)
        out.append(key)
    seen: set[str] = set()
    ordered: list[str] = []
    for i in out:
        if i not in TABLE_INFO:
            raise SystemExit(f"unknown table {i!r}; expected I–VII or S1–S15 (S8a)")
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    return ordered


def _banner(msg: str) -> None:
    print(f"\n===== {msg} =====", flush=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _fig_1() -> None:
    _run("make_fig0_architecture_audit_map.py")
    _copy_pair(
        FIGURES_ROOT / "architecture" / "fig0_architecture_audit_map",
        PAPER_FIG_DIR / "fig01_architecture_audit_map",
    )


def _fig_2() -> None:
    rho = _NPPAD_FULL / "rho_curve.csv"
    if not rho.is_file():
        raise FileNotFoundError(f"{rho} not found (run the NPPAD full grid first)")
    dest = PAPER_FIG_DIR / "fig02_rho_trajectories.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run("make_fig1_rho.py",
         "--results-dir", str(_NPPAD_FULL),
         "--out", str(dest))
    svg = dest.with_suffix(".svg")
    if svg.is_file():
        print(f"  paper <- {svg.relative_to(REPO)}", flush=True)


def _fig_3_to_7() -> None:
    if not _DUMP_DIR.is_dir():
        raise FileNotFoundError(
            f"{_DUMP_DIR} not found (need output/intermediate/eval_dump/full_seed42)")
    flags = ["--dump-dir", str(_DUMP_DIR),
             "--out-dir", str(FIGURES_ROOT / "fusion")]
    nppad = nppad_root()
    if nppad.exists():
        flags.extend(["--data-root", str(nppad)])
    _run("make_fig_runlevel.py", *flags)
    mapping = {
        "fig2_fusion_weight_dist": "fig03_fusion_weight_dist",
        "fig3_fusion_weight_by_class": "fig04_fusion_weight_by_class",
        "fig4_gate_heatmap": "fig05_gate_heatmap",
        "fig5_tsne": "fig06_tsne",
        "fig6_confusion_matrix": "fig07_confusion_matrix",
    }
    fusion = FIGURES_ROOT / "fusion"
    for src, dst in mapping.items():
        try:
            _copy_pair(fusion / src, PAPER_FIG_DIR / dst)
        except FileNotFoundError as exc:
            print(f"  [skip] {exc}", flush=True)


def _fig_8() -> None:
    if not (_NPPAD_FULL / "results_table.csv").is_file():
        raise FileNotFoundError(f"{_NPPAD_FULL / 'results_table.csv'} not found")
    dest = PAPER_FIG_DIR / "fig08_diagnostics.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run("make_fig6.py",
         "--results-dir", str(_NPPAD_FULL),
         "--out", str(dest))


def _fig_v7(paper_id: str) -> None:
    only = _FIG_V7_ONLY[paper_id]
    _run("make_figs_v7.py", "--only", only)
    src_name = {
        "9": "fig9_ablation_forest",
        "10": "fig8_early_indicator_trajectories",
        "11": "fig10_regime_map",
        "13": "fig7_degradation_heatmap",
    }[paper_id]
    dst_name = {
        "9": "fig09_ablation_forest",
        "10": "fig10_early_indicator_trajectories",
        "11": "fig11_regime_map",
        "13": "fig13_degradation_heatmap",
    }[paper_id]
    _copy_pair(FIGURES_ROOT / "five_cell" / src_name, PAPER_FIG_DIR / dst_name)


def _fig_12() -> None:
    _run("make_fig12_decision_workflow.py")
    _copy_pair(
        FIGURES_ROOT / "architecture" / "fig12_decision_workflow",
        PAPER_FIG_DIR / "fig12_decision_workflow",
    )


def emit_figures(ids: list[str]) -> None:
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    # 3–7 share one generator; run it once if any of them is requested
    done_runlevel = False
    for i in ids:
        _banner(f"Fig. {i}  {FIG_INFO[i]}")
        try:
            if i == "1":
                _fig_1()
            elif i == "2":
                _fig_2()
            elif i in {"3", "4", "5", "6", "7"}:
                if not done_runlevel:
                    _fig_3_to_7()
                    done_runlevel = True
            elif i == "8":
                _fig_8()
            elif i in _FIG_V7_ONLY:
                _fig_v7(i)
            elif i == "12":
                _fig_12()
            else:
                raise SystemExit(f"no generator for Fig. {i}")
        except (FileNotFoundError, SystemExit) as exc:
            print(f"  [skip] Fig. {i}: {exc}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] Fig. {i}: {type(exc).__name__}: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _refresh_table_sources() -> None:
    """Rebuild summary CSVs from experiment directories when they exist."""
    if EXPERIMENTS_ROOT.is_dir() and any(EXPERIMENTS_ROOT.iterdir()):
        try:
            _run("export_results.py")
        except SystemExit as exc:
            print(f"  [skip] export_results: {exc}", flush=True)
        try:
            _run("export_epoch_indicators.py")
        except SystemExit as exc:
            print(f"  [skip] export_epoch_indicators: {exc}", flush=True)
        try:
            _run("extract_per_seed.py")
        except SystemExit as exc:
            print(f"  [skip] extract_per_seed: {exc}", flush=True)
        diag = _NPPAD_FULL / "diagnostics"
        if diag.is_dir():
            try:
                _run("diagnose_controls.py",
                     "--diag-dir", str(diag),
                     "--out", str(TABLES_ROOT / "reviewer_controls" / "custom_paired.csv"))
            except SystemExit as exc:
                print(f"  [skip] diagnose_controls: {exc}", flush=True)


def _write_protocol_csv(src_json: Path, dest: Path) -> Path | None:
    if not src_json.is_file():
        print(f"  [skip] missing {src_json}", flush=True)
        return None
    report = json.loads(src_json.read_text(encoding="utf-8"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_json, dest.with_suffix(".json"))
    per = (report.get("dataset_report") or report.get("report") or {}).get("per_class", {})
    if not per:
        print(f"  paper <- {dest.with_suffix('.json').relative_to(REPO)}", flush=True)
        return dest.with_suffix(".json")
    import csv
    rows = []
    for name, rec in per.items():
        if not isinstance(rec, dict):
            continue
        rows.append({
            "class": name,
            "train_runs": rec.get("train_runs", rec.get("n_train_runs", "")),
            "val_runs": rec.get("val_runs", rec.get("n_val_runs", "")),
            "test_runs": rec.get("test_runs", rec.get("n_test_runs", "")),
            "train_windows": rec.get("train_windows", rec.get("n_train", "")),
            "val_windows": rec.get("val_windows", rec.get("n_val", "")),
            "test_windows": rec.get("test_windows", rec.get("n_test", "")),
        })
    with dest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  paper <- {dest.relative_to(REPO)}", flush=True)
    return dest


def emit_tables(ids: list[str], *, refresh: bool = True) -> None:
    from atsf_dcg import manuscript_tables as mt

    PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if refresh:
        _banner("refresh table sources from experiments/")
        _refresh_table_sources()

    mt.retire_misnamed(PAPER_TABLE_DIR)

    full = _NPPAD_FULL
    wrote_ii_iii = False
    for i in ids:
        _banner(f"Table {i}  {TABLE_INFO[i]}")
        if i in _SKIP_TABLES:
            print(f"  [skip] Table {i} is manuscript text, not a generated CSV",
                  flush=True)
            continue
        if i == "I":
            _write_protocol_csv(
                full / "protocol_report.json",
                PAPER_TABLE_DIR / "Table_I_protocol.csv")
            continue
        if i in ("II", "III"):
            if not wrote_ii_iii:
                mt.emit("II", paper=PAPER_TABLE_DIR, full_dir=full)
                wrote_ii_iii = True
            continue
        mt.emit(i, paper=PAPER_TABLE_DIR, full_dir=full)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Emit paper figures and tables (RESS Fig. 1–13 / Table I–VII / S1–S15).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run with no arguments to print the catalog.",
    )
    ap.add_argument("cmd", nargs="?", default="list",
                    choices=("all", "fig", "table", "list", "help"),
                    help="all | fig | table | list")
    ap.add_argument("ids", nargs="*",
                    help="figure numbers (1–13) or table ids (I–VII, S1–S15, S8a)")
    ap.add_argument("--no-refresh", action="store_true",
                    help="do not rebuild summary CSVs before copying tables")
    args = ap.parse_args(argv)

    if args.cmd in ("list", "help") and not args.ids:
        print(catalog())
        return

    if args.cmd == "all":
        emit_tables(_expand_table_ids(["all"]), refresh=not args.no_refresh)
        emit_figures(_expand_fig_ids(["all"]))
        _banner("done")
        print(f"figures: {PAPER_FIG_DIR}")
        print(f"tables:  {PAPER_TABLE_DIR}")
        return

    if args.cmd == "fig":
        ids = _expand_fig_ids(args.ids or ["all"])
        emit_figures(ids)
        return

    if args.cmd == "table":
        ids = _expand_table_ids(args.ids or ["all"])
        emit_tables(ids, refresh=not args.no_refresh)
        return

    print(catalog())


if __name__ == "__main__":
    main()
