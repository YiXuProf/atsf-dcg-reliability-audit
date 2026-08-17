#!/usr/bin/env python3
"""Collect five-cell result tables into output/tables/five_cell_summary/.

    python scripts/export_results.py
    python scripts/export_results.py --diagnostics
"""
import argparse
import shutil
import tarfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atsf_dcg.paths import (  # noqa: E402
    CELL_EXPORT_TAGS, EXPERIMENTS_ROOT, TABLES_ROOT,
)

TABLES = ["results_table", "remedy_table", "significance", "degradation_table"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostics", action="store_true",
                    help="also pack diagnostics/*.jsonl as tar.gz per cell")
    args = ap.parse_args()

    export = TABLES_ROOT / "five_cell_summary"
    export.mkdir(parents=True, exist_ok=True)
    n_csv = n_tar = 0

    for dirname, tag in CELL_EXPORT_TAGS.items():
        if dirname == "nppad_atsf_full":
            continue
        d = EXPERIMENTS_ROOT / dirname
        if not d.is_dir():
            print(f"!! missing dir: {d} (skipped)")
            continue
        for t in TABLES:
            src = d / f"{t}.csv"
            if src.exists():
                shutil.copy(src, export / f"{t}_{tag}.csv")
                n_csv += 1
            else:
                print(f"   (no {t}.csv in {dirname})")
        if args.diagnostics:
            diag = d / "diagnostics"
            if diag.is_dir():
                out = export / f"diagnostics_{tag}.tar.gz"
                with tarfile.open(out, "w:gz") as tar:
                    tar.add(diag, arcname="diagnostics")
                n_tar += 1
                print(f"   packed {out} "
                      f"({out.stat().st_size / 1e6:.1f} MB, "
                      f"{len(list(diag.glob('*.jsonl')))} jsonl)")

    print(f"\n=== {export}/: {n_csv} csv" +
          (f" + {n_tar} diagnostics archives" if args.diagnostics else "") +
          " ===")
    for p in sorted(export.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
