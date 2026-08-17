# -*- coding: utf-8 -*-
"""Extract per-seed raw metrics from diagnostics JSONL files into one CSV.

Usage (from the repository root)::

    python scripts/extract_per_seed.py --cell O --cell A --cell B --cell C --cell D
    python scripts/extract_per_seed.py          # all of O/A/B/C/D

Writes ``output/tables/per_seed_finals/per_seed_raw.csv``.
Bare ``--cell A`` resolves to ``output/experiments/CellA_NPPAD_TimesNet``.
``--cell O=/path/to/dir`` still accepted.

Each --root must contain a ``diagnostics/`` subfolder with ``{config}_seed{N}.jsonl``
files produced by atsf_dcg.run_experiments. The script reads the LAST record with a
``"final"`` field per file and writes a long-format table:

    cell, config, seed, accuracy, macro_f1, rho_last, H_alpha, alpha_tvar,
    perm_null_z, epochs_run, deg_<name> ...   (one column per degradation type)

Rows with no "final" record (incomplete runs) are skipped with a warning.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atsf_dcg.paths import TABLES_ROOT, cell_experiment_dir  # noqa: E402

PAT = re.compile(r"^(?P<slug>.+)_seed(?P<seed>\d+)\.jsonl$")

# final-diagnostics keys that get their own columns (besides accuracy/macro_f1)
SCALAR_KEYS = ["rho_last", "H_alpha", "alpha_tvar", "epochs_run"]


def scan_root(root: Path, cell: str) -> list[dict]:
    diag = root / "diagnostics"
    if not diag.is_dir():
        raise SystemExit(f"[error] {diag} not found -- pass the results root "
                         f"that contains diagnostics/")
    rows = []
    for path in sorted(diag.glob("*_seed*.jsonl")):
        m = PAT.match(path.name)
        if not m:
            continue
        final = None
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "final" in rec:
                    final = rec["final"]
        if final is None:
            print(f"[warn] no final row in {path.name} (incomplete run?) -- skipped")
            continue
        f = dict(final)
        row = {
            "cell": cell,
            "config": m.group("slug"),
            "seed": int(m.group("seed")),
            "accuracy": f.pop("accuracy", None),
            "macro_f1": f.pop("macro_f1", None),
        }
        for k in SCALAR_KEYS:
            row[k] = f.pop(k, None)
        pn = f.pop("perm_null", None)
        row["perm_null_z"] = pn.get("z") if isinstance(pn, dict) else None
        deg = f.pop("degradation", None)
        if isinstance(deg, dict):
            for name, acc in deg.items():
                row[f"deg_{name}"] = acc
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None,
                    help="single results root (use with a bare --cell LABEL)")
    ap.add_argument("--cell", action="append", default=None,
                    help="bare label O/A/B/C/D (resolves under output/experiments) "
                         "or LABEL=/path/to/results; default: all five cells")
    ap.add_argument("--out", type=Path,
                    default=TABLES_ROOT / "per_seed_finals" / "per_seed_raw.csv")
    args = ap.parse_args()

    jobs: list[tuple[Path, str]] = []
    cells = args.cell or ["O", "A", "B", "C", "D"]
    for spec in cells:
        if "=" in spec:
            label, _, p = spec.partition("=")
            jobs.append((Path(p), label))
        elif args.root is not None:
            jobs.append((args.root, spec))
        else:
            jobs.append((cell_experiment_dir(spec), spec))

    rows: list[dict] = []
    for root, label in jobs:
        got = scan_root(root, label)
        print(f"[cell {label}] {len(got)} runs from {root}")
        rows.extend(got)

    df = pd.DataFrame(rows).sort_values(["cell", "config", "seed"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[done] {len(df)} rows x {len(df.columns)} cols -> {args.out}")
    # quick sanity summary
    summ = df.groupby(["cell", "config"])["seed"].nunique()
    bad = summ[summ < summ.max()]
    if len(bad):
        print("[warn] configs with fewer seeds than the maximum:")
        print(bad.to_string())


if __name__ == "__main__":
    main()
