#!/usr/bin/env python3
"""Run all v7 experiment cells sequentially (Step 4 + Step 5).

Idempotent-ish: rerunning re-runs everything (deterministic, same outputs).
Use --skip to skip already-finished cells, e.g. --skip A B after a partial run.

Usage (from the repository root)::

    python scripts/run_all_cells.py
    python scripts/run_all_cells.py A B   # skip cells A and B

Datasets must already sit under data/ (see data/README.md). Override with
env vars only if needed: NPPAD_ROOT, TEP_ROOT, PAD_ROOT.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atsf_dcg.paths import (  # noqa: E402
    REPO_ROOT, nppad_root, tep_root, paderborn_root, cell_experiment_dir,
)

PROJECT = REPO_ROOT
NPPAD = os.environ.get("NPPAD_ROOT", str(nppad_root()))
TEP = os.environ.get("TEP_ROOT", str(tep_root()))
PAD = os.environ.get("PAD_ROOT", str(paderborn_root()))

SEEDS = "42 43 44 45 46 47 48 49 50 51"
FLAGS = "--degradation --log-epoch-indicators"

_OUT_A = cell_experiment_dir("A")
_OUT_B = cell_experiment_dir("B")
_OUT_C = cell_experiment_dir("C")
_OUT_O = cell_experiment_dir("O")
_OUT_D = cell_experiment_dir("D")

# name, extra args (appended after the common parts)
CELLS = [
    ("A", f"--cell replication --dataset nppad --arch tsnet --data-root \"{NPPAD}\" "
          f"--configs full w/o_spectral w/o_temporal w/o_fusion w/o_gating "
          f"tsnet_vanilla --out-dir \"{_OUT_A}\""),
    ("B", f"--cell replication --dataset tep --arch atsf --data-root \"{TEP}\" "
          f"--configs full w/o_spectral w/o_temporal w/o_fusion w/o_gating "
          f"--out-dir \"{_OUT_B}\""),
    ("C", f"--cell replication --dataset tep --arch tsnet --data-root \"{TEP}\" "
          f"--configs full w/o_spectral w/o_temporal w/o_fusion w/o_gating "
          f"tsnet_vanilla --out-dir \"{_OUT_C}\""),
    ("O", f"--cell full --dataset nppad --arch atsf --data-root \"{NPPAD}\" "
          f"--configs full w/o_spectral w/o_temporal w/o_fusion "
          f"w/o_dynamic_gating w/o_gating --out-dir \"{_OUT_O}\""),
    ("D", f"--cell replication --dataset paderborn --arch atsf "
          f"--data-root \"{PAD}\" "
          f"--configs full w/o_spectral w/o_temporal w/o_fusion w/o_gating "
          f"--out-dir \"{_OUT_D}\""),
]


def main() -> None:
    skip = set(sys.argv[1:])  # e.g. `python3 run_all_cells.py A B`
    t0 = time.time()
    for name, extra in CELLS:
        if name in skip:
            print(f"===== Cell {name}: skipped =====", flush=True)
            continue
        cmd = (f"{sys.executable} -m atsf_dcg.run_experiments {extra} "
               f"--seeds {SEEDS} {FLAGS}")
        print(f"\n===== Cell {name} START {time.strftime('%H:%M:%S')} =====\n"
              f"{cmd}", flush=True)
        t1 = time.time()
        r = subprocess.run(cmd, shell=True, cwd=PROJECT)
        dt = (time.time() - t1) / 60
        if r.returncode != 0:
            print(f"===== Cell {name} FAILED (rc={r.returncode}) after "
                  f"{dt:.1f} min — STOPPING. Fix and rerun with: "
                  f"python3 run_all_cells.py "
                  f"{' '.join(c for c, _ in CELLS if c != name and c not in skip)}"
                  f" =====", flush=True)
            sys.exit(r.returncode)
        print(f"===== Cell {name} DONE in {dt:.1f} min =====", flush=True)
    print(f"\nALL CELLS DONE in {(time.time() - t0) / 60:.1f} min total",
          flush=True)


if __name__ == "__main__":
    main()
