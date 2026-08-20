#!/usr/bin/env python3
"""Resume an interrupted Cell D (or any cell) — run only missing (config, seed) pairs.

A run counts as DONE only if its jsonl contains a complete "final" record;
an interrupted run leaves a jsonl with epoch rows but no final -> re-run.

Usage (from the repository root)::

    python scripts/run_cellD_resume.py
    python scripts/run_cellD_resume.py --dry

Paderborn data must already sit under data/Paderborn/ (see data/README.md).
Override with PAD_ROOT only if needed.

After it finishes, merge + export:
    python -m atsf_dcg.run_experiments --merge-only --out-dir output/experiments/CellD_Paderborn_ATSF
    python scripts/export_results.py
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atsf_dcg.paths import REPO_ROOT, paderborn_root, cell_experiment_dir  # noqa: E402

PROJECT = REPO_ROOT
PAD = os.environ.get("PAD_ROOT", str(paderborn_root()))

OUT_DIR = str(cell_experiment_dir("D"))
CONFIGS = ["full", "w/o_spectral", "w/o_temporal", "w/o_fusion", "w/o_gating"]
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
EXTRA = (f"--cell replication --dataset paderborn --arch atsf "
         f"--data-root \"{PAD}\" --out-dir \"{OUT_DIR}\" "
         f"--degradation --log-epoch-indicators")


def slug(config: str) -> str:
    # 'w/o_spectral' -> 'wo_spectral' (matches run_experiments' file naming)
    return config.replace("/", "").replace(" ", "_")


def is_complete(jsonl: Path) -> bool:
    """Complete iff any line parses to a dict containing a 'final' record."""
    if not jsonl.exists():
        return False
    try:
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # possibly truncated tail line
                if isinstance(row, dict) and "final" in row:
                    return True
    except OSError:
        return False
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="report only, run nothing")
    args = ap.parse_args()

    diag = PROJECT / OUT_DIR / "diagnostics"
    missing = {}  # seed -> [configs]
    done_n = 0
    for seed in SEEDS:
        for cfg in CONFIGS:
            f = diag / f"{slug(cfg)}_seed{seed}.jsonl"
            if is_complete(f):
                done_n += 1
            else:
                missing.setdefault(seed, []).append(cfg)

    total = len(SEEDS) * len(CONFIGS)
    print(f"{OUT_DIR}: {done_n}/{total} runs complete")
    if not missing:
        print("nothing to do — all runs complete. Now merge:")
        print(f"  python3 -m atsf_dcg.run_experiments --merge-only "
              f"--out-dir {OUT_DIR}")
        return

    for seed in sorted(missing):
        print(f"  missing seed {seed}: {missing[seed]}")
    if args.dry:
        return

    t0 = time.time()
    for seed in sorted(missing):
        cfgs = missing[seed]
        cmd = (f"{sys.executable} -m atsf_dcg.run_experiments {EXTRA} "
               f"--seeds {seed} --configs {' '.join(cfgs)}")
        print(f"\n===== seed {seed} START {time.strftime('%H:%M:%S')} "
              f"({len(cfgs)} configs) =====\n{cmd}", flush=True)
        r = subprocess.run(cmd, shell=True, cwd=PROJECT)
        if r.returncode != 0:
            sys.exit(f"seed {seed} FAILED (rc={r.returncode}) — fix, "
                     f"then re-run this script (it will skip completed runs)")
        print(f"===== seed {seed} DONE =====", flush=True)

    print(f"\nRESUME DONE in {(time.time() - t0) / 60:.1f} min. Now merge:")
    print(f"  python3 -m atsf_dcg.run_experiments --merge-only "
          f"--out-dir {OUT_DIR}")


if __name__ == "__main__":
    main()
