#!/usr/bin/env python3
"""Extract per-epoch indicators + final records from output/experiments/*/jsonl.

Usage (from the repository root)::

    python scripts/export_epoch_indicators.py
    python scripts/export_epoch_indicators.py --inspect

Writes:
    output/tables/epoch_indicators/epoch_indicators_{tag}.csv
    output/tables/per_seed_finals/finals_{tag}.csv
"""
import argparse, csv, glob, json, os, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atsf_dcg.paths import (  # noqa: E402
    CELL_EXPORT_TAGS, EXPERIMENTS_ROOT, TABLES_ROOT,
)

def find_dicts_with(obj, required_keys, path=""):
    """Recursively collect dicts containing ALL required_keys."""
    hits = []
    if isinstance(obj, dict):
        if all(k in obj for k in required_keys):
            hits.append(obj)
        for k, v in obj.items():
            hits += find_dicts_with(v, required_keys, path + "." + str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits += find_dicts_with(v, required_keys, path + f"[{i}]")
    return hits

def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        kk = f"{prefix}{k}"
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[kk] = v
        elif isinstance(v, dict):
            out.update(flatten(v, kk + "."))
        # skip lists/other
    return out

def parse_jsonl(path):
    epoch_rows, final_rows, other_keysigs = [], [], set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ehits = find_dicts_with(rec, ("epoch", "h_alpha"))
            if ehits:
                for h in ehits:
                    epoch_rows.append(flatten(h))
                continue
            fhits = find_dicts_with(rec, ("final",))
            if fhits:
                for h in fhits:
                    fr = flatten(h)
                    # if {"final": true, ...rest...} keep the rest; if {"final": {...}} flatten nested
                    final_rows.append(fr)
                continue
            if isinstance(rec, dict):
                other_keysigs.add("|".join(sorted(rec.keys())))
    return epoch_rows, final_rows, other_keysigs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="experiments root (default: output/experiments)")
    ap.add_argument("--out", default=None,
                    help="if set, write both CSV families into this one folder")
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()
    root = os.path.abspath(args.root or str(EXPERIMENTS_ROOT))
    epoch_dir = (args.out or str(TABLES_ROOT / "epoch_indicators"))
    finals_dir = (args.out or str(TABLES_ROOT / "per_seed_finals"))
    os.makedirs(epoch_dir, exist_ok=True)
    os.makedirs(finals_dir, exist_ok=True)

    rdirs = [d for d in sorted(glob.glob(os.path.join(root, "*"))) if os.path.isdir(d)]
    if not rdirs:
        sys.exit(f"no experiment dirs under {root}")

    if args.inspect:
        # print structure of first jsonl found (recursive)
        for rd in rdirs:
            fs = sorted(glob.glob(os.path.join(rd, "**", "*.jsonl"), recursive=True))
            if fs:
                print("INSPECT:", fs[0])
                with open(fs[0]) as f:
                    for i, line in enumerate(f):
                        if i >= 5:
                            break
                        try:
                            rec = json.loads(line)
                            print(f"line {i}: keys = {list(rec.keys()) if isinstance(rec, dict) else type(rec)}")
                            print("  sample:", json.dumps(rec)[:600])
                        except Exception as e:
                            print(f"line {i}: parse error {e}")
                return
        sys.exit("no jsonl found")

    summary = []
    for rd in rdirs:
        tag = CELL_EXPORT_TAGS.get(os.path.basename(rd), os.path.basename(rd))
        all_epoch, all_final = [], []
        keysig_seen = set()
        nfiles = 0
        for jf in sorted(glob.glob(os.path.join(rd, "**", "*.jsonl"), recursive=True)):
            nfiles += 1
            base = os.path.basename(jf).replace(".jsonl", "")
            # config_seed{N}
            if "_seed" in base:
                cfg, seed = base.rsplit("_seed", 1)
            else:
                cfg, seed = base, ""
            er, fr, ok = parse_jsonl(jf)
            keysig_seen |= ok
            for r in er:
                r["config"], r["seed"] = cfg, seed
            for r in fr:
                r["config"], r["seed"] = cfg, seed
            all_epoch += er
            all_final += fr

        if all_epoch:
            cols = ["config", "seed"] + sorted({k for r in all_epoch for k in r} - {"config", "seed"})
            p = os.path.join(epoch_dir, f"epoch_indicators_{tag}.csv")
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                w.writerows(all_epoch)
            ep_status = f"{len(all_epoch)} epoch rows -> {os.path.basename(p)} ({os.path.getsize(p)//1024} KB)"
        else:
            ep_status = "NO per-epoch h_alpha records found"
        if all_final:
            cols = ["config", "seed"] + sorted({k for r in all_final for k in r} - {"config", "seed"})
            p = os.path.join(finals_dir, f"finals_{tag}.csv")
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                w.writerows(all_final)
            fi_status = f"{len(all_final)} final rows -> {os.path.basename(p)} ({os.path.getsize(p)//1024} KB)"
        else:
            fi_status = "NO final records found"
        summary.append((tag, nfiles, ep_status, fi_status, keysig_seen))
        if nfiles == 0:
            # show what IS in there, to locate the jsonl nesting
            sample = []
            for dp, dn, fn in os.walk(rd):
                for x in fn[:5]:
                    sample.append(os.path.relpath(os.path.join(dp, x), rd))
                if len(sample) >= 8:
                    break
            print(f"  [debug] {tag} contains: {sample[:8] if sample else '(empty dir)'}")

    print("\n================ SUMMARY ================")
    any_epoch = False
    for tag, nfiles, ep, fi, ks in summary:
        print(f"[{tag}] {nfiles} jsonl files")
        print(f"  epoch indicators: {ep}")
        print(f"  final records:    {fi}")
        if "epoch rows" in ep:
            any_epoch = True
        if ks and "NO per-epoch" in ep:
            print(f"  other record key-signatures seen: {sorted(ks)[:5]}")
    print("=========================================")
    if not any_epoch:
        print("RESULT: no per-epoch indicator logging in ANY cell -> runs were made WITHOUT --log-epoch-indicators.")
        print("        finals_*.csv still exported (cross-sectional rules possible).")
    else:
        print(f"RESULT: per-epoch indicators found. CSVs in {epoch_dir} and {finals_dir}")
    print("epoch CSVs:", sorted(os.listdir(epoch_dir)))
    print("finals CSVs:", sorted(os.listdir(finals_dir)))

if __name__ == "__main__":
    main()
