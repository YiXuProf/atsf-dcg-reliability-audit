#!/usr/bin/env python3
"""Scan all Paderborn .mat files for loadmat failures.

Usage (from the repository root)::

    python scripts/scan_bad_mat.py
    python scripts/scan_bad_mat.py data/Paderborn
"""
import sys
from pathlib import Path
from scipy.io import loadmat

root = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).resolve().parents[1] / "data" / "Paderborn")
files = sorted(root.rglob("*.mat"))
bad = []
for i, p in enumerate(files, 1):
    try:
        loadmat(str(p))
    except Exception as e:
        bad.append((str(p.relative_to(root)), f"{type(e).__name__}: {e}"))
    if i % 400 == 0:
        print(f"... {i}/{len(files)} scanned, {len(bad)} bad", flush=True)
print(f"\n=== {len(files)} scanned, {len(bad)} bad ===")
for f, e in bad:
    print("BAD:", f, "|", e)
