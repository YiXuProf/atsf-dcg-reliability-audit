"""Probe the real TEP .mat file structure so the loader can be fixed.

Usage (from the repository root)::
    python scripts/probe_tep_mat.py
    python scripts/probe_tep_mat.py data/TEP
Prints a recursive structural summary of faultfreetraining.mat (and the
faulty training file briefly) — paste ALL output back.
"""
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio


def summarize(name, obj, depth=0, max_depth=4):
    pad = "  " * depth
    t = type(obj).__name__
    if isinstance(obj, np.ndarray):
        info = f"ndarray shape={obj.shape} dtype={obj.dtype}"
        if obj.dtype.names:  # structured/void array (MATLAB struct/table)
            print(f"{pad}{name}: {info} FIELDS={list(obj.dtype.names)}")
            if depth < max_depth:
                flat = obj.ravel()
                probe = flat[0] if flat.size else None
                for fname in obj.dtype.names[:8]:
                    try:
                        field = obj[fname]
                        summarize(f".{fname}", field, depth + 1, max_depth)
                        if probe is not None and hasattr(probe, "__getitem__"):
                            pass
                    except Exception as e:  # noqa: BLE001
                        print(f"{pad}  .{fname}: <error {e}>")
                if flat.size > 1:
                    print(f"{pad}  (struct array with {flat.size} elements; "
                          f"showing fields of the whole array)")
        elif obj.dtype == object:
            print(f"{pad}{name}: {info} (object array)")
            if depth < max_depth and obj.size:
                first = obj.ravel()[0]
                summarize(f"{name}[0]", first, depth + 1, max_depth)
        else:
            print(f"{pad}{name}: {info}")
    elif isinstance(obj, dict):
        print(f"{pad}{name}: dict keys={list(obj.keys())[:12]}")
        if depth < max_depth:
            for k in list(obj.keys())[:12]:
                summarize(str(k), obj[k], depth + 1, max_depth)
    else:
        print(f"{pad}{name}: {t} = {str(obj)[:120]}")


def probe_file(path: Path):
    print(f"\n===== {path.name} ({path.stat().st_size/1e6:.1f} MB) =====")
    m = sio.loadmat(path)
    for key in m:
        if key.startswith("__"):
            continue
        summarize(key, m[key])


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[1] / "data" / "TEP")
    for fname in ("faultfreetraining.mat", "faultytraining.mat"):
        p = root / fname
        if p.exists():
            probe_file(p)
        else:
            print(f"MISSING: {p}")


if __name__ == "__main__":
    main()
