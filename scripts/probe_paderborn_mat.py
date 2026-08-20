#!/usr/bin/env python3
"""Probe ONE Paderborn bearing .mat file before extracting the full archive.

Usage::

    python probe_paderborn_mat.py /path/to/N15_M07_F10_K001_1.mat

Prints what ``scipy.io.loadmat`` actually finds in the file (top-level
variables, struct fields, the Y channel records with names/dtypes/shapes),
checks the assumptions the ``atsf_dcg.data_paderborn`` loader relies on
(6 dynamic channels, 64 kHz vs 4 kHz sample counts, downsample factor 16,
run-file naming), and finishes with a clear OK/ISSUES verdict.  Standalone
(numpy + scipy only); if the ``atsf_dcg`` package is importable it also
cross-checks the real loader on the file.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

FNAME_RE = re.compile(
    r"^(?P<setting>N\d+_M\d+_F\d+)_(?P<bearing>[A-Za-z]+\d+)_(?P<run>\d+)$")
BEARING_RE = re.compile(r"^(?P<code>K[ABI]?)(?P<num>\d{2,3})$")
EXPECTED_64K = ("phase_current_1", "phase_current_2", "vibration_1")
EXPECTED_4K = ("force", "torque", "speed")
EXPECTED_DROP = ("temp",)
DOWNSAMPLE = 16


def _scalar_str(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    a = np.asarray(x)
    if a.dtype == object and a.size == 1:
        return _scalar_str(a.reshape(-1)[0])
    return str(a.reshape(-1)[0]) if a.size else ""


def _describe(a: np.ndarray) -> str:
    a = np.asarray(a)
    names = f" fields={list(a.dtype.names)}" if a.dtype.names else ""
    return f"dtype={a.dtype}{names} shape={a.shape}"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"ERROR: no such file: {path}")
        return 2
    issues: list[str] = []

    # ---- filename convention ----------------------------------------------
    print(f"file: {path}")
    m = FNAME_RE.match(path.stem)
    if m:
        bm = BEARING_RE.match(m.group("bearing"))
        code = bm.group("code") if bm else "<unknown>"
        cls = {"K": "Healthy", "KA": "OuterRing", "KI": "InnerRing",
               "KB": "BothRings"}.get(code, "<unknown>")
        print(f"  name parse: setting={m.group('setting')} "
              f"bearing={m.group('bearing')} code={code} class={cls} "
              f"run={m.group('run')}")
        if not bm:
            issues.append(f"bearing code '{m.group('bearing')}' does not "
                          "match K###/KA##/KI##/KB##")
    else:
        issues.append(f"file name '{path.name}' does not match "
                      "'{setting}_{bearing}_{run#}.mat'")

    # ---- loadmat -----------------------------------------------------------
    try:
        from scipy.io import loadmat
    except ImportError:
        print("ERROR: scipy is required (pip install scipy)")
        return 2
    try:
        m = loadmat(str(path))
    except NotImplementedError as e:
        print(f"ERROR: scipy.io.loadmat cannot decode this file: {e}\n"
              "       -> looks like MATLAB v7.3/HDF5 or MCOS; the loader "
              "assumes a classicLevel .mat. Inspect with h5py if so.")
        return 1
    keys = [k for k in m if not k.startswith("__")]
    print(f"  top-level variables: {keys}")
    if not keys:
        print("ERROR: no data variables found")
        return 1

    struct = None
    for k in keys:
        v = np.asarray(m[k])
        print(f"  variable '{k}': {_describe(v)}")
        if v.dtype.names and "Y" in v.dtype.names and struct is None:
            struct, sname = v, k
    if struct is None:
        print("ERROR: no variable is a struct with a 'Y' field; the loader "
              "expects the measurement struct {Info, X, Y, Description}")
        return 1
    top = struct.reshape(-1)[0]
    print(f"  measurement struct '{sname}' fields: {list(struct.dtype.names)}")
    for f in struct.dtype.names:
        if f != "Y":
            print(f"    {f}: {_describe(np.asarray(top[f]))}")

    # ---- Y channel records ---------------------------------------------------
    y = np.asarray(top["Y"])
    while y.dtype == object and y.size == 1:
        y = np.asarray(y.reshape(-1)[0])
    channels: dict[str, np.ndarray] = {}
    if y.dtype.names and {"Name", "Data"} <= set(y.dtype.names):
        els = list(y.reshape(-1))
        print(f"  Y: struct array, {len(els)} channel records "
              f"(fields {list(y.dtype.names)})")
        for el in els:
            channels[_scalar_str(el["Name"]).strip().lower()] = el["Data"]
    elif y.dtype == object:
        els = list(y.reshape(-1))
        print(f"  Y: object/cell array, {len(els)} elements")
        for raw in els:
            el = np.asarray(raw)
            if el.dtype.names and {"Name", "Data"} <= set(el.dtype.names):
                e = el.reshape(-1)[0]
                channels[_scalar_str(e["Name"]).strip().lower()] = e["Data"]
            else:
                issues.append(f"unsupported Y element {_describe(el)}")
    else:
        issues.append(f"unsupported Y encoding: {_describe(y)}")

    lengths: dict[str, int] = {}
    for name, data in channels.items():
        d = np.asarray(data)
        while d.dtype == object and d.size == 1:
            d = np.asarray(d.reshape(-1)[0])
        lengths[name] = int(d.size)
        print(f"    channel '{name}': dtype={d.dtype} shape={d.shape} "
              f"samples={d.size}")

    # ---- loader assumptions -------------------------------------------------
    for c in EXPECTED_64K + EXPECTED_4K:
        if c not in channels:
            issues.append(f"expected channel '{c}' not found")
    for c in EXPECTED_DROP:
        if c not in channels:
            print(f"  note: droppable channel '{c}' absent (fine)")
    if all(c in lengths for c in EXPECTED_64K + EXPECTED_4K):
        l64 = lengths[EXPECTED_64K[0]]
        l4 = lengths[EXPECTED_4K[0]]
        ratio = l64 / max(l4, 1)
        print(f"  64k/4k length ratio: {l64}/{l4} = {ratio:.2f} "
              f"(loader assumes {DOWNSAMPLE})")
        if abs(ratio - DOWNSAMPLE) > 0.5:
            issues.append(f"unexpected sample-count ratio {ratio:.2f}")
        pooled = l64 // DOWNSAMPLE
        print(f"  after mean-pool x{DOWNSAMPLE} + alignment: "
              f"(6, {min(pooled, l4)}) on the 4 kHz grid "
              f"-> {(min(pooled, l4) - 128) // 128 + 1 if min(pooled, l4) >= 128 else 0} "
              "windows of 128 @ stride 128")

    # ---- cross-check the real loader if available ----------------------------
    try:
        here = Path(__file__).resolve().parent
        for cand in (here / "project", here, Path.cwd()):
            if (cand / "atsf_dcg").is_dir():
                sys.path.insert(0, str(cand))
                break
        from atsf_dcg.data_paderborn import _load_run  # noqa
        w: list[str] = []
        arr = _load_run(path, w)
        print(f"  atsf_dcg.data_paderborn._load_run: OK -> {arr.shape} "
              f"{arr.dtype}" + (f" warnings={w}" if w else ""))
    except Exception as e:  # noqa: BLE001 - probe must never crash here
        print(f"  atsf_dcg cross-check unavailable/failed: "
              f"{type(e).__name__}: {e}")

    print("VERDICT: " + ("OK — loader assumptions hold."
                             if not issues else
                             "ISSUES:\n  - " + "\n  - ".join(issues)))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
