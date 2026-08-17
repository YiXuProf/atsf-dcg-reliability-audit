"""Paderborn (PU / KAt) bearing dataset adapter (plan v6, Stage 2b).

Source: Lessmeier et al. (2016), "Condition Monitoring of Bearing Damage in
Electromechanical Drive Systems by Using Motor Current Signals of Electric
Motors: A Benchmark Data Set for Data-Driven Classification",
PHM Society European Conference, doi:10.36001/phme.2016.v3i1.1577.
License: CC BY-NC 4.0 (non-commercial research use only).

DATASET FACTS (verified against the research brief):

- 32 bearings: 6 healthy (K001..K006); artificial damage KA01/03/05/06/07/
  08/09 (outer ring) and KI01/03/05/07/08 (inner ring); real
  accelerated-lifetime damage KA04/15/16/22/30 (outer), KI04/14/16/17/18/21
  (inner), KB23/24/27 (both rings).  The bearing-code letter(s) define the
  class: K=Healthy, KA=OuterRing, KI=InnerRing, KB=BothRings.
- Each bearing: 4 operating settings (N15_M07_F10, N09_M07_F10,
  N15_M01_F10, N15_M07_F04) x 20 runs of exactly 4 s = 80 ``.mat`` files
  named ``{setting}_{bearing}_{run#}.mat`` (run# 1..20).
- Each ``.mat`` file holds one struct with fields ``{Info, X, Y,
  Description}``; ``Y`` is a struct array of 7 channel records with fields
  ``Name`` and ``Data``: ``phase_current_1``, ``phase_current_2``,
  ``vibration_1`` (64 kHz, 256,000 samples), ``force``, ``torque``,
  ``speed`` (4 kHz, 16,000 samples) and ``temp`` (1 Hz, 4 samples).
  ``scipy.io.loadmat`` reads these files (they are NOT v7.3/MCOS); the
  loader still verifies readability defensively and raises a clear error
  listing what it found otherwise.  Run ``probe_paderborn_mat.py`` on one
  downloaded file to confirm the layout before extracting the full
  5.1 GB archive.

KNOWN UPSTREAM DEFECT (recorded in the paper's protocol report): the
official KA08 archive contains ONE corrupt file,
``KA08/N15_M01_F10_KA08_2.mat`` — ``scipy.io.loadmat`` raises
``TypeError: Expecting matrix here`` on it, and re-downloading yields the
same corrupt bytes (it is corrupt upstream, not a transfer artifact).
The adapter therefore treats ANY file that fails to load/parse as
corrupt: it is SKIPPED with a warning recorded in
``report["warnings"]`` and listed in ``report["excluded_files"]``, and
the load does not abort.  Exclusion happens BEFORE split assignment, so
the exact 60/20/20 split counts and the (bearing, setting) stratification
are computed from the successfully-loaded runs only: KA08 contributes 79
runs and the KA class 959 runs instead of 960 (all other counts
unchanged).

ADAPTER PROTOCOL (mirrors ``data.load_nppad`` semantics exactly):

- Channels: the 6 dynamic channels (2 phase currents + 1 vibration +
  force/torque/speed), aligned on a common 4 kHz grid: the 64 kHz channels
  are downsampled by exactly 16 via anti-alias mean-pooling (reshape to
  (-1, 16) + mean, after trimming to a multiple of 16); the 4 kHz channels
  are used as-is; all six are then trimmed to their common minimum length.
  The 1 Hz ``temp`` channel (4 samples/run) is dropped.  Each run becomes a
  ``(16000, 6)`` float64 array (shorter runs are accepted gracefully).
- Classes: 4-class by bearing-code letter: K->Healthy, KA->OuterRing,
  KI->InnerRing, KB->BothRings (label order follows the ``classes``
  argument).  The label is parsed from the file name (bearing-code regex);
  files matching the run-file pattern with an unknown bearing code are a
  hard error.
- Split unit = one .mat file, i.e. one (bearing, setting, run#) triple —
  never a window.  Per class, exact (train, val, test) counts are computed
  as 60%/20%/20% of the class's SUCCESSFULLY-LOADED runs (corrupt/
  unreadable files are excluded first — see the known-upstream-defect note
  above) and enforced through
  ``data._build_bundle(split_counts={class_index: counts})``.
- Stratification: within each class, runs are grouped into (bearing,
  setting) cells and dealt to train/val/test round-robin (per-cell quotas,
  largest-remainder, seeded shuffle of cell and run order) so every bearing
  and every operating setting appears in every split as evenly as possible
  — for the real data exactly 12/4/4 runs of each (bearing, setting) cell
  land in train/val/test (the one cell containing the corrupt KA08 file
  has 19 runs and is dealt as evenly as the algorithm handles any
  non-multiple count).  Same philosophy as
  ``data_tep._select_runs``'s evenly-spaced selection.  Because
  ``_build_bundle`` applies its own deterministic ``split_seed`` permutation
  per class, the adapter pre-computes that permutation (same seed, same
  per-class rng consumption order) and places each run at the list position
  that maps to its assigned split — the net assignment is exactly the
  stratified one while ``_build_bundle`` performs the split bookkeeping,
  train-only z-score and windowing.
- LEAKAGE NOTE (by design): the split is run-level, so the SAME bearing
  appears in train/val/test (consistent with the NPPAD/TEP run-level
  protocol); a bearing-held-out variant is available via
  ``load_paderborn(..., split_unit="bearing")`` / ``--split-unit bearing``.
- Per-channel z-score from the TRAINING runs only; sliding windows
  (default ``window=128, stride=128`` — 32 ms, 125 windows per 4 s run)
  are extracted afterwards by ``data._build_bundle``.

Expected real-data bundle (default arguments, with the one corrupt KA08
file excluded — see above): 2559 runs total — per class (runs,
train/val/test): K 480 (288/96/96), KA 959 (575/192/192), KI 880
(528/176/176), KB 240 (144/48/48); windows x125/run -> X_train
(191875, 6, 128), X_val (64000, 6, 128), X_test (64000, 6, 128).

``make_synthetic_paderborn`` mirrors ``data_tep.make_synthetic_tep`` for
tests/smoke runs: class-distinct noisy sinusoids on 6 channels, with a
fault signature (harmonic ramp + vibration bursts) for the damaged classes.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .data import DatasetBundle, _build_bundle

# ---------------------------------------------------------------------------
# dataset constants (Lessmeier et al. 2016)
# ---------------------------------------------------------------------------

CLASS_CODES: tuple[str, ...] = ("K", "KA", "KI", "KB")
CLASS_NAMES_PADERBORN: list[str] = ["Healthy", "OuterRing", "InnerRing",
                                    "BothRings"]
_CODE_TO_NAME: dict[str, str] = dict(zip(CLASS_CODES, CLASS_NAMES_PADERBORN))

BEARINGS_BY_CLASS: dict[str, list[str]] = {
    "K": [f"K{i:03d}" for i in range(1, 7)],
    "KA": ["KA01", "KA03", "KA05", "KA06", "KA07", "KA08", "KA09",
           "KA04", "KA15", "KA16", "KA22", "KA30"],
    "KI": ["KI01", "KI03", "KI05", "KI07", "KI08",
           "KI04", "KI14", "KI16", "KI17", "KI18", "KI21"],
    "KB": ["KB23", "KB24", "KB27"],
}
SETTINGS: tuple[str, ...] = ("N15_M07_F10", "N09_M07_F10", "N15_M01_F10",
                             "N15_M07_F04")
RUNS_PER_BEARING = 80  # 4 settings x 20 runs of 4 s

CITATION = ("Lessmeier et al. 2016, PHM Society European Conference, "
            "doi:10.36001/phme.2016.v3i1.1577 (license CC BY-NC 4.0)")

# channel contract: 6 dynamic channels on a common 4 kHz grid
CHANNELS_64K: tuple[str, ...] = ("phase_current_1", "phase_current_2",
                                 "vibration_1")
CHANNELS_4K: tuple[str, ...] = ("force", "torque", "speed")
CHANNEL_NAMES_PADERBORN: list[str] = list(CHANNELS_64K) + list(CHANNELS_4K)
DROPPED_CHANNELS: tuple[str, ...] = ("temp",)  # 1 Hz, 4 samples/run: useless
DOWNSAMPLE_FACTOR = 16       # 64 kHz -> 4 kHz (anti-alias mean-pool)
FS_TARGET_HZ = 4000
SAMPLES_PER_RUN = 16000      # 4 s at 4 kHz after alignment
WINDOWS_PER_RUN = 125        # at window=stride=128

# {setting}_{bearing}_{run#}, e.g. N15_M07_F10_K001_1(.mat)
_FNAME_RE = re.compile(
    r"^(?P<setting>N\d+_M\d+_F\d+)_(?P<bearing>[A-Za-z]+\d+)_(?P<run>\d+)$")
# K### (healthy) / KA## / KI## / KB## (damaged)
_BEARING_RE = re.compile(r"^(?P<code>K[ABI]?)(?P<num>\d{2,3})$")

_SPLIT_FRACS = (0.60, 0.20, 0.20)  # train / val / test

# Work-order Cell D bearing-level split (split_seed=42).  Counts are bearings,
# not runs; every run of a bearing stays in one partition.
BEARING_SPLIT_QUOTAS: dict[str, tuple[int, int, int]] = {
    "K": (4, 1, 1),    # 6 healthy
    "KA": (7, 3, 2),   # 12 outer-ring
    "KI": (7, 2, 2),   # 11 inner-ring
    "KB": (1, 1, 1),   # 3 compound (degenerates to 1/1/1; do not pool)
}


# ---------------------------------------------------------------------------
# filename / metadata parsing
# ---------------------------------------------------------------------------

def parse_run_filename(stem: str) -> tuple[str, str, str, int]:
    """Parse ``{setting}_{bearing}_{run#}`` -> (setting, bearing, code, run#).

    ``code`` is the class letter(s) of the bearing code: "K", "KA", "KI" or
    "KB".  Raises ValueError with a clear message for names that do not look
    like a Paderborn run file or whose bearing code is unknown.
    """
    m = _FNAME_RE.match(stem)
    if not m:
        raise ValueError(
            f"'{stem}' is not a Paderborn run-file name "
            "('{setting}_{bearing}_{run#}', e.g. 'N15_M07_F10_K001_1')")
    bearing = m.group("bearing")
    bm = _BEARING_RE.match(bearing)
    if not bm:
        raise ValueError(
            f"unknown bearing code '{bearing}' in '{stem}': expected K### "
            "(healthy, e.g. K001) or KA##/KI##/KB## (outer/inner/both rings, "
            "e.g. KA01, KI21, KB23)")
    return m.group("setting"), bearing, bm.group("code"), int(m.group("run"))


# ---------------------------------------------------------------------------
# .mat parsing (defensive: struct-array and cell-array Y encodings)
# ---------------------------------------------------------------------------

def _scalar_str(x) -> str:
    """MATLAB char field (numpy str array / bytes / nested object) -> str."""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    a = np.asarray(x)
    if a.dtype == object and a.size == 1:
        return _scalar_str(a.reshape(-1)[0])
    if a.size == 0:
        return ""
    return str(a.reshape(-1)[0])


def _as_1d_float(x) -> np.ndarray:
    """Channel Data field -> 1-D float64 (unwraps 1-element object cells)."""
    a = np.asarray(x)
    while a.dtype == object and a.size == 1:
        a = np.asarray(a.reshape(-1)[0])
    return np.asarray(a, dtype=np.float64).reshape(-1)


def _channel_entries(y: np.ndarray, fname: str) -> list[tuple[str, np.ndarray]]:
    """Y field -> [(channel_name, 1-D float64 data)].

    Handles both encodings scipy produces for the 7-record channel struct
    array: (a) a structured ndarray with fields Name/Data (any shape), and
    (b) an object/cell array whose elements are dicts or (1x1) structs with
    Name/Data fields.
    """
    y = np.asarray(y)
    while y.dtype == object and y.size == 1:  # 1x1 cell wrapping the array
        y = np.asarray(y.reshape(-1)[0])
    entries: list[tuple[str, np.ndarray]] = []
    if y.dtype.names and {"Name", "Data"} <= set(y.dtype.names):
        for el in y.reshape(-1):
            entries.append((_scalar_str(el["Name"]), _as_1d_float(el["Data"])))
        return entries
    if y.dtype == object:
        for raw in y.reshape(-1):
            if isinstance(raw, dict):
                entries.append((str(raw["Name"]), _as_1d_float(raw["Data"])))
                continue
            el = np.asarray(raw)
            if el.dtype.names and {"Name", "Data"} <= set(el.dtype.names):
                e = el.reshape(-1)[0]
                entries.append((_scalar_str(e["Name"]), _as_1d_float(e["Data"])))
            else:
                raise ValueError(
                    f"'{fname}': unsupported Y element (type "
                    f"{type(raw).__name__}, dtype {el.dtype}); expected "
                    "channel records with Name/Data fields")
        return entries
    raise ValueError(
        f"'{fname}': unsupported Y encoding (dtype {y.dtype}, shape "
        f"{y.shape}); expected a struct array of channel records with "
        "Name/Data fields")


def _read_measurement_struct(path: Path) -> np.ndarray:
    """loadmat one run file -> the measurement struct (must have field Y)."""
    try:
        from scipy.io import loadmat  # lazy heavy import
    except ImportError as e:  # pragma: no cover - scipy is a hard dependency
        raise ImportError("reading Paderborn .mat files requires scipy") from e
    try:
        m = loadmat(str(path))
    except NotImplementedError as e:
        raise ValueError(
            f"'{path.name}' could not be decoded by scipy.io.loadmat "
            f"({e}); it looks like a MATLAB v7.3/HDF5 or MCOS file, which "
            "the Paderborn release is not expected to be — run "
            "probe_paderborn_mat.py on this file to inspect its layout") from e
    except Exception as e:
        raise ValueError(
            f"could not read '{path.name}' with scipy.io.loadmat: "
            f"{type(e).__name__}: {e}") from e
    keys = [k for k in m if not k.startswith("__")]
    if not keys:
        raise ValueError(f"no data variable found in '{path.name}'")
    found = []
    for k in keys:
        v = np.asarray(m[k])
        if v.dtype.names and "Y" in v.dtype.names:
            return v
        names = f"fields={list(v.dtype.names)}" if v.dtype.names else \
            f"dtype={v.dtype}"
        found.append(f"'{k}' ({names}, shape {v.shape})")
    raise ValueError(
        f"no measurement struct with a 'Y' field in '{path.name}'; found "
        f"variables: {found}")


def _assemble_run(entries: list[tuple[str, np.ndarray]], fname: str,
                  warnings: list[str]) -> np.ndarray:
    """Channel records -> (L, 6) float64 array on the common 4 kHz grid.

    64 kHz channels (phase currents, vibration) are anti-alias mean-pooled
    by exactly ``DOWNSAMPLE_FACTOR`` (reshape+mean after trimming to a
    multiple of 16); 4 kHz channels (force, torque, speed) are used as-is;
    ``temp`` and any other channels are dropped.  All six channels are then
    trimmed to their common minimum length, so short signals are accepted
    gracefully.
    """
    ch: dict[str, np.ndarray] = {}
    for name, data in entries:
        key = name.strip().lower()
        if key and key not in ch:
            ch[key] = data
    missing = [c for c in CHANNEL_NAMES_PADERBORN if c not in ch]
    if missing:
        raise ValueError(
            f"'{fname}' is missing channels {missing}; found channels "
            f"{sorted(ch)}")
    pooled: list[np.ndarray] = []
    for c in CHANNELS_64K:
        d = ch[c]
        n = (len(d) // DOWNSAMPLE_FACTOR) * DOWNSAMPLE_FACTOR
        pooled.append(d[:n].reshape(-1, DOWNSAMPLE_FACTOR).mean(axis=1))
    mech = [ch[c] for c in CHANNELS_4K]
    if pooled[0].size and mech[0].size:
        ratio = len(ch[CHANNELS_64K[0]]) / max(len(mech[0]), 1)
        if abs(ratio - DOWNSAMPLE_FACTOR) > 0.5:
            warnings.append(
                f"'{fname}': 64 kHz/4 kHz length ratio is {ratio:.2f} "
                f"(expected ~{DOWNSAMPLE_FACTOR}); channels are still "
                "aligned by trimming to the common length")
    L = min(len(a) for a in pooled + mech)
    return np.stack([a[:L] for a in pooled + mech], axis=1)


def _load_run(path: Path, warnings: list[str]) -> np.ndarray:
    """One .mat run file -> (L, 6) float64 array on the 4 kHz grid."""
    struct = _read_measurement_struct(path)
    y = struct.reshape(-1)[0]["Y"]
    return _assemble_run(_channel_entries(y, path.name), path.name, warnings)


# ---------------------------------------------------------------------------
# stratified run-level split (adapter side; bookkeeping by _build_bundle)
# ---------------------------------------------------------------------------

def _split_counts_for(n: int) -> tuple[int, int, int] | None:
    """Exact (train, val, test) counts = 60/20/20 of ``n`` (None if n < 3)."""
    n_test = max(1, int(round(n * _SPLIT_FRACS[2])))
    n_val = max(1, int(round(n * _SPLIT_FRACS[1])))
    n_train = n - n_test - n_val
    if n_train < 1:
        return None
    return n_train, n_val, n_test


def _stratified_assignment(units: list[tuple[str, str, int]],
                           counts: tuple[int, int, int],
                           split_seed: int, class_index: int) -> list[int]:
    """Assign each unit to a split (0=train, 1=val, 2=test), stratified
    round-robin over (bearing, setting) cells.

    Per split, every cell gets a base quota of ``count // n_cells`` runs and
    the remaining ``count % n_cells`` runs go to seeded-shuffled cells
    (largest remainder); within a cell the runs are seeded-shuffled before
    being dealt to train/val/test.  For the real data (20 runs per cell)
    every cell contributes exactly 12/4/4 runs to train/val/test, so every
    bearing and every operating setting appears in every split.
    """
    rng = np.random.default_rng([split_seed, class_index])
    cells: dict[tuple[str, str], list[int]] = {}
    for i, (bearing, setting, _) in enumerate(units):
        cells.setdefault((bearing, setting), []).append(i)
    cell_keys = sorted(cells)
    n_cells = len(cell_keys)
    # Per-cell quotas, two phases, always respecting each cell's remaining
    # capacity (per-cell quotas then differ by at most one run):
    # phase 1 — PRESENCE: splits whose quota reaches every cell (q >=
    #   n_cells) reserve one run per cell, so every cell appears in every
    #   such split regardless of how the remainders land;
    # phase 2 — REMAINDER: the rest of each quota is dealt one run at a
    #   time, cyclically over the cells in a seeded order rotated per split
    #   (so the +1 remainders spread out).
    cell_order = [cell_keys[int(i)] for i in rng.permutation(n_cells)]
    quotas: dict[tuple[str, str], list[int]] = {k: [0, 0, 0]
                                                for k in cell_keys}
    used = {k: 0 for k in cell_keys}
    for si, q in enumerate(counts):
        if q >= n_cells:
            for key in cell_keys:
                if used[key] < len(cells[key]):
                    quotas[key][si] += 1
                    used[key] += 1
    for si, q in enumerate(counts):
        rem = q - sum(quotas[key][si] for key in cell_keys)
        order = cell_order[si % n_cells:] + cell_order[:si % n_cells]
        dealt = 0
        while dealt < rem:
            progressed = False
            for key in order:
                if dealt >= rem:
                    break
                if used[key] < len(cells[key]):
                    quotas[key][si] += 1
                    used[key] += 1
                    dealt += 1
                    progressed = True
            if not progressed:  # pragma: no cover - counts sum to n units
                raise AssertionError("stratified quotas exceed capacity")
    assign = [-1] * len(units)
    for key in cell_keys:
        idxs = cells[key]
        order = rng.permutation(len(idxs))
        q_tr, q_va, q_te = quotas[key]
        for j, u in enumerate(idxs[int(p)] for p in order):
            assign[u] = 0 if j < q_tr else (1 if j < q_tr + q_va else 2)
    assert all(a >= 0 for a in assign)
    return assign


def partition_bearings(
    bearings_by_code: dict[str, list[str]],
    split_seed: int = 42,
) -> dict[str, dict[str, list[str]]]:
    """Stratified bearing IDs -> train/val/test (work-order Table A.1).

    Permutation uses ``np.random.default_rng(split_seed)`` only, consumed in
    CLASS_CODES order.  Each class must have exactly
    ``sum(BEARING_SPLIT_QUOTAS[code])`` bearings.
    """
    rng = np.random.default_rng(split_seed)
    out: dict[str, dict[str, list[str]]] = {}
    for code in CLASS_CODES:
        bids = sorted(set(bearings_by_code.get(code, [])))
        if not bids:
            continue
        if code not in BEARING_SPLIT_QUOTAS:
            raise ValueError(f"no bearing-split quota for class code {code!r}")
        n_tr, n_va, n_te = BEARING_SPLIT_QUOTAS[code]
        need = n_tr + n_va + n_te
        if len(bids) != need:
            raise ValueError(
                f"bearing-level split for '{code}': expected {need} bearings "
                f"({n_tr}/{n_va}/{n_te}), found {len(bids)}: {bids}")
        order = [bids[int(i)] for i in rng.permutation(len(bids))]
        out[code] = {
            "test": sorted(order[:n_te]),
            "val": sorted(order[n_te:n_te + n_va]),
            "train": sorted(order[n_te + n_va:]),
        }
        got = (len(out[code]["train"]), len(out[code]["val"]),
               len(out[code]["test"]))
        if got != (n_tr, n_va, n_te):
            raise AssertionError(f"{code} bearing counts {got} != {(n_tr, n_va, n_te)}")
    return out


def _place_units_for_build_bundle(
    units: list, arrays: list[np.ndarray], assign: list[int],
    counts: tuple[int, int, int], perm: np.ndarray, li: int,
) -> list[tuple[int, np.ndarray]]:
    """Place runs so ``_build_bundle``'s permutation recovers ``assign``.

    ``assign[u]`` is 0/1/2 = train/val/test.  ``perm`` is the permutation
    ``_build_bundle`` will consume for this class (n >= 2).
    """
    n_tr, n_va, n_te = counts
    n = len(units)
    by_split: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for u, s in enumerate(assign):
        by_split[s].append(u)
    slots = {
        2: sorted(int(p) for p in perm[:n_te]),
        1: sorted(int(p) for p in perm[n_te:n_te + n_va]),
        0: sorted(int(p) for p in perm[n_te + n_va:]),
    }
    placement = [-1] * n
    for s in (0, 1, 2):
        if len(by_split[s]) != len(slots[s]):
            raise AssertionError(
                f"class {li} split {s}: {len(by_split[s])} units vs "
                f"{len(slots[s])} slots (counts={counts})")
        for u, p in zip(by_split[s], slots[s]):
            placement[p] = u
    assert all(u >= 0 for u in placement)
    return [(li, arrays[u]) for u in placement]


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def load_paderborn(root: str, window: int = 128, stride: int = 128,
                   classes: tuple[str, ...] = CLASS_CODES,
                   split_seed: int = 42,
                   split_unit: str = "run") -> DatasetBundle:
    """Load the Paderborn .mat tree under ``root`` -> DatasetBundle.

    Every ``{setting}_{bearing}_{run#}.mat`` file = one 4 s run.  Labels
    come from the bearing code in the file name; ``classes`` selects/orders
    the code letters (default all four).  ``split_unit="run"`` (default) is
    the original 60/20/20 run-level split, stratified over (bearing, setting)
    cells.  ``split_unit="bearing"`` puts every run of a bearing into one
    partition (work-order quotas; ``split_seed=42``).  z-score uses the
    training partition only.  Files that fail to load (e.g. the corrupt
    ``KA08/N15_M01_F10_KA08_2.mat``) are skipped with a warning BEFORE split
    assignment and listed in ``report["excluded_files"]``.
    """
    root_p = Path(root)
    if split_unit not in ("run", "bearing"):
        raise ValueError(
            f"split_unit must be 'run' or 'bearing', got {split_unit!r}")
    for c in classes:
        if c not in _CODE_TO_NAME:
            raise ValueError(
                f"unknown class code '{c}'; expected a subset of "
                f"{list(CLASS_CODES)}")
    classes = tuple(classes)
    class_names = [_CODE_TO_NAME[c] for c in classes]
    warnings: list[str] = []

    # ---- discover run files ----------------------------------------------
    # per class: {unit triple: path}; unknown bearing codes are a hard error
    per_class: dict[int, dict[tuple[str, str, int], Path]] = {
        li: {} for li in range(len(classes))}
    n_ignored = n_dupe = 0
    for p in sorted(root_p.rglob("*.mat")):
        m = _FNAME_RE.match(p.stem)
        if not m:
            n_ignored += 1  # not a run file (e.g. readme/auxiliary .mat)
            continue
        setting, bearing, code, runno = parse_run_filename(p.stem)
        if code not in classes:
            continue  # class not requested
        li = classes.index(code)
        unit = (bearing, setting, runno)
        if unit in per_class[li]:
            n_dupe += 1
            continue
        per_class[li][unit] = p
    if n_ignored:
        warnings.append(f"{n_ignored} .mat file(s) under '{root}' do not "
                        "match the run-file pattern and were ignored")
    if n_dupe:
        warnings.append(f"{n_dupe} duplicate (bearing, setting, run) "
                        "file(s) ignored (first occurrence kept)")

    # ---- load runs ---------------------------------------------------------
    # Each unit is loaded individually inside try/except: a file that fails
    # to load/parse (e.g. the corrupt KA08/N15_M01_F10_KA08_2.mat shipped in
    # the official archive — see the module docstring) is SKIPPED with a
    # warning and listed in report["excluded_files"]; the load does NOT
    # abort.  Exclusion happens HERE, before split assignment, so the exact
    # 60/20/20 split counts and the stratification below are computed from
    # the successfully-loaded runs only.
    class_units: dict[int, list[tuple[str, str, int]]] = {}
    class_arrays: dict[int, list[np.ndarray]] = {}
    excluded_files: list[str] = []
    for li in range(len(classes)):
        units = sorted(per_class[li])
        if not units:
            continue
        ok_units: list[tuple[str, str, int]] = []
        ok_arrays: list[np.ndarray] = []
        for u in units:
            path = per_class[li][u]
            unit_warnings: list[str] = []  # dropped if the load fails
            try:
                arr = _load_run(path, unit_warnings)
            except Exception as e:
                try:
                    rel = str(path.relative_to(root_p))
                except ValueError:  # pragma: no cover - rglob is under root
                    rel = path.name
                excluded_files.append(rel)
                warnings.append(
                    f"corrupt/unreadable file excluded: {rel} "
                    f"({type(e).__name__}: {e})")
                continue
            warnings.extend(unit_warnings)
            ok_units.append(u)
            ok_arrays.append(arr)
        if ok_units:
            class_units[li] = ok_units
            class_arrays[li] = ok_arrays
        else:
            warnings.append(
                f"class '{class_names[li]}': all {len(units)} discovered "
                "run file(s) failed to load and were excluded")
    if not class_units:
        raise ValueError(
            f"no usable Paderborn runs found under '{root}' (expected files "
            "named '{setting}_{bearing}_{run#}.mat', e.g. "
            "'N15_M07_F10_K001_1.mat')")

    # ---- split (run-level 60/20/20, or bearing-level quotas) ---------------
    bearing_to_split: dict[str, int] = {}
    bearing_parts: dict[str, dict[str, list[str]]] = {}
    counts_by_class: dict[int, tuple[int, int, int]] = {}
    if split_unit == "bearing":
        bearings_by_code: dict[str, list[str]] = {}
        for li, units in class_units.items():
            bearings_by_code[classes[li]] = sorted({b for b, _s, _r in units})
        bearing_parts = partition_bearings(bearings_by_code, split_seed)
        for code, d in bearing_parts.items():
            for b in d["train"]:
                bearing_to_split[b] = 0
            for b in d["val"]:
                bearing_to_split[b] = 1
            for b in d["test"]:
                bearing_to_split[b] = 2
        for li, units in class_units.items():
            assign_tmp = [bearing_to_split[u[0]] for u in units]
            counts_by_class[li] = (
                sum(a == 0 for a in assign_tmp),
                sum(a == 1 for a in assign_tmp),
                sum(a == 2 for a in assign_tmp),
            )
    else:
        for li, units in class_units.items():
            counts = _split_counts_for(len(units))
            if counts is None:
                warnings.append(
                    f"class '{class_names[li]}' has only {len(units)} run(s); "
                    "too few for the 60/20/20 split_counts path — falling "
                    "back to _build_bundle's standard split")
            else:
                counts_by_class[li] = counts

    # _build_bundle applies rng.permutation(n) per class (label order,
    # classes with n >= 2) with np.random.default_rng(split_seed).  Replicate
    # that consumption to learn which list position each split slice will
    # draw, and place each run at the position mapping to its assigned
    # split — the net assignment is exactly the intended one.
    rng_repl = np.random.default_rng(split_seed)
    runs: list[tuple[int, np.ndarray]] = []
    assignment_report: dict[str, dict[str, list[str]]] = {}
    for li in sorted(class_units):
        units = class_units[li]
        n = len(units)
        if n < 2:
            runs += [(li, class_arrays[li][u]) for u in range(n)]
            continue
        perm = rng_repl.permutation(n)  # consumed for every class with n>=2
        if li not in counts_by_class:
            runs += [(li, class_arrays[li][u]) for u in range(n)]
            continue
        if split_unit == "bearing":
            assign = [bearing_to_split[u[0]] for u in units]
        else:
            assign = _stratified_assignment(units, counts_by_class[li],
                                            split_seed, li)
        runs += _place_units_for_build_bundle(
            units, class_arrays[li], assign, counts_by_class[li], perm, li)
        by_split: dict[int, list[int]] = {0: [], 1: [], 2: []}
        for u, s in enumerate(assign):
            by_split[s].append(u)
        key = [f"{b}_{s}_{r}" for b, s, r in units]
        assignment_report[class_names[li]] = {
            "train": [key[u] for u in by_split[0]],
            "val": [key[u] for u in by_split[1]],
            "test": [key[u] for u in by_split[2]],
        }

    if split_unit == "bearing":
        warnings.append(
            "bearing-level split: every run of a bearing stays in one "
            "partition (zero bearing overlap across train/val/test)")
        warnings.append(
            "KB bearing split degenerates to 1/1/1 (3 bearings); "
            "do not pool KB with other classes")
    else:
        warnings.append(
            "run-level split: the same bearing appears in train/val/test by "
            "design (consistent with the NPPAD/TEP run-level protocol); "
            "stratified so every bearing and operating setting appears in "
            "every split as evenly as possible — a bearing-held-out variant "
            "is available via split_unit='bearing'")

    bundle = _build_bundle(
        runs, class_names, window, stride, split_seed=split_seed,
        warnings=warnings,
        split_counts={li: c for li, c in counts_by_class.items()} or None)
    extra: dict = {
        "dataset": "Paderborn",
        "root": str(root_p),
        "classes": list(classes),
        "channels": CHANNEL_NAMES_PADERBORN,
        "dropped_channels": list(DROPPED_CHANNELS),
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "fs_hz": FS_TARGET_HZ,
        "split_unit": split_unit,
        "split_counts_per_class": {class_names[li]: list(c)
                                   for li, c in sorted(counts_by_class.items())},
        "runs_per_class": {class_names[li]: len(u)
                           for li, u in sorted(class_units.items())},
        "excluded_files": excluded_files,
        "split_assignment": assignment_report,
        "citation": CITATION,
    }
    if split_unit == "bearing":
        extra["bearing_partition_by_class"] = {
            _CODE_TO_NAME[code]: {
                "train": bearing_parts[code]["train"],
                "val": bearing_parts[code]["val"],
                "test": bearing_parts[code]["test"],
            }
            for code in bearing_parts
        }
        extra["bearing_partition"] = {
            split: sorted(b for d in bearing_parts.values() for b in d[split])
            for split in ("train", "val", "test")
        }
        extra["bearing_split_counts"] = {
            _CODE_TO_NAME[code]: [
                len(bearing_parts[code]["train"]),
                len(bearing_parts[code]["val"]),
                len(bearing_parts[code]["test"]),
            ]
            for code in bearing_parts
        }
        kb = bearing_parts.get("KB")
        extra["kb_degenerate_111"] = bool(
            kb and (len(kb["train"]), len(kb["val"]), len(kb["test"]))
            == (1, 1, 1))
    bundle.report.update(extra)
    return bundle


def make_synthetic_paderborn(n_classes: int = 4, runs_per_class: int = 8,
                             n_samples: int = 1600, window: int = 128,
                             stride: int = 128, seed: int = 0,
                             split_unit: str = "run",
                             runs_per_bearing: int = 2
                             ) -> DatasetBundle:
    """Synthetic stand-in for Paderborn (tests / smoke), mirroring
    ``data_tep.make_synthetic_tep``.

    6 channels (the adapter contract) of class-distinct noisy sinusoids at
    the 4 kHz-grid length ``n_samples``; damaged classes (k > 0) ramp a
    second harmonic in after a class-specific breakpoint and add impulsive
    bursts to the vibration channel (a crude bearing-fault signature).

    ``split_unit="run"`` (default) is the original class-then-run synthetic
    path.  ``split_unit="bearing"`` builds the 32-bearing work-order tree
    in memory (4 classes) and applies ``partition_bearings``.
    """
    if split_unit == "bearing":
        return _make_synthetic_paderborn_bearing(
            n_samples=n_samples, window=window, stride=stride, seed=seed,
            runs_per_bearing=runs_per_bearing)
    if n_classes > len(CLASS_NAMES_PADERBORN):
        class_names = CLASS_NAMES_PADERBORN + [
            f"class_{k}" for k in range(len(CLASS_NAMES_PADERBORN), n_classes)]
    else:
        class_names = CLASS_NAMES_PADERBORN[:n_classes]
    n_channels = len(CHANNEL_NAMES_PADERBORN)
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float64)

    runs: list[tuple[int, np.ndarray]] = []
    for k in range(n_classes):
        f0 = 0.02 + 0.004 * k
        phi = 2.0 * np.pi * k / n_classes
        bp = 96 + (k * 29) % max(n_samples - 192, 1)
        for _ in range(runs_per_class):
            amp = rng.uniform(0.7, 1.3, size=(1, n_channels))
            f_chan = f0 * rng.uniform(0.95, 1.05, size=(1, n_channels))
            phi_chan = phi + rng.uniform(-0.2, 0.2, size=(1, n_channels))
            sig = amp * np.sin(2.0 * np.pi * (f_chan * t[:, None]) + phi_chan)
            if k > 0:  # fault signature: harmonic ramp after the breakpoint
                post = np.clip((t - bp) / max(n_samples - bp, 1),
                               0.0, 1.0)[:, None]
                sig = sig + 0.5 * amp * post * np.sin(
                    2.0 * np.pi * (2.5 * f_chan * t[:, None]) + 2.0 * phi_chan)
                # impulsive bursts on the vibration channel (index 2)
                bursts = (rng.standard_normal(n_samples) > 2.5).astype(
                    np.float64) * post.reshape(-1)
                sig[:, 2] += 2.0 * amp[0, 2] * bursts
            sig = sig + 0.3 * rng.standard_normal((n_samples, n_channels))
            runs.append((k, sig))

    bundle = _build_bundle(runs, class_names, window, stride, split_seed=seed)
    bundle.report.update({"dataset": "Paderborn", "synthetic": True,
                          "channels": CHANNEL_NAMES_PADERBORN,
                          "split_unit": "run"})
    return bundle


def _synth_paderborn_signal(k: int, n_classes: int, n_samples: int,
                            n_channels: int, rng: np.random.Generator
                            ) -> np.ndarray:
    """One synthetic (L, 6) run; same recipe as make_synthetic_paderborn."""
    t = np.arange(n_samples, dtype=np.float64)
    f0 = 0.02 + 0.004 * k
    phi = 2.0 * np.pi * k / n_classes
    bp = 96 + (k * 29) % max(n_samples - 192, 1)
    amp = rng.uniform(0.7, 1.3, size=(1, n_channels))
    f_chan = f0 * rng.uniform(0.95, 1.05, size=(1, n_channels))
    phi_chan = phi + rng.uniform(-0.2, 0.2, size=(1, n_channels))
    sig = amp * np.sin(2.0 * np.pi * (f_chan * t[:, None]) + phi_chan)
    if k > 0:
        post = np.clip((t - bp) / max(n_samples - bp, 1), 0.0, 1.0)[:, None]
        sig = sig + 0.5 * amp * post * np.sin(
            2.0 * np.pi * (2.5 * f_chan * t[:, None]) + 2.0 * phi_chan)
        bursts = (rng.standard_normal(n_samples) > 2.5).astype(
            np.float64) * post.reshape(-1)
        sig[:, 2] += 2.0 * amp[0, 2] * bursts
    return sig + 0.3 * rng.standard_normal((n_samples, n_channels))


def _make_synthetic_paderborn_bearing(
        n_samples: int, window: int, stride: int, seed: int,
        runs_per_bearing: int) -> DatasetBundle:
    """32-bearing synthetic tree + bearing-level split (CLI smoke / tests)."""
    if runs_per_bearing < 1:
        raise ValueError("runs_per_bearing must be >= 1")
    class_names = list(CLASS_NAMES_PADERBORN)
    n_channels = len(CHANNEL_NAMES_PADERBORN)
    rng = np.random.default_rng(seed)
    class_units: dict[int, list[tuple[str, str, int]]] = {}
    class_arrays: dict[int, list[np.ndarray]] = {}
    for li, code in enumerate(CLASS_CODES):
        units: list[tuple[str, str, int]] = []
        arrays: list[np.ndarray] = []
        for bearing in BEARINGS_BY_CLASS[code]:
            for r in range(1, runs_per_bearing + 1):
                units.append((bearing, SETTINGS[0], r))
                arrays.append(_synth_paderborn_signal(
                    li, len(CLASS_CODES), n_samples, n_channels, rng))
        class_units[li] = units
        class_arrays[li] = arrays

    bearings_by_code = {
        code: list(BEARINGS_BY_CLASS[code]) for code in CLASS_CODES}
    parts = partition_bearings(bearings_by_code, split_seed=42)
    bearing_to_split: dict[str, int] = {}
    for code, d in parts.items():
        for b in d["train"]:
            bearing_to_split[b] = 0
        for b in d["val"]:
            bearing_to_split[b] = 1
        for b in d["test"]:
            bearing_to_split[b] = 2

    counts_by_class: dict[int, tuple[int, int, int]] = {}
    for li, units in class_units.items():
        assign_tmp = [bearing_to_split[u[0]] for u in units]
        counts_by_class[li] = (
            sum(a == 0 for a in assign_tmp),
            sum(a == 1 for a in assign_tmp),
            sum(a == 2 for a in assign_tmp),
        )

    rng_repl = np.random.default_rng(42)
    runs: list[tuple[int, np.ndarray]] = []
    assignment_report: dict[str, dict[str, list[str]]] = {}
    for li in sorted(class_units):
        units = class_units[li]
        n = len(units)
        perm = rng_repl.permutation(n)
        assign = [bearing_to_split[u[0]] for u in units]
        runs += _place_units_for_build_bundle(
            units, class_arrays[li], assign, counts_by_class[li], perm, li)
        by_split: dict[int, list[int]] = {0: [], 1: [], 2: []}
        for u, s in enumerate(assign):
            by_split[s].append(u)
        key = [f"{b}_{s}_{r}" for b, s, r in units]
        assignment_report[class_names[li]] = {
            "train": [key[u] for u in by_split[0]],
            "val": [key[u] for u in by_split[1]],
            "test": [key[u] for u in by_split[2]],
        }

    warnings = [
        "synthetic Paderborn bearing-level split (32 bearings)",
        "bearing-level split: every run of a bearing stays in one "
        "partition (zero bearing overlap across train/val/test)",
        "KB bearing split degenerates to 1/1/1 (3 bearings); "
        "do not pool KB with other classes",
    ]
    bundle = _build_bundle(
        runs, class_names, window, stride, split_seed=42,
        warnings=warnings,
        split_counts={li: c for li, c in counts_by_class.items()})
    bundle.report.update({
        "dataset": "Paderborn",
        "synthetic": True,
        "channels": CHANNEL_NAMES_PADERBORN,
        "split_unit": "bearing",
        "split_counts_per_class": {class_names[li]: list(c)
                                   for li, c in sorted(counts_by_class.items())},
        "runs_per_class": {class_names[li]: len(u)
                           for li, u in sorted(class_units.items())},
        "split_assignment": assignment_report,
        "bearing_partition_by_class": {
            _CODE_TO_NAME[code]: {
                "train": parts[code]["train"],
                "val": parts[code]["val"],
                "test": parts[code]["test"],
            }
            for code in parts
        },
        "bearing_partition": {
            split: sorted(b for d in parts.values() for b in d[split])
            for split in ("train", "val", "test")
        },
        "bearing_split_counts": {
            _CODE_TO_NAME[code]: [
                len(parts[code]["train"]),
                len(parts[code]["val"]),
                len(parts[code]["test"]),
            ]
            for code in parts
        },
        "kb_degenerate_111": True,
    })
    return bundle
