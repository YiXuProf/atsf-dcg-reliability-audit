"""NPPAD data loading and synthetic data generation (SPEC.md §data.py).

Conventions
-----------
- Windows are float32 arrays of shape ``(n, C, T)``; labels are int64 ``(n,)``.
- The split unit is a **simulation run** (one CSV file), never a window.
  Runs are stratified per class 70/15/15 with a fixed-seed RNG
  (``np.random.default_rng(42)`` for :func:`load_nppad`, the ``seed`` argument
  for :func:`make_synthetic`), so splits are exactly reproducible.
- **Single-run exception**: a class with exactly one usable run cannot be
  split at run level.  Such a run is split *within-run* by time into
  disjoint segments — proportional 70/15/15 when the run is long enough,
  otherwise a fixed ``window``-length test tail (and val tail when
  affordable) with the remainder for training.  Segments are windowed with
  ``stride=_WITHIN_RUN_STRIDE``; since segments share no timesteps, no
  window pair crosses a split boundary.  Affected classes are listed in
  ``report["within_run_split_classes"]`` and in ``warnings`` — a documented
  protocol exception, not silent leakage.
- Per-channel z-score uses statistics of the **training runs only**; sliding
  windows (default ``window=128, stride=64``) are extracted afterwards.
- ``DatasetBundle.report`` holds, per class, run/window counts per split plus a
  ``warnings`` list (paper §4.1 protocol table).

``CLASS_NAMES`` order defines the label ids 0..17.  Both the list and the
``_LABEL_ALIASES`` table (used to infer labels from CSV relative paths) may be
adjusted to the actual dataset layout without touching the loader logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Label order = ids 0..17 (adjust to the dataset if needed).
CLASS_NAMES: list[str] = [
    "Normal", "ATWS", "FLB", "LACP", "LLB", "LOCA", "LOCAC", "LOF", "LR",
    "MD", "RI", "RW", "SGATR", "SGBTR", "SLBIC", "SLBOC", "SP", "TT",
]

# Extra path-name aliases -> canonical class name.  Keys are matched after
# normalisation (uppercase, non-alphanumeric stripped), so e.g. a directory
# "SGTR loop A" normalises to "SGTRLOOPA" and maps to "SGATR".
_LABEL_ALIASES: dict[str, str] = {
    "NORM": "Normal", "NORMALOPERATION": "Normal", "BASELINE": "Normal",
    "ANTICIPATEDTRANSIENTWITHOUTSCRAM": "ATWS",
    "FEEDWATERLINEBREAK": "FLB", "FWLB": "FLB",
    "LOSSOFACPOWER": "LACP", "LOSSOFAC": "LACP",
    "LETDOWNLINEBREAK": "LLB",
    "LOSSOFCOOLANTACCIDENT": "LOCA",
    "LOSSOFCOOLANTACCIDENTCOLDLEG": "LOCAC", "COLDLEGLOCA": "LOCAC",
    "LOSSOFFLOW": "LOF",
    "LOADREJECTION": "LR", "LOSSOFREGULATION": "LR",
    "MALFUNCTION": "MD", "MALFUNCTIONDISTURBANCE": "MD",
    "REACTIVITYINSERTION": "RI",
    "RODWITHDRAWAL": "RW", "REACTIVITYWITHDRAWAL": "RW",
    "SGTRLOOPA": "SGATR", "SGTRLOOPB": "SGBTR",
    "SGTUBERUPTUREA": "SGATR", "SGTUBERUPTUREB": "SGBTR",
    "STEAMLINEBREAKINSIDECONTAINMENT": "SLBIC", "SLBINSIDECONTAINMENT": "SLBIC",
    "STEAMLINEBREAKOUTSIDECONTAINMENT": "SLBOC", "SLBOUTSIDECONTAINMENT": "SLBOC",
    "SPURIOUS": "SP", "SPURIOUSTRIP": "SP",
    "TURBINETRIP": "TT",
}


def _normalize(text: str) -> str:
    """Uppercase and strip every non-alphanumeric character."""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def _build_alias_map() -> dict[str, str]:
    m = {_normalize(name): name for name in CLASS_NAMES}
    for alias, target in _LABEL_ALIASES.items():
        if target in CLASS_NAMES:
            m[_normalize(alias)] = target
    return m


ALIAS_MAP: dict[str, str] = _build_alias_map()

# Column names (after normalisation) treated as time/index artifacts.
_TIME_LIKE = {
    "TIME", "T", "TIMESTAMP", "TIMESTEP", "STEP", "INDEX", "IDX", "SAMPLE",
    "SAMPLENO", "NO", "ID", "ITER", "ITERATION", "SECOND", "SECONDS", "CLOCK",
}

_ZSCORE_EPS = 1e-8

# Stride used when windowing the disjoint within-run segments of single-run
# classes.  Overlapping windows *within* one split are not leakage (segments
# never share timesteps across splits); a shorter stride merely compensates
# for the small number of windows obtainable from a short run.
_WITHIN_RUN_STRIDE = 32


@dataclass
class DatasetBundle:
    """Windowed, z-scored splits plus the protocol report (SPEC.md §data.py)."""

    X_train: np.ndarray  # (n, C, T) float32, z-scored
    y_train: np.ndarray  # (n,) int64
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]
    report: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# label inference / CSV parsing
# ---------------------------------------------------------------------------

def _match_label(rel_path: Path) -> str | None:
    """Infer the class name from a CSV path relative to the data root.

    Tries (in order) the parent directory name, the file stem, and the leading
    alphabetic prefix of the stem; each candidate is normalised and looked up
    in ``ALIAS_MAP`` (exact match first, then longest-prefix match).
    """
    parts = rel_path.parts
    stem = rel_path.stem
    candidates: list[str] = []
    if len(parts) > 1:
        candidates.append(_normalize(parts[-2]))
    candidates.append(_normalize(stem))
    m = re.match(r"[A-Za-z_\-]+", stem)
    if m:
        candidates.append(_normalize(m.group(0)))

    for cand in candidates:
        if cand in ALIAS_MAP:
            return ALIAS_MAP[cand]
    keys = sorted(ALIAS_MAP, key=len, reverse=True)  # longest key wins
    for cand in candidates:
        if not cand:
            continue
        for k in keys:
            if cand.startswith(k):
                return ALIAS_MAP[k]
    return None


def _read_run_csv(path: Path) -> pd.DataFrame:
    """Read one run: keep numeric sensor columns, drop time/index columns,
    linearly interpolate missing values within the run."""
    df = pd.read_csv(path)
    keep = []
    for c in df.columns:
        nc = _normalize(c)
        if nc in _TIME_LIKE or nc.startswith("UNNAMED"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            keep.append(c)
    df = df[keep] if keep else df.iloc[:, :0]
    df = df.dropna(axis=1, how="all")
    df = df.interpolate(method="linear", limit_direction="both")
    df = df.dropna(axis=1, how="any")  # residual NaNs (should not happen)
    return df


# ---------------------------------------------------------------------------
# split / normalise / window pipeline (shared by load_nppad & make_synthetic)
# ---------------------------------------------------------------------------

def _windowize(arr: np.ndarray, window: int, stride: int) -> np.ndarray:
    """(L, C) -> (n, C, window) float32 with the given stride."""
    L = len(arr)
    if L < window:
        return np.empty((0, arr.shape[1], window), dtype=np.float32)
    starts = np.arange(0, L - window + 1, stride)
    return np.stack([arr[s:s + window].T for s in starts]).astype(np.float32)


def _build_bundle(
    runs: list[tuple[int, np.ndarray]],
    class_names: list[str],
    window: int,
    stride: int,
    split_seed: int,
    warnings: list[str] | None = None,
    split_counts: tuple[int, int, int] | dict[int, tuple[int, int, int]] | None = None,
) -> DatasetBundle:
    """Stratified run-level 70/15/15 split -> train-only z-score -> windows.

    ``split_counts`` (n_train, n_val, n_test) requests EXACT per-class run
    counts instead of proportional 70/15/15 rounding (used by the TEP
    adapter, which pre-selects exactly ``sum(split_counts)`` runs per class).
    It only applies to classes with ``n == sum(split_counts)`` runs (n >= 3);
    every other class follows the standard path below, so the NPPAD default
    (``split_counts=None``) is byte-identical to the historical behaviour.
    Alternatively ``split_counts`` may be a dict ``{class_index:
    (n_train, n_val, n_test)}`` requesting exact counts per class (used by
    the Paderborn adapter, whose classes have different run counts); the
    same ``n == sum(counts)`` applicability rule applies per class.
    """
    warnings = list(warnings or [])
    rng = np.random.default_rng(split_seed)
    per_class = {
        name: dict(train_runs=0, val_runs=0, test_runs=0,
                   train_windows=0, val_windows=0, test_windows=0)
        for name in class_names
    }

    by_class: dict[int, list[np.ndarray]] = {}
    for li, arr in runs:
        by_class.setdefault(li, []).append(arr)

    # Each entry: (label, array, window_stride).  Whole runs use ``stride``;
    # within-run segments of single-run classes use ``window`` (non-overlapping).
    splits: dict[str, list[tuple[int, np.ndarray, int]]] = {"train": [], "val": [], "test": []}
    within_run: list[str] = []
    for li in sorted(by_class):
        rs = by_class[li]
        n = len(rs)
        if n == 1:
            arr = rs[0]
            T = len(arr)
            name = class_names[li]
            b1, b2 = int(T * 0.70), int(T * 0.85)
            if b1 >= window and (b2 - b1) >= window and (T - b2) >= window:
                # long enough for a proportional 70/15/15 time split
                segs = {"train": arr[:b1], "val": arr[b1:b2], "test": arr[b2:]}
                note = "proportional 70/15/15"
            elif T >= 3 * window:
                # short run: fixed window-length val/test tails, rest for train
                segs = {"train": arr[:T - 2 * window],
                        "val": arr[T - 2 * window:T - window],
                        "test": arr[T - window:]}
                note = (f"train={T - 2 * window}/val={window}/test={window} "
                        "timesteps (fixed tails)")
            elif T >= 2 * window:
                # very short run: fixed window-length test tail, no val segment
                segs = {"train": arr[:T - window], "val": None,
                        "test": arr[T - window:]}
                note = (f"train={T - window}/test={window} timesteps "
                        "(fixed test tail, no val)")
            else:
                segs = None
            if segs is not None:
                for split in ("train", "val", "test"):
                    seg = segs[split]
                    if seg is not None and len(seg) >= window:
                        splits[split].append((li, seg, _WITHIN_RUN_STRIDE))
                within_run.append(name)
                warnings.append(
                    f"class '{name}' has only 1 run ({T} timesteps); split "
                    f"within-run by time ({note}), disjoint segments windowed "
                    f"with stride={_WITHIN_RUN_STRIDE} (documented protocol "
                    "exception)")
            else:
                splits["test"].append((li, arr, stride))
                warnings.append(
                    f"class '{name}' has only 1 run ({T} timesteps), too short "
                    "for any within-run split; it goes to test entirely and no "
                    "train/val data remain for this class")
            continue
        order = rng.permutation(n)
        if n >= 3:
            sc = split_counts.get(li) if isinstance(split_counts, dict) \
                else split_counts
            if sc is not None and n == sum(sc):
                n_train, n_val, n_test = sc
                if n_train <= 0 or n_val < 0 or n_test <= 0:
                    raise ValueError(
                        f"split_counts {sc} invalid: need "
                        "n_train > 0 and n_test > 0")
            else:
                if sc is not None:
                    warnings.append(
                        f"class '{class_names[li]}': split_counts "
                        f"{sc} requested but the class has {n} "
                        "runs (expected sum="
                        f"{sum(sc)}); falling back to proportional "
                        "70/15/15")
                n_test = max(1, int(n * 0.15 + 0.5))
                n_val = max(1, int(n * 0.15 + 0.5))
                if n_test + n_val >= n:  # degenerate rounding guard
                    n_test, n_val = 1, 1
        else:  # n == 2
            n_test, n_val = 1, 0
            warnings.append(
                f"class '{class_names[li]}' has only 2 runs; "
                "assigned test=1, train=1, val=0")
        test_i = order[:n_test]
        val_i = order[n_test:n_test + n_val]
        train_i = order[n_test + n_val:]
        for i in train_i:
            splits["train"].append((li, rs[i], stride))
        for i in val_i:
            splits["val"].append((li, rs[i], stride))
        for i in test_i:
            splits["test"].append((li, rs[i], stride))

    missing = [class_names[k] for k in range(len(class_names)) if k not in by_class]
    for name in missing:
        warnings.append(f"class '{name}' has no runs at all")

    # --- per-channel z-score from TRAIN runs only -------------------------
    stats_source = [a for _, a, _ in splits["train"]]
    if not stats_source:
        stats_source = [a for _, a in runs]
        warnings.append("no training runs available; z-score statistics "
                        "computed from ALL runs (protocol violation fallback)")
    cat = np.concatenate(stats_source, axis=0)
    mean = cat.mean(axis=0)
    std = cat.std(axis=0)
    std = np.where(std < _ZSCORE_EPS, 1.0, std)

    def _norm(a: np.ndarray) -> np.ndarray:
        return (a - mean) / std

    # --- windowise ---------------------------------------------------------
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    n_channels = runs[0][1].shape[1]
    for split in ("train", "val", "test"):
        Xs, ys = [], []
        for li, arr, st in splits[split]:
            w = _windowize(_norm(arr), window, st)
            per_class[class_names[li]][f"{split}_runs"] += 1
            per_class[class_names[li]][f"{split}_windows"] += len(w)
            if len(w):
                Xs.append(w)
                ys.append(np.full(len(w), li, dtype=np.int64))
        X = np.concatenate(Xs, axis=0) if Xs else np.empty((0, n_channels, window), np.float32)
        y = np.concatenate(ys, axis=0) if ys else np.empty((0,), np.int64)
        out[split] = (X.astype(np.float32, copy=False), y)

    report = {
        "window": window,
        "stride": stride,
        "split_seed": split_seed,
        "n_runs": len(runs),
        "n_channels": n_channels,
        "per_class": per_class,
        "within_run_split_classes": within_run,
        "warnings": warnings,
    }
    return DatasetBundle(
        X_train=out["train"][0], y_train=out["train"][1],
        X_val=out["val"][0], y_val=out["val"][1],
        X_test=out["test"][0], y_test=out["test"][1],
        class_names=list(class_names), report=report,
    )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def load_nppad(root: str, window: int = 128, stride: int = 64) -> DatasetBundle:
    """Load the NPPAD CSV tree under ``root``.

    Every CSV = one simulation run.  The label is inferred from the relative
    path (parent directory name first, then file-name prefix) after
    normalisation and alias lookup; files with unknown labels, no usable
    numeric columns, or fewer than ``window`` timesteps are skipped and
    counted in ``report["warnings"]``.
    """
    root_p = Path(root)
    warnings: list[str] = []
    parsed: list[tuple[int, pd.DataFrame, str]] = []
    skipped_unknown = skipped_short = skipped_empty = 0

    for p in sorted(root_p.rglob("*.csv")):
        rel = p.relative_to(root_p)
        label = _match_label(rel)
        if label is None:
            skipped_unknown += 1
            warnings.append(f"unknown label for '{rel}'; file skipped")
            continue
        df = _read_run_csv(p)
        if df.shape[1] == 0 or len(df) == 0:
            skipped_empty += 1
            warnings.append(f"no usable numeric columns in '{rel}'; file skipped")
            continue
        if len(df) < window:
            skipped_short += 1
            warnings.append(
                f"run '{rel}' has {len(df)} timesteps < window={window}; skipped")
            continue
        parsed.append((CLASS_NAMES.index(label), df, str(rel)))

    if skipped_unknown:
        warnings.append(f"total files skipped due to unknown label: {skipped_unknown}")
    if not parsed:
        raise ValueError(f"no usable NPPAD runs found under '{root}'")

    # align channels across runs: intersection of column names, first-run order
    cols = list(parsed[0][1].columns)
    colset = set(cols)
    for _, df, _ in parsed[1:]:
        colset &= set(df.columns)
    cols = [c for c in cols if c in colset]
    if len(cols) < len(parsed[0][1].columns):
        warnings.append("runs disagree on columns; using their intersection "
                        f"({len(cols)} channels)")
    runs = [(li, df[cols].to_numpy(dtype=np.float64)) for li, df, _ in parsed]

    bundle = _build_bundle(runs, CLASS_NAMES, window, stride, split_seed=42,
                           warnings=warnings)
    bundle.report["root"] = str(root_p)
    return bundle


def make_synthetic(n_classes: int = 18, runs_per_class: int = 6,
                   window: int = 128, stride: int = 64, n_channels: int = 96,
                   seed: int = 0) -> DatasetBundle:
    """Synthetic stand-in for NPPAD (smoke tests).

    Each class gets a distinct base frequency, phase and transient breakpoint:
    channels are noisy sinusoids at the class frequency; after the breakpoint a
    second harmonic ramps in.  Classes are easily separable but not trivially
    (per-channel frequency jitter, random amplitude, additive Gaussian noise).
    """
    if n_classes > len(CLASS_NAMES):
        class_names = CLASS_NAMES + [f"class_{k}" for k in range(len(CLASS_NAMES), n_classes)]
    else:
        class_names = CLASS_NAMES[:n_classes]
    rng = np.random.default_rng(seed)
    L = 4 * window  # 512 timesteps per run -> 7 windows per run at stride 64
    t = np.arange(L, dtype=np.float64)

    runs: list[tuple[int, np.ndarray]] = []
    for k in range(n_classes):
        f0 = 0.02 + 0.004 * k                     # class-specific base frequency
        phi = 2.0 * np.pi * k / n_classes         # class-specific phase
        bp = 96 + (k * 29) % (L - 192)            # class-specific transient point
        for _ in range(runs_per_class):
            amp = rng.uniform(0.7, 1.3, size=(1, n_channels))
            f_chan = f0 * rng.uniform(0.95, 1.05, size=(1, n_channels))
            phi_chan = phi + rng.uniform(-0.2, 0.2, size=(1, n_channels))
            sig = amp * np.sin(2.0 * np.pi * (f_chan * t[:, None]) + phi_chan)
            # transient: second harmonic ramps in after the breakpoint
            post = np.clip((t - bp) / max(L - bp, 1), 0.0, 1.0)[:, None]
            sig = sig + 0.5 * amp * post * np.sin(
                2.0 * np.pi * (2.5 * f_chan * t[:, None]) + 2.0 * phi_chan)
            sig = sig + 0.3 * rng.standard_normal((L, n_channels))
            runs.append((k, sig))

    return _build_bundle(runs, class_names, window, stride, split_seed=seed)
