"""TEP (Tennessee Eastman Process) data loading adapter (plan v6, Stage 2a).

Source: Rieth et al. (2017), Harvard Dataverse doi:10.7910/DVN/6C3JR1.

PREFERRED INPUT: the original ``.RData`` files
(``TEP_FaultFree_Training.RData`` / ``TEP_Faulty_Training.RData`` /
``TEP_FaultFree_Testing.RData`` / ``TEP_Faulty_Testing.RData``), read with
the pure-Python ``rdata`` package (``pip install rdata`` — no R, no
compilation).  The MathWorks ``.mat`` mirror
(``faultfreetraining.mat`` etc.) is ALSO accepted as a fallback, but note
those files store MATLAB ``table`` objects (MCOS / HDF5 v7.3) which
``scipy.io.loadmat`` cannot fully decode — the .RData path is the reliable
one on Windows.

Layout (each of the four files holds ONE variable — ``fault_free_training``,
``faulty_training``, ``fault_free_testing``, ``faulty_testing`` — either a
MATLAB struct array / record array or a plain 2-D numeric array):

- columns: ``faultNumber``, ``simulationRun``, ``sample``, then
  ``xmeas_1..41`` and ``xmv_1..11`` (52 process variables);
- training files: 500 samples per run, 500 runs per class;
- faults are introduced 1 hour into the run (3-minute sampling): the fault
  is active from sample index 20 (0-based).  To be safe only POST-ONSET
  windows are used for faulty runs: samples with index >= ``onset`` (21),
  i.e. the first 21 samples are dropped before windowing;
- only the two TRAINING files are used (the testing files have a different
  onset and length; they are not needed for the replication protocol and are
  only checked for presence).

Protocol (mirrors ``data.load_nppad`` semantics exactly):

- 18 classes: fault 0 (``Normal``) + faults {1,2,4,5,6,7,8,10,11,12,13,14,
  16,17,18,19,20} — faults 3, 9 and 15 are excluded (not separable /
  standard exclusion in the TEP literature);
- per class, ``sum(runs_per_class)`` runs are selected deterministically
  (evenly spaced over the available run ids) and split run-level stratified
  with ``split_seed`` into EXACTLY ``runs_per_class`` = (train, val, test)
  runs — default (98, 21, 21) of the 500 available;
- per-channel z-score from the TRAINING runs only (post-onset samples for
  faulty runs), sliding windows T=128 stride=128 extracted afterwards:
  3 windows per run -> 98*3*18 = 5292 train / 1134 val / 1134 test windows
  with the defaults;
- the split/window/z-score machinery is ``data._build_bundle`` itself, so
  ``DatasetBundle.report`` has the same fields ``run_experiments`` consumes,
  plus TEP-specific extras (``dataset``, ``onset``, ``runs_per_class``,
  ``files``).

``make_synthetic_tep`` mirrors ``data.make_synthetic`` for tests/smoke runs:
class-distinct noisy sinusoids, 500-sample runs, the same onset trimming for
faulty classes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .data import DatasetBundle, _build_bundle

# 18 classes: Normal + all faults except 3, 9, 15 (literature exclusion).
TEP_FAULT_IDS: list[int] = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17,
                            18, 19, 20]
CLASS_NAMES_TEP: list[str] = ["Normal"] + [f"F{k}" for k in TEP_FAULT_IDS]

CHANNEL_NAMES_TEP: list[str] = (
    [f"xmeas_{k}" for k in range(1, 42)] + [f"xmv_{k}" for k in range(1, 12)]
)

_META_COLS = ("faultNumber", "simulationRun", "sample")

# default column order when a file holds a bare 2-D numeric array
_DEFAULT_COLS: list[str] = list(_META_COLS) + CHANNEL_NAMES_TEP

# expected (candidate file names, preferred variable name) pairs; only the
# training files are loaded, the testing files are only checked for presence.
# .RData (original Rieth release) is preferred; .mat (MathWorks mirror) is a
# fallback (scipy can only decode the plain-struct layouts, not MCOS tables).
_TRAINING_FILES = (
    (("TEP_FaultFree_Training.RData", "faultfreetraining.mat"),
     "fault_free_training"),
    (("TEP_Faulty_Training.RData", "faultytraining.mat"),
     "faulty_training"),
)
_TESTING_FILES = (
    (("TEP_FaultFree_Testing.RData", "faultfreetesting.mat"),
     "fault_free_testing"),
    (("TEP_Faulty_Testing.RData", "faultytesting.mat"),
     "faulty_testing"),
)

# 500 samples/run; fault introduced 1 h in (3-min sampling) = index 20;
# keep only windows fully at sample index >= 21 (defensive post-onset rule)
SAMPLES_PER_RUN = 500
DEFAULT_ONSET = 21


# ---------------------------------------------------------------------------
# .mat parsing (defensive: struct array / record array / plain numeric)
# ---------------------------------------------------------------------------

def _squeeze_field(arr: np.ndarray) -> np.ndarray:
    """(N,1) / (1,N) / () MATLAB field arrays -> 1-D numpy array.

    Object arrays are unwrapped element-wise first: scipy represents a
    struct field either as an object array of N scalars (per-element struct
    arrays) or as a 1-element object array holding one (N,1) sub-array
    (dataframe-like 1x1 structs) — both flatten to the same N-vector here.
    """
    a = np.asarray(arr)
    if a.dtype == object:
        elems = [np.asarray(e).reshape(-1) for e in a.reshape(-1)]
        if not elems:
            return np.asarray([])
        return np.concatenate(elems)
    return a.reshape(-1)


def _var_to_frame(var: np.ndarray, vname: str) -> pd.DataFrame:
    """Convert one loaded .mat variable to a DataFrame with named columns.

    Handles (a) structured/record ndarrays (MATLAB struct arrays — the real
    MathWorks layout), (b) object arrays whose elements are dicts or
    ``np.void`` structs, and (c) plain 2-D numeric arrays (column order
    ``faultNumber, simulationRun, sample, xmeas_1..41, xmv_1..11``).
    """
    var = np.asarray(var)
    if var.dtype.names:  # (a) structured array, possibly shaped (N,1)
        rec = var.reshape(-1)
        data = {name: _squeeze_field(rec[name]) for name in var.dtype.names}
        return pd.DataFrame(data)
    if var.dtype == object:  # (b) object array of dicts / np.void structs
        elems = list(var.reshape(-1))
        if elems and isinstance(elems[0], dict):
            return pd.DataFrame(elems)
        if elems and getattr(elems[0], "dtype", None) is not None \
                and elems[0].dtype.names:
            return pd.DataFrame(
                [{n: _squeeze_field(e[n]) for n in e.dtype.names}
                 for e in elems])
        raise ValueError(
            f"variable '{vname}' is an object array whose elements are "
            f"neither dicts nor structs (first element type: "
            f"{type(elems[0]) if elems else 'empty'})")
    if np.issubdtype(var.dtype, np.number):  # (c) plain numeric matrix
        mat = np.atleast_2d(var)
        if mat.shape[0] < mat.shape[1] and mat.shape[0] == len(_DEFAULT_COLS):
            mat = mat.T  # defensive: column-vector orientation
        if mat.shape[1] == len(_DEFAULT_COLS):
            cols = _DEFAULT_COLS
        elif mat.shape[1] == len(CHANNEL_NAMES_TEP):
            cols = CHANNEL_NAMES_TEP
        else:
            cols = [f"col_{k}" for k in range(mat.shape[1])]
        return pd.DataFrame(mat, columns=cols)
    raise ValueError(f"variable '{vname}' has unsupported dtype {var.dtype}")


def _load_rdata_frame(path: Path, preferred_name: str) -> pd.DataFrame:
    """Load one .RData file via the pure-Python ``rdata`` package.

    Each Rieth file holds a single R data.frame (``fault_free_training`` /
    ``faulty_training`` / ...); ``rdata.conversion.convert`` hands it back
    as a pandas DataFrame with the original column names.  The conversion
    builds the full frame in memory — the faulty training file is ~2.5M
    rows x 55 columns (~1.1 GB as float64), so 8 GB+ RAM is recommended.
    """
    try:
        import rdata  # lazy: only needed for the .RData path
    except ImportError as e:
        raise ImportError(
            "reading TEP .RData files requires the pure-Python 'rdata' "
            "package — install it with:  pip install rdata") from e
    conv = rdata.conversion.convert(rdata.parser.parse_file(str(path)))
    frames = {k: v for k, v in conv.items() if isinstance(v, pd.DataFrame)}
    if not frames:
        raise ValueError(
            f"no data.frame variable found in '{path.name}' "
            f"(variables: {list(conv)[:6]})")
    if preferred_name in frames:
        df = frames[preferred_name]
    else:
        df = next(iter(frames.values()))
        if len(frames) > 1:
            print(f"[tep] warning: '{path.name}' holds {len(frames)} frames; "
                  f"using the first ({next(iter(frames))})", flush=True)
    return df.reset_index(drop=True)


def _load_training_frame(path: Path, preferred_name: str) -> pd.DataFrame:
    """Load one TEP file (.RData preferred, .mat fallback) -> DataFrame."""
    if path.suffix.lower() == ".rdata":
        return _load_rdata_frame(path, preferred_name)
    m = loadmat(str(path))
    keys = [k for k in m if not k.startswith("__")]
    if not keys:
        raise ValueError(f"no data variable found in '{path.name}'")
    key = preferred_name if preferred_name in m else keys[0]
    return _var_to_frame(m[key], key)


def _normalize_columns(df: pd.DataFrame, fname: str) -> pd.DataFrame:
    """Case/whitespace-insensitive column lookup -> canonical names."""
    ren: dict[str, str] = {}
    canon = {c.lower(): c for c in list(_META_COLS) + CHANNEL_NAMES_TEP}
    for c in df.columns:
        key = str(c).strip().lower()
        if key in canon:
            ren[c] = canon[key]
    df = df.rename(columns=ren)
    missing = [c for c in CHANNEL_NAMES_TEP if c not in df.columns]
    if missing:
        raise ValueError(
            f"'{fname}' is missing {len(missing)} channel columns "
            f"(e.g. {missing[:3]}); found columns {list(df.columns)[:8]}...")
    return df


# ---------------------------------------------------------------------------
# run extraction
# ---------------------------------------------------------------------------

def _select_runs(run_ids: np.ndarray, n_select: int) -> np.ndarray:
    """Deterministic EVENLY SPACED selection of ``n_select`` run ids."""
    ids = np.sort(np.unique(run_ids))
    if n_select > len(ids):
        raise ValueError(
            f"requested {n_select} runs but only {len(ids)} available")
    idx = np.linspace(0, len(ids) - 1, num=n_select).round().astype(int)
    return ids[idx]


def _extract_runs(df: pd.DataFrame, fault: int, n_select: int, onset: int,
                  warnings: list[str]) -> list[np.ndarray]:
    """Slice one fault class into per-run (n_samples, 52) float64 arrays.

    Runs shorter than ``SAMPLES_PER_RUN`` after onset trimming are skipped
    with a warning; samples are ordered by the ``sample`` column if present.
    """
    if "faultNumber" in df.columns:
        df = df[df["faultNumber"].to_numpy() == fault]
    elif fault != 0:
        raise ValueError("faulty data frame has no 'faultNumber' column")
    if "simulationRun" not in df.columns:
        raise ValueError("data frame has no 'simulationRun' column")
    run_ids = df["simulationRun"].to_numpy()
    chan = df[CHANNEL_NAMES_TEP].to_numpy(dtype=np.float64)
    order_col = df["sample"].to_numpy() if "sample" in df.columns else None

    runs: list[np.ndarray] = []
    for rid in _select_runs(run_ids, n_select):
        pos = np.flatnonzero(run_ids == rid)
        block = chan[pos]
        if order_col is not None:
            block = block[np.argsort(order_col[pos], kind="stable")]
        if fault != 0:  # faulty run: post-onset samples only
            block = block[onset:]
        if len(block) < 1:
            warnings.append(f"fault {fault} run {rid}: no usable samples")
            continue
        runs.append(block)
    return runs


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def load_tep(root: str, window: int = 128, stride: int = 128,
             runs_per_class: tuple[int, int, int] = (98, 21, 21),
             split_seed: int = 42, onset: int = DEFAULT_ONSET) -> DatasetBundle:
    """Load the MathWorks TEP .mat files under ``root`` -> DatasetBundle.

    Only the two TRAINING files are used.  ``runs_per_class`` = exact
    (train, val, test) run counts per class; ``sum(runs_per_class)`` runs
    are selected deterministically (evenly spaced run ids) per class and
    split run-level stratified with ``split_seed`` via
    ``data._build_bundle(split_counts=runs_per_class)``.  See module
    docstring for the full protocol.
    """
    root_p = Path(root)
    warnings: list[str] = []
    n_select = int(sum(runs_per_class))

    def _resolve(candidates: tuple[str, ...]) -> Path | None:
        for fname in candidates:
            p = root_p / fname
            if p.is_file():
                return p
        return None

    for candidates, _ in _TESTING_FILES:  # presence check only (not loaded)
        if _resolve(candidates) is None:
            warnings.append(
                f"testing file '{candidates[0]}' not found under "
                f"'{root}'; only the training files are used, "
                "so this is informational")

    frames: dict[str, pd.DataFrame] = {}
    used_files: list[str] = []
    for candidates, vname in _TRAINING_FILES:
        path = _resolve(candidates)
        if path is None:
            raise FileNotFoundError(
                f"required TEP file not found under '{root}': none of "
                f"{candidates} (download the .RData files from Harvard "
                "Dataverse doi:10.7910/DVN/6C3JR1 — see data/README.md)")
        frames[vname] = _normalize_columns(
            _load_training_frame(path, vname), path.name)
        used_files.append(path.name)

    runs: list[tuple[int, np.ndarray]] = []
    free = _extract_runs(frames["fault_free_training"], 0, n_select,
                         onset, warnings)
    runs += [(0, a) for a in free]
    faulty = frames["faulty_training"]
    for fault in TEP_FAULT_IDS:
        li = CLASS_NAMES_TEP.index(f"F{fault}")
        rs = _extract_runs(faulty, fault, n_select, onset, warnings)
        if len(rs) < n_select:
            warnings.append(
                f"fault {fault}: only {len(rs)}/{n_select} usable runs")
        runs += [(li, a) for a in rs]

    if not runs:
        raise ValueError(f"no usable TEP runs found under '{root}'")

    bundle = _build_bundle(runs, CLASS_NAMES_TEP, window, stride,
                           split_seed=split_seed, warnings=warnings,
                           split_counts=tuple(int(v) for v in runs_per_class))
    bundle.report.update({
        "dataset": "TEP",
        "root": str(root_p),
        "onset": onset,
        "runs_per_class": [int(v) for v in runs_per_class],
        "runs_selected_per_class": n_select,
        "files": used_files,
    })
    return bundle


def make_synthetic_tep(n_classes: int = 4, runs_per_class: int = 8,
                       n_channels: int = 52, n_samples: int = 500,
                       window: int = 128, stride: int = 128,
                       onset: int = DEFAULT_ONSET, seed: int = 0
                       ) -> DatasetBundle:
    """Synthetic stand-in for TEP (tests / smoke), mirroring make_synthetic.

    Class 0 ("Normal") runs keep the full ``n_samples`` timesteps; faulty
    classes are pre-trimmed to the post-onset region (``onset`` dropped),
    exactly like :func:`load_tep`.  Each class has a distinct base
    frequency/phase; faulty classes additionally ramp a second harmonic in
    after the onset (a crude fault signature).
    """
    if n_classes > len(CLASS_NAMES_TEP):
        class_names = CLASS_NAMES_TEP + [
            f"class_{k}" for k in range(len(CLASS_NAMES_TEP), n_classes)]
    else:
        class_names = CLASS_NAMES_TEP[:n_classes]
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float64)

    runs: list[tuple[int, np.ndarray]] = []
    for k in range(n_classes):
        f0 = 0.02 + 0.004 * k
        phi = 2.0 * np.pi * k / n_classes
        for _ in range(runs_per_class):
            amp = rng.uniform(0.7, 1.3, size=(1, n_channels))
            f_chan = f0 * rng.uniform(0.95, 1.05, size=(1, n_channels))
            phi_chan = phi + rng.uniform(-0.2, 0.2, size=(1, n_channels))
            sig = amp * np.sin(2.0 * np.pi * (f_chan * t[:, None]) + phi_chan)
            if k > 0:  # fault signature: harmonic ramp after the onset
                post = np.clip((t - onset) / max(n_samples - onset, 1),
                               0.0, 1.0)[:, None]
                sig = sig + 0.5 * amp * post * np.sin(
                    2.0 * np.pi * (2.5 * f_chan * t[:, None]) + 2.0 * phi_chan)
            sig = sig + 0.3 * rng.standard_normal((n_samples, n_channels))
            if k > 0:
                sig = sig[onset:]  # post-onset only, as in load_tep
            runs.append((k, sig))

    bundle = _build_bundle(runs, class_names, window, stride, split_seed=seed)
    bundle.report.update({"dataset": "TEP", "synthetic": True,
                          "onset": onset})
    return bundle
