"""Tests for the TEP adapter (atsf_dcg/data_tep.py, plan v6 Stage 2a).

Writes REAL .mat fixtures with scipy.io.savemat into a temp dir in three
layouts the loader must handle defensively:

- (A) plain 2-D numeric arrays (column order faultNumber, simulationRun,
  sample, xmeas_1..41, xmv_1..11);
- (B) a 1x1 "dataframe-like" struct whose fields are full (N,1) column
  arrays;
- (C) a per-element MATLAB struct array (record array — small fixture:
  savemat of per-element structs is O(seconds/10k elements)).

Checks: 18-class mapping (faults 3/9/15 excluded), 52 channels, 3 windows
per run at window=stride=128, exact (train,val,test) run split via
runs_per_class, fault-onset trimming (pre-onset samples never windowed),
train-only z-score (val/test stats NOT recomputed), determinism, the exact
98/21/21 split counts at the _build_bundle level, and make_synthetic_tep.

Run from the project root:  ``python tests/test_tep.py``
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.io

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atsf_dcg.data import _build_bundle  # noqa: E402
from atsf_dcg.data_tep import (  # noqa: E402
    CHANNEL_NAMES_TEP, CLASS_NAMES_TEP, SAMPLES_PER_RUN, TEP_FAULT_IDS,
    load_tep, make_synthetic_tep,
)

COLS = ["faultNumber", "simulationRun", "sample"] + CHANNEL_NAMES_TEP
ALL_FAULTS = TEP_FAULT_IDS  # 3/9/15 are never written as classes but ARE
                            # included as decoy rows in one fixture


def _block(rng: np.random.Generator, fault: int, run: int,
           samples: int = SAMPLES_PER_RUN, onset_marker: bool = False
           ) -> np.ndarray:
    """One run: (samples, 55) with faultNumber/simulationRun/sample cols."""
    data = rng.standard_normal((samples, 52)) + fault
    if onset_marker and fault != 0:
        data[:21] = 1.0e6  # pre-onset: must never survive into a window
    return np.column_stack([
        np.full(samples, fault, dtype=np.float64),
        np.full(samples, run, dtype=np.float64),
        np.arange(1, samples + 1, dtype=np.float64),
        data,
    ])


def _write_plain(root: Path, n_runs: int, decoys: bool = False,
                 onset_marker: bool = False, seed: int = 0) -> None:
    """Layout (A): plain 2-D numeric arrays."""
    rng = np.random.default_rng(seed)
    free = np.vstack([_block(rng, 0, r, onset_marker=onset_marker)
                      for r in range(1, n_runs + 1)])
    scipy.io.savemat(root / "faultfreetraining.mat",
                     {"fault_free_training": free})
    faults = list(ALL_FAULTS) + ([3, 9, 15] if decoys else [])
    faulty = np.vstack([_block(rng, f, r, onset_marker=onset_marker)
                        for f in faults for r in range(1, n_runs + 1)])
    scipy.io.savemat(root / "faultytraining.mat",
                     {"faulty_training": faulty})
    # testing files: present but unused (deliberately wrong sample count)
    scipy.io.savemat(root / "faultfreetesting.mat",
                     {"fault_free_testing": free[:10]})
    scipy.io.savemat(root / "faultytesting.mat",
                     {"faulty_testing": faulty[:10]})


def _to_frame_struct(arr: np.ndarray) -> np.ndarray:
    """Layout (B): 1x1 struct whose fields are full (N,1) column arrays."""
    dt = [(c, "f8", (arr.shape[0], 1)) for c in COLS]
    rec = np.zeros(1, dtype=dt)
    for i, c in enumerate(COLS):
        rec[c][0] = arr[:, i:i + 1]
    return rec


def _write_frame_struct(root: Path, n_runs: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    free = np.vstack([_block(rng, 0, r) for r in range(1, n_runs + 1)])
    scipy.io.savemat(root / "faultfreetraining.mat",
                     {"fault_free_training": _to_frame_struct(free)})
    faulty = np.vstack([_block(rng, f, r)
                        for f in ALL_FAULTS for r in range(1, n_runs + 1)])
    scipy.io.savemat(root / "faultytraining.mat",
                     {"faulty_training": _to_frame_struct(faulty)})


def _write_record_struct(root: Path, n_runs: int, samples: int = 150,
                         seed: int = 0) -> None:
    """Layout (C): per-element MATLAB struct array (record array)."""
    rng = np.random.default_rng(seed)

    def rec(arr: np.ndarray) -> np.ndarray:
        return np.rec.fromarrays([arr[:, i] for i in range(arr.shape[1])],
                                 names=",".join(COLS))

    free = np.vstack([_block(rng, 0, r, samples=samples)
                      for r in range(1, n_runs + 1)])
    scipy.io.savemat(root / "faultfreetraining.mat",
                     {"fault_free_training": rec(free)})
    faulty = np.vstack([_block(rng, f, r, samples=samples)
                        for f in ALL_FAULTS for r in range(1, n_runs + 1)])
    scipy.io.savemat(root / "faultytraining.mat",
                     {"faulty_training": rec(faulty)})


def _check_bundle_core(bundle, tag: str, splits=(4, 1, 1), windows=3,
                       n_classes: int = 18) -> None:
    assert bundle.class_names == CLASS_NAMES_TEP, bundle.class_names
    assert len(bundle.class_names) == n_classes
    assert "F3" not in bundle.class_names and "F9" not in bundle.class_names \
        and "F15" not in bundle.class_names
    n_tr, n_v, n_te = splits
    assert bundle.X_train.shape == (n_tr * windows * n_classes, 52, 128), \
        (tag, bundle.X_train.shape)
    assert bundle.X_val.shape == (n_v * windows * n_classes, 52, 128)
    assert bundle.X_test.shape == (n_te * windows * n_classes, 52, 128)
    assert bundle.y_train.dtype == np.int64
    for name in bundle.class_names:
        pc = bundle.report["per_class"][name]
        assert (pc["train_runs"], pc["val_runs"], pc["test_runs"]) == splits, \
            (tag, name, pc)
        assert pc["train_windows"] == n_tr * windows
    assert set(np.unique(bundle.y_train)) == set(range(n_classes))
    rep = bundle.report
    for key in ("window", "stride", "split_seed", "n_runs", "n_channels",
                    "per_class", "within_run_split_classes", "warnings",
                    "dataset", "onset", "runs_per_class"):
        assert key in rep, (tag, key)
    assert rep["dataset"] == "TEP" and rep["n_channels"] == 52
    print(f"[test] OK ({tag}): shapes/splits/report — "
          f"train {bundle.X_train.shape[0]}, val {bundle.X_val.shape[0]}, "
          f"test {bundle.X_test.shape[0]} windows.")


def main() -> None:
    t0 = time.time()

    # ---- (A) plain-array layout, decoy faults 3/9/15, onset markers -------
    tmp = Path(tempfile.mkdtemp(prefix="atsf_dcg_tep_plain_"))
    _write_plain(tmp, n_runs=6, decoys=True, onset_marker=True)
    b = load_tep(str(tmp), runs_per_class=(4, 1, 1))
    _check_bundle_core(b, "plain 2-D arrays")
    # onset trimming: pre-onset 1e6 markers must never enter a window
    for split in ("X_train", "X_val", "X_test"):
        assert np.abs(getattr(b, split)).max() < 1.0e3, \
            f"pre-onset samples leaked into {split}"

    # ---- determinism -------------------------------------------------------
    b_again = load_tep(str(tmp), runs_per_class=(4, 1, 1))
    for s in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"):
        assert np.array_equal(getattr(b, s), getattr(b_again, s)), s
    print("[test] OK: determinism (two load_tep calls identical).")

    # ---- z-score stats from TRAIN runs only --------------------------------
    # 3 constant-valued runs per class (0, 100, 200), split (1,1,1).  Under
    # train-only stats the single constant train run has std 0 -> guarded to
    # 1.0, so train windows are exactly 0 and val/test windows are
    # |v_val - v_train| in {100, 200}.  Under leaky all-data stats the
    # normalized values would be in {0, +-1.22} -- the two hypotheses are
    # disjoint, and the check holds for ANY permutation of the three runs.
    tmp_z = Path(tempfile.mkdtemp(prefix="atsf_dcg_tep_z_"))

    def const_block(value: float, fault: int, run: int) -> np.ndarray:
        samples = SAMPLES_PER_RUN
        data = np.full((samples, 52), value)
        return np.column_stack([
            np.full(samples, fault), np.full(samples, run),
            np.arange(1, samples + 1), data])

    free = np.vstack([const_block(v, 0, r)
                      for r, v in enumerate((0.0, 100.0, 200.0), start=1)])
    faulty = np.vstack([const_block(v, f, r)
                        for f in ALL_FAULTS
                        for r, v in enumerate((0.0, 100.0, 200.0), start=1)])
    scipy.io.savemat(tmp_z / "faultfreetraining.mat",
                     {"fault_free_training": free})
    scipy.io.savemat(tmp_z / "faultytraining.mat",
                     {"faulty_training": faulty})
    bz = load_tep(str(tmp_z), runs_per_class=(1, 1, 1))
    # Replicate _build_bundle's deterministic split assignment (split_seed=42,
    # one rng.permutation(3) per class in class order; slices test/val/train)
    # to learn which raw value each class's train/val run holds.
    rng_split = np.random.default_rng(42)
    raw = [0.0, 100.0, 200.0]  # run ids 1,2,3 in every class
    train_raw, val_raw = [], []
    for _ in CLASS_NAMES_TEP:
        order = rng_split.permutation(3)
        val_raw.append(raw[order[1]])
        train_raw.append(raw[order[2]])
    # stats pool all training-run SAMPLES: the Normal train run contributes
    # 500 samples, faulty train runs 479 (post-onset) -> sample-weighted stats
    w = np.array([float(SAMPLES_PER_RUN)] + [float(SAMPLES_PER_RUN - 21)] * 17)
    tr = np.asarray(train_raw)
    m = float((w * tr).sum() / w.sum())
    s = float(np.sqrt((w * (tr - m) ** 2).sum() / w.sum()))  # ddof=0
    s = s if s >= 1e-8 else 1.0
    ytr, yva = bz.y_train, bz.y_val
    for c in range(len(CLASS_NAMES_TEP)):
        ztr = bz.X_train[ytr == c].astype(np.float64)
        zva = bz.X_val[yva == c].astype(np.float64)
        assert np.allclose(ztr, (train_raw[c] - m) / s, atol=1e-3), c
        assert np.allclose(zva, (val_raw[c] - m) / s, atol=1e-3), c
    # leaky alternative (stats over ALL runs: mean 100, std 81.65) must NOT
    # explain the val windows at the same tolerance
    n_distinguishing = sum(
        not np.allclose(bz.X_val[yva == c].astype(np.float64),
                        (val_raw[c] - 100.0) / np.std(raw), atol=1e-3)
        for c in range(len(CLASS_NAMES_TEP)))
    assert n_distinguishing >= 17, n_distinguishing
    print(f"[test] OK: z-score from train runs only (exact per-class match "
          f"to train-stats hypothesis; {n_distinguishing}/18 classes "
          "distinguish it from the leaky all-data hypothesis).")

    # ---- (B) 1x1 dataframe-like struct layout ------------------------------
    tmp_b = Path(tempfile.mkdtemp(prefix="atsf_dcg_tep_framestruct_"))
    _write_frame_struct(tmp_b, n_runs=6)
    bb = load_tep(str(tmp_b), runs_per_class=(4, 1, 1))
    _check_bundle_core(bb, "1x1 struct with column-array fields")

    # ---- (C) per-element struct array (record) layout ----------------------
    tmp_c = Path(tempfile.mkdtemp(prefix="atsf_dcg_tep_recstruct_"))
    _write_record_struct(tmp_c, n_runs=3, samples=150)
    bc = load_tep(str(tmp_c), runs_per_class=(1, 1, 1))
    # 150 samples -> 1 window/run (faulty: 150-21=129 >= 128; normal: 150)
    _check_bundle_core(bc, "per-element struct array", splits=(1, 1, 1),
                       windows=1)

    # ---- exact 98/21/21 split counts (unit level, _build_bundle) ----------
    names = CLASS_NAMES_TEP[:2]
    rng_u = np.random.default_rng(0)
    runs = [(li, rng_u.standard_normal((149, 2)))
            for li in (0, 1) for _ in range(140)]
    bb2 = _build_bundle(runs, names, window=128, stride=128, split_seed=42,
                        split_counts=(98, 21, 21))
    for name in names:
        pc = bb2.report["per_class"][name]
        assert (pc["train_runs"], pc["val_runs"], pc["test_runs"]) == \
            (98, 21, 21), pc
        assert pc["train_windows"] == 98  # 149 samples -> 1 window/run
    assert bb2.X_train.shape[0] == 2 * 98
    # identical assignment to the proportional path for n=140 (98/21/21 ==
    # 70/15/15 of 140): same rng consumption, same slices
    bb3 = _build_bundle(runs, names, window=128, stride=128, split_seed=42)
    assert np.array_equal(bb2.X_train, bb3.X_train)
    assert np.array_equal(bb2.X_test, bb3.X_test)
    print("[test] OK: exact split_counts (98,21,21) == proportional 70/15/15 "
          "of 140 runs.")

    # ---- make_synthetic_tep ------------------------------------------------
    syn = make_synthetic_tep()
    assert syn.class_names == CLASS_NAMES_TEP[:4]
    assert syn.X_train.shape[1] == 52 and syn.X_train.shape[2] == 128
    syn2 = make_synthetic_tep()
    assert np.array_equal(syn.X_train, syn2.X_train)
    print(f"[test] OK: make_synthetic_tep {syn.X_train.shape} "
          f"(8 runs/class -> {_sum(syn)} windows total).")

    print(f"[test_tep] ALL OK ({time.time() - t0:.1f}s)")


def _sum(bundle) -> int:
    return len(bundle.y_train) + len(bundle.y_val) + len(bundle.y_test)


if __name__ == "__main__":
    main()
