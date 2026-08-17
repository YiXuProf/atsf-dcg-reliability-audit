"""Tests for the Paderborn bearing adapter (atsf_dcg/data_paderborn.py).

Builds FAKE .mat trees on the fly with scipy.io.savemat, mimicking the real
per-run struct {Info, X, Y, Description} where Y is a 7-record channel
struct array with Name/Data fields — in BOTH encodings scipy can hand back:

- (A) a structured ndarray with fields Name/Data (MATLAB struct array);
- (B) an object/cell array of dicts (saved as a cell array of structs).

Fixtures are tiny 4 s-equivalents: 64 kHz channels shortened to 3200
samples (-> 200 after mean-pool x16), 4 kHz channels 200 samples, temp 4
samples.  window=stride=128 -> exactly 1 window per run.

Checks: filename/label parsing (incl. KB -> BothRings and unknown-code
errors), channel alignment to (L, 6) with the temp channel dropped,
mean-pool downsample semantics (factor exactly 16, first pooled sample =
mean of first 16 raw), run-level 60/20/20 exact split counts via the
per-class split_counts dict, (bearing, setting) stratification,
determinism across two calls, corrupt-.mat exclusion (the official KA08
archive ships one corrupt file; the loader must skip it with a warning
BEFORE split assignment so counts adapt), train-only z-score (hypothesis
test in the style of test_tep.py), windowing shapes,
make_synthetic_paderborn, and the run_experiments CLI wiring for
--dataset paderborn.

Run from the project root:  ``python tests/test_paderborn.py``
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.io

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atsf_dcg.data_paderborn import (  # noqa: E402
    BEARINGS_BY_CLASS, CHANNEL_NAMES_PADERBORN, CLASS_NAMES_PADERBORN,
    DOWNSAMPLE_FACTOR, _load_run, load_paderborn, make_synthetic_paderborn,
    parse_run_filename, partition_bearings,
)

L64 = 3200            # fake 64 kHz channel length -> 200 after pooling
L4 = L64 // DOWNSAMPLE_FACTOR
CHANS_64K = ("phase_current_1", "phase_current_2", "vibration_1")
CHANS_4K = ("force", "torque", "speed")
SETTINGS2 = ("N15_M07_F10", "N09_M07_F10")


# ---------------------------------------------------------------------------
# fixture writers
# ---------------------------------------------------------------------------

def _chan(rng: np.random.Generator, n: int, value: float | None):
    if value is not None:
        return np.full((n, 1), float(value))
    return rng.standard_normal((n, 1))


def _write_run(path: Path, bearing: str, rng: np.random.Generator,
               encoding: str = "struct", value: float | None = None,
               arange: bool = False) -> None:
    """One fake Paderborn run file: struct {Info, X, Y, Description} with Y
    a 7-record channel struct array (encoding 'struct') or a cell array of
    dicts (encoding 'cell')."""
    chans = []
    for nm in CHANS_64K:
        data = np.arange(L64, dtype=np.float64).reshape(-1, 1) if arange \
            else _chan(rng, L64, value)
        chans.append((nm, data))
    for nm in CHANS_4K:
        data = np.arange(L4, dtype=np.float64).reshape(-1, 1) if arange \
            else _chan(rng, L4, value)
        chans.append((nm, data))
    chans.append(("temp", np.full((4, 1), 36.5)))
    if encoding == "struct":
        Y = np.zeros((1, len(chans)), dtype=[("Name", "O"), ("Data", "O")])
        for i, (nm, d) in enumerate(chans):
            Y[0, i]["Name"], Y[0, i]["Data"] = nm, d
    else:  # cell array of dicts -> scipy loads structs inside an object array
        Y = np.empty((1, len(chans)), dtype=object)
        for i, (nm, d) in enumerate(chans):
            Y[0, i] = {"Name": nm, "Data": d}
    top = np.zeros((1, 1), dtype=[("Info", "O"), ("X", "O"), ("Y", "O"),
                                  ("Description", "O")])
    top[0, 0]["Info"] = "fake Info"
    top[0, 0]["X"] = np.zeros((3, 2))
    top[0, 0]["Y"] = Y
    top[0, 0]["Description"] = "fake Description"
    scipy.io.savemat(path, {bearing: top})


def _write_tree(root: Path, bearings: list[str], settings: tuple[str, ...],
                runs_per_cell: int, encoding: str = "struct",
                value_fn=None, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for b in bearings:
        for s in settings:
            for r in range(1, runs_per_cell + 1):
                v = value_fn(b, s, r) if value_fn is not None else None
                _write_run(root / f"{s}_{b}_{r}.mat", b, rng,
                           encoding=encoding, value=v)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def _check_core(bundle, tag: str, n_runs: int, splits: tuple[int, int, int],
                windows: int = 1, class_names=("Healthy", "OuterRing")):
    assert bundle.class_names == list(class_names), bundle.class_names
    n_tr, n_va, n_te = splits
    nc = len(class_names)
    assert bundle.X_train.shape == (n_tr * windows * nc, 6, 128), \
        (tag, bundle.X_train.shape)
    assert bundle.X_val.shape == (n_va * windows * nc, 6, 128)
    assert bundle.X_test.shape == (n_te * windows * nc, 6, 128)
    assert bundle.X_train.dtype == np.float32
    assert bundle.y_train.dtype == np.int64
    assert set(np.unique(bundle.y_train)) == set(range(nc))
    rep = bundle.report
    for key in ("window", "stride", "split_seed", "n_runs", "n_channels",
                "per_class", "within_run_split_classes", "warnings",
                "dataset", "classes", "channels", "dropped_channels",
                "downsample_factor", "split_counts_per_class",
                "runs_per_class", "excluded_files", "split_assignment",
                "citation"):
        assert key in rep, (tag, key)
    assert rep["dataset"] == "Paderborn" and rep["n_channels"] == 6
    assert rep["n_runs"] == n_runs * nc
    assert rep["channels"] == CHANNEL_NAMES_PADERBORN
    assert "temp" in rep["dropped_channels"] \
        and "temp" not in rep["channels"]
    for name in class_names:
        pc = rep["per_class"][name]
        assert (pc["train_runs"], pc["val_runs"], pc["test_runs"]) == splits, \
            (tag, name, pc)
        assert pc["train_windows"] == n_tr * windows
        assert rep["split_counts_per_class"][name] == list(splits)
        assert rep["runs_per_class"][name] == n_runs
        sa = rep["split_assignment"][name]
        assert (len(sa["train"]), len(sa["val"]), len(sa["test"])) == splits
    assert any("same bearing" in w for w in rep["warnings"]), \
        "leakage-by-design warning missing"
    print(f"[test] OK ({tag}): shapes/splits/report — "
          f"train {bundle.X_train.shape[0]}, val {bundle.X_val.shape[0]}, "
          f"test {bundle.X_test.shape[0]} windows.")


def _test_mean_pool_semantics() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="atsf_dcg_pu_pool_"))
    for enc in ("struct", "cell"):
        p = tmp / f"N15_M07_F10_K001_1.mat"
        rng = np.random.default_rng(0)
        _write_run(p, "K001", rng, encoding=enc, arange=True)
        arr = _load_run(p, [])
        assert arr.shape == (L4, 6), (enc, arr.shape)
        # mean-pool semantics: first pooled sample = mean of first 16 raw
        for ci in range(3):  # 64 kHz channels were arange(3200)
            assert arr[0, ci] == np.arange(16).mean() == 7.5, (enc, ci)
            assert arr[-1, ci] == np.arange(L64 - 16, L64).mean() == 3191.5
            assert len(arr[:, ci]) * DOWNSAMPLE_FACTOR == L64
        for ci in range(3, 6):  # 4 kHz channels used as-is (arange(200))
            assert arr[5, ci] == 5.0 and arr[-1, ci] == float(L4 - 1)
        p.unlink()
    print("[test] OK: downsample factor exactly 16 with mean-pool semantics "
          "(both Y encodings); 4 kHz channels as-is; temp dropped.")


def _test_loader(encoding: str) -> None:
    tmp = Path(tempfile.mkdtemp(prefix=f"atsf_dcg_pu_{encoding}_"))
    # 2 bearings x 2 settings x 5 runs = 20 runs/class -> 60/20/20 = (12,4,4)
    # via the per-class split_counts dict (70/15/15 would give (14,3,3))
    _write_tree(tmp, ["K001", "K002", "KA01", "KA03"], SETTINGS2,
                runs_per_cell=5, encoding=encoding)
    b = load_paderborn(str(tmp), classes=("K", "KA"))
    _check_core(b, f"{encoding} encoding", n_runs=20, splits=(12, 4, 4))
    # labels come from the filename bearing code: K00x -> Healthy (0),
    # KA0x -> OuterRing (1)
    sa = b.report["split_assignment"]
    assert {u.split("_")[0] for u in sa["Healthy"]["train"]} == {"K001", "K002"}
    assert {u.split("_")[0] for u in sa["OuterRing"]["train"]} == {"KA01",
                                                                   "KA03"}
    # determinism across two calls (arrays + assignment)
    b2 = load_paderborn(str(tmp), classes=("K", "KA"))
    for s in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"):
        assert np.array_equal(getattr(b, s), getattr(b2, s)), s
    assert b.report["split_assignment"] == b2.report["split_assignment"]
    print(f"[test] OK ({encoding}): determinism — two load_paderborn calls "
          "byte-identical.")


def _test_stratification() -> None:
    """2 bearings x 2 settings x 6 runs = 24 runs/class -> (14, 5, 5);
    every (bearing, setting) cell must appear in EVERY split."""
    tmp = Path(tempfile.mkdtemp(prefix="atsf_dcg_pu_strat_"))
    bearings = ["K001", "K002", "KA01", "KA03"]
    _write_tree(tmp, bearings, SETTINGS2, runs_per_cell=6)
    b = load_paderborn(str(tmp), classes=("K", "KA"))
    _check_core(b, "stratification", n_runs=24, splits=(14, 5, 5))
    class_bearings = {"Healthy": ["K001", "K002"],
                      "OuterRing": ["KA01", "KA03"]}
    for name, cbs in class_bearings.items():
        all_cells = {(bb, s) for bb in cbs for s in SETTINGS2}
        sa = b.report["split_assignment"][name]
        for split in ("train", "val", "test"):
            cells = set()
            for u in sa[split]:  # key = "{bearing}_{setting}_{run}"
                parts = u.split("_")
                cells.add((parts[0], "_".join(parts[1:4])))
            assert cells == all_cells, (name, split, cells)
    print("[test] OK: stratified split — every (bearing, setting) cell in "
          "every split (14/5/5 of 24 runs per class).")


def _split_cells(sa_entries: list[str]) -> set[tuple[str, str]]:
    """split_assignment keys '{bearing}_{setting}_{run}' -> (bearing, setting)."""
    cells = set()
    for u in sa_entries:
        parts = u.split("_")
        cells.add((parts[0], "_".join(parts[1:4])))
    return cells


def _test_corrupt_file() -> None:
    """One .mat file is garbage (mirrors the corrupt
    KA08/N15_M01_F10_KA08_2.mat shipped in the official archive): the load
    must succeed, skip the file with a warning BEFORE split assignment, and
    adapt the split counts to the successfully-loaded runs."""
    tmp = Path(tempfile.mkdtemp(prefix="atsf_dcg_pu_corrupt_"))
    bearings = ["K001", "K002", "KA01", "KA03"]
    _write_tree(tmp, bearings, SETTINGS2, runs_per_cell=5)
    # corrupt ONE OuterRing run, placed in a per-bearing subdirectory like
    # the real archive layout (KA08/N15_M01_F10_KA08_2.mat)
    bad_rel = "KA03/N09_M07_F10_KA03_2.mat"
    (tmp / "N09_M07_F10_KA03_2.mat").unlink()
    (tmp / "KA03").mkdir()
    (tmp / bad_rel).write_bytes(b"\x00\x01\x02 not a mat file \xff\xfe" * 64)

    b = load_paderborn(str(tmp), classes=("K", "KA"))
    rep = b.report
    # (a) the load succeeded and the bad file was excluded + warning recorded
    got = [Path(p).as_posix() for p in rep["excluded_files"]]
    assert got == [Path(bad_rel).as_posix()], rep["excluded_files"]
    warn = [w for w in rep["warnings"] if "corrupt/unreadable file excluded"
            in w]
    assert len(warn) == 1 and Path(bad_rel).as_posix() in warn[0].replace("\\", "/"), warn
    # (b) split counts adapt: OuterRing has 19 runs -> 60/20/20 = (11,4,4);
    #     Healthy is untouched at 20 runs -> (12,4,4)
    assert rep["runs_per_class"] == {"Healthy": 20, "OuterRing": 19}
    assert rep["split_counts_per_class"]["Healthy"] == [12, 4, 4]
    assert rep["split_counts_per_class"]["OuterRing"] == [11, 4, 4]
    for name, splits in (("Healthy", (12, 4, 4)), ("OuterRing", (11, 4, 4))):
        pc = rep["per_class"][name]
        assert (pc["train_runs"], pc["val_runs"], pc["test_runs"]) == splits
        sa = rep["split_assignment"][name]
        assert (len(sa["train"]), len(sa["val"]), len(sa["test"])) == splits
    assert b.X_train.shape == (23, 6, 128)  # (12 + 11) runs x 1 window
    assert b.X_val.shape == (8, 6, 128) and b.X_test.shape == (8, 6, 128)
    # the excluded unit appears in NO split
    all_assigned = [u for name in rep["split_assignment"]
                    for s in ("train", "val", "test")
                    for u in rep["split_assignment"][name][s]]
    assert not any(u.startswith("KA03_N09_M07_F10_2") for u in all_assigned)
    # (c) stratification still holds: every (bearing, setting) cell in every
    #     split, including the 4-run cell that lost the corrupt file
    class_bearings = {"Healthy": ["K001", "K002"],
                      "OuterRing": ["KA01", "KA03"]}
    for name, cbs in class_bearings.items():
        all_cells = {(bb, s) for bb in cbs for s in SETTINGS2}
        for split in ("train", "val", "test"):
            cells = _split_cells(rep["split_assignment"][name][split])
            assert cells == all_cells, (name, split, cells)
    # (d) determinism: a second load is identical, exclusions included
    b2 = load_paderborn(str(tmp), classes=("K", "KA"))
    for s in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"):
        assert np.array_equal(getattr(b, s), getattr(b2, s)), s
    assert b2.report["split_assignment"] == rep["split_assignment"]
    assert [Path(p).as_posix() for p in b2.report["excluded_files"]] == [
        Path(bad_rel).as_posix()]
    print("[test] OK: corrupt .mat excluded with warning BEFORE split "
          "assignment — counts adapt (OuterRing 19 runs -> 11/4/4, Healthy "
          "unchanged), stratification holds, determinism preserved.")


def _test_zscore_train_only() -> None:
    """Constant-valued runs (distinct value per run): the observed z-scored
    windows must match the train-only-stats hypothesis exactly and NOT the
    leaky all-data hypothesis (style of test_tep.py)."""
    tmp = Path(tempfile.mkdtemp(prefix="atsf_dcg_pu_z_"))
    bearings = ["K001", "KA01"]
    settings4 = ("N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04")
    units = {b: sorted((s, r) for s in settings4 for r in range(1, 6))
             for b in bearings}
    val_of = {b: {u: 10.0 * i for i, u in enumerate(us)}
              for b, us in units.items()}
    _write_tree(tmp, bearings, settings4, runs_per_cell=5,
                value_fn=lambda b, s, r: val_of[b][(s, r)])
    b = load_paderborn(str(tmp), classes=("K", "KA"))
    # _build_bundle pools ALL classes' training-run samples for the stats
    # (equal run lengths here -> plain mean/std over the pooled values)
    def vals(name, bearing, split):
        out = []
        for u in b.report["split_assignment"][name][split]:
            parts = u.split("_")  # key = "{bearing}_{setting}_{run}"
            out.append(val_of[bearing][("_".join(parts[1:4]), int(parts[4]))])
        return np.asarray(out)

    cls = (("Healthy", "K001"), ("OuterRing", "KA01"))
    pooled_tr = np.concatenate([vals(n, br, "train") for n, br in cls])
    m, s = float(pooled_tr.mean()), float(pooled_tr.std())  # ddof=0
    s = s if s >= 1e-8 else 1.0
    pooled_all = np.concatenate(
        [np.asarray(list(val_of[br].values())) for _, br in cls])
    m_all, s_all = float(pooled_all.mean()), float(pooled_all.std())
    n_distinguishing = 0
    for name, bearing in cls:
        v_tr = vals(name, bearing, "train")
        v_va = vals(name, bearing, "val")
        v_te = vals(name, bearing, "test")
        assert len(v_tr) == 12 and len(v_va) == 4 and len(v_te) == 4
        for split, v in (("train", v_tr), ("val", v_va), ("test", v_te)):
            X = getattr(b, f"X_{split}")
            y = getattr(b, f"y_{split}")
            li = b.class_names.index(name)
            z = X[y == li].astype(np.float64)
            expected = np.sort((v - m) / s)
            got = np.sort(z[:, 0, 0])  # constant windows: any channel/time
            assert np.allclose(got, expected, atol=1e-3), \
                (name, split, got, expected)
        # leaky alternative (stats over ALL 40 runs) must NOT explain val
        X, y = b.X_val, b.y_val
        z = X[y == b.class_names.index(name)].astype(np.float64)
        leaky = np.sort((v_va - m_all) / s_all)
        if not np.allclose(np.sort(z[:, 0, 0]), leaky, atol=1e-3):
            n_distinguishing += 1
    assert n_distinguishing == 2, n_distinguishing
    print("[test] OK: z-score from train runs only (exact match to "
          "train-stats hypothesis; 2/2 classes distinguish it from the "
          "leaky all-data hypothesis).")


def _test_metadata_edges() -> None:
    # unknown bearing code -> clear error
    for bad in ("N15_M07_F10_KX01_1", "N15_M07_F10_K1_1",
                "N15_M07_F10_KB1234_1"):
        try:
            parse_run_filename(bad)
        except ValueError as e:
            assert "unknown bearing code" in str(e), str(e)
        else:
            raise AssertionError(f"{bad} should have raised")
    try:
        parse_run_filename("random_file")
    except ValueError as e:
        assert "not a Paderborn run-file name" in str(e)
    else:
        raise AssertionError("malformed name should have raised")
    # KB codes map to BothRings
    s, b, c, r = parse_run_filename("N15_M07_F10_KB23_4")
    assert (s, b, c, r) == ("N15_M07_F10", "KB23", "KB", 4)
    assert parse_run_filename("N09_M07_F10_K001_20")[2] == "K"
    assert parse_run_filename("N15_M01_F10_KI21_7")[2] == "KI"
    assert parse_run_filename("N15_M07_F04_KA30_11")[2] == "KA"
    # full 4-class tree incl. KB -> label 3 = BothRings
    tmp = Path(tempfile.mkdtemp(prefix="atsf_dcg_pu_kb_"))
    _write_tree(tmp, ["K001", "KA01", "KI01", "KB23"], ("N15_M07_F10",),
                runs_per_cell=3)
    b4 = load_paderborn(str(tmp))  # default classes = all four codes
    assert b4.class_names == CLASS_NAMES_PADERBORN
    assert b4.class_names[3] == "BothRings"
    _check_core(b4, "KB -> BothRings (4-class)", n_runs=3, splits=(1, 1, 1),
                class_names=tuple(CLASS_NAMES_PADERBORN))
    sa = b4.report["split_assignment"]["BothRings"]
    assert all("KB23" in u for u in sa["train"] + sa["val"] + sa["test"])
    print("[test] OK: metadata edges — unknown codes rejected, KB maps to "
          "BothRings, 4-class label order intact.")


def _test_synthetic() -> None:
    syn = make_synthetic_paderborn()
    assert syn.class_names == CLASS_NAMES_PADERBORN[:4]
    assert syn.X_train.shape[1] == 6 and syn.X_train.shape[2] == 128
    assert syn.report["dataset"] == "Paderborn" \
        and syn.report["synthetic"] is True
    syn2 = make_synthetic_paderborn()
    assert np.array_equal(syn.X_train, syn2.X_train)
    smoke = make_synthetic_paderborn(n_classes=6, runs_per_class=3,
                                     n_samples=512, seed=0)
    assert smoke.X_train.shape[1] == 6
    print(f"[test] OK: make_synthetic_paderborn {syn.X_train.shape} "
          "(deterministic; smoke variant works).")


def _test_cli_wiring() -> None:
    """End-to-end: --dataset paderborn --synthetic through run_experiments."""
    tmp = Path(tempfile.mkdtemp(prefix="atsf_dcg_pu_cli_"))
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    cmd = [sys.executable, "-m", "atsf_dcg.run_experiments",
           "--cell", "replication", "--dataset", "paderborn",
           "--synthetic", "--epochs", "1", "--seeds", "42",
           "--configs", "full"]
    print(f"[test] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True,
                          env=env)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"CLI failed ({proc.returncode})")
    out_dir = tmp / "output" / "experiments" / "CellD_Paderborn_ATSF"
    assert out_dir.is_dir(), f"expected default out dir {out_dir}"
    proto = json.loads((out_dir / "protocol_report.json").read_text())
    assert proto["dataset"] == "paderborn"
    assert proto["data_source"] == "synthetic_paderborn"
    assert proto["dataset_report"]["dataset"] == "Paderborn"
    assert proto["dataset_report"]["n_channels"] == 6
    assert proto["dataset_report"]["stride"] == 128  # paderborn default
    print("[test] OK: CLI wiring — --dataset paderborn end-to-end "
          "(protocol report + default stride 128 + out-dir resolution).")


def _bearings_in(keys: list[str]) -> set[str]:
    """'{bearing}_{setting}_{run}' keys -> bearing IDs."""
    out = set()
    for u in keys:
        # setting is N##_M##_F##; bearing is the prefix before that
        m = re.search(r"_(N\d+_M\d+_F\d+)_", u)
        if m:
            out.add(u[:m.start()])
        else:
            out.add(u.split("_")[0])
    return out


def _test_split_unit_run_default() -> None:
    """split_unit='run' (explicit) is identical to the default call."""
    tmp = Path(tempfile.mkdtemp(prefix="atsf_dcg_pu_runflag_"))
    bearings = ["K001", "K002", "KA01", "KA03"]
    _write_tree(tmp, bearings, SETTINGS2, runs_per_cell=5)
    a = load_paderborn(str(tmp), classes=("K", "KA"))
    b = load_paderborn(str(tmp), classes=("K", "KA"), split_unit="run")
    for s in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"):
        assert np.array_equal(getattr(a, s), getattr(b, s)), s
    assert a.report["split_assignment"] == b.report["split_assignment"]
    assert a.report.get("split_unit", "run") == "run"
    assert b.report["split_unit"] == "run"
    print("[test] OK: split_unit='run' bit-identical to the default loader.")


def _test_bearing_split() -> None:
    """32 bearings, 1 setting, 2 runs: zero overlap, 19/7/6, KB 1/1/1.

    One KA08 run is corrupt: the bearing stays, that run is skipped.
    """
    tmp = Path(tempfile.mkdtemp(prefix="atsf_dcg_pu_bearing_"))
    all_b = [b for code in ("K", "KA", "KI", "KB")
             for b in BEARINGS_BY_CLASS[code]]
    _write_tree(tmp, all_b, ("N15_M07_F10",), runs_per_cell=2)
    bad = tmp / "N15_M07_F10_KA08_2.mat"
    assert bad.is_file()
    bad.write_bytes(b"\x00\x01 not a mat \xff" * 32)

    bundle = load_paderborn(str(tmp), split_unit="bearing", split_seed=42)
    rep = bundle.report
    assert rep["split_unit"] == "bearing"
    assert rep["kb_degenerate_111"] is True
    got = [Path(p).as_posix() for p in rep["excluded_files"]]
    assert any("KA08_2" in p for p in got), rep["excluded_files"]

    part = rep["bearing_partition"]
    tr, va, te = set(part["train"]), set(part["val"]), set(part["test"])
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    assert len(tr) == 19 and len(va) == 7 and len(te) == 6
    assert (tr | va | te) == set(all_b)

    counts = rep["bearing_split_counts"]
    assert counts["Healthy"] == [4, 1, 1]
    assert counts["OuterRing"] == [7, 3, 2]
    assert counts["InnerRing"] == [7, 2, 2]
    assert counts["BothRings"] == [1, 1, 1]

    # run-level assignment has no bearing overlap either
    sa = rep["split_assignment"]
    b_tr = set()
    b_va = set()
    b_te = set()
    for name in sa:
        b_tr |= _bearings_in(sa[name]["train"])
        b_va |= _bearings_in(sa[name]["val"])
        b_te |= _bearings_in(sa[name]["test"])
    assert b_tr == tr and b_va == va and b_te == te

    # windows: 1 per run (L=200, T=stride=128). Each bearing has 2 runs
    # except KA08 which lost one file.
    windows_per_bearing = {b: 2 for b in all_b}
    windows_per_bearing["KA08"] = 1
    n_tr_w = sum(windows_per_bearing[b] for b in tr)
    n_va_w = sum(windows_per_bearing[b] for b in va)
    n_te_w = sum(windows_per_bearing[b] for b in te)
    assert bundle.X_train.shape[0] == n_tr_w, (
        bundle.X_train.shape[0], n_tr_w)
    assert bundle.X_val.shape[0] == n_va_w
    assert bundle.X_test.shape[0] == n_te_w

    # partition_bearings is deterministic at seed 42
    parts = partition_bearings(
        {code: list(BEARINGS_BY_CLASS[code])
         for code in ("K", "KA", "KI", "KB")},
        split_seed=42)
    assert set(parts["K"]["train"] + parts["K"]["val"] + parts["K"]["test"]
               ) == set(BEARINGS_BY_CLASS["K"])
    print("[test] OK: bearing-level split — 19/7/6 bearings, zero overlap, "
          "KB 1/1/1, corrupt KA08 run skipped, window counts match.")


def _test_cli_split_unit_bearing() -> None:
    """--split-unit bearing smoke writes to the bearing out-dir, not Cell D."""
    tmp = Path(tempfile.mkdtemp(prefix="atsf_dcg_pu_cli_bearing_"))
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    cmd = [sys.executable, "-m", "atsf_dcg.run_experiments",
           "--cell", "replication", "--dataset", "paderborn",
           "--split-unit", "bearing", "--synthetic", "--smoke",
           "--configs", "full"]
    print(f"[test] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True,
                          env=env)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"CLI bearing smoke failed ({proc.returncode})")
    out_dir = tmp / "output" / "experiments" / "CellD_Paderborn_ATSF_bearing"
    assert out_dir.is_dir(), f"expected {out_dir}"
    assert not (tmp / "output" / "experiments" / "CellD_Paderborn_ATSF"
                ).exists(), "bearing run must not write the run-level Cell D dir"
    proto = json.loads((out_dir / "protocol_report.json").read_text())
    assert proto["split_unit"] == "bearing"
    assert proto["dataset_report"]["split_unit"] == "bearing"
    assert proto["dataset_report"]["kb_degenerate_111"] is True
    part = proto["dataset_report"]["bearing_partition"]
    assert set(part["train"]).isdisjoint(part["val"])
    assert set(part["train"]).isdisjoint(part["test"])
    assert set(part["val"]).isdisjoint(part["test"])
    assert len(part["train"]) == 19 and len(part["val"]) == 7
    assert len(part["test"]) == 6
    print("[test] OK: CLI --split-unit bearing smoke -> "
          "CellD_Paderborn_ATSF_bearing (Cell D untouched).")


def _test_cli_split_unit_bearing_rejected_on_nppad() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="atsf_dcg_pu_cli_bad_"))
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    cmd = [sys.executable, "-m", "atsf_dcg.run_experiments",
           "--dataset", "nppad", "--split-unit", "bearing",
           "--synthetic", "--smoke"]
    proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True,
                          env=env)
    assert proc.returncode != 0
    blob = proc.stderr + proc.stdout
    assert "paderborn" in blob.lower()
    print("[test] OK: --split-unit bearing rejected on --dataset nppad.")


def main() -> None:
    t0 = time.time()
    _test_mean_pool_semantics()
    _test_loader("struct")
    _test_loader("cell")
    _test_stratification()
    _test_corrupt_file()
    _test_zscore_train_only()
    _test_metadata_edges()
    _test_synthetic()
    _test_split_unit_run_default()
    _test_bearing_split()
    _test_cli_wiring()
    _test_cli_split_unit_bearing()
    _test_cli_split_unit_bearing_rejected_on_nppad()
    print(f"[test_paderborn] ALL OK ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
