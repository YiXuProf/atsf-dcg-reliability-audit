"""Dataset locations inside this repository.

All public datasets are expected under ``<repo>/data/``. After cloning this
package, download NPPAD / TEP / Paderborn into those folders (see
``data/README.md``). There is no fallback to sibling folders or cloud paths
outside the repo. Override with ``NPPAD_ROOT`` / ``TEP_ROOT`` / ``PAD_ROOT``
only if you must point elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
OUTPUT_ROOT = REPO_ROOT / "output"
FIGURES_ROOT = OUTPUT_ROOT / "figures"
TABLES_ROOT = OUTPUT_ROOT / "tables"
EXPERIMENTS_ROOT = OUTPUT_ROOT / "experiments"
INTERMEDIATE_ROOT = OUTPUT_ROOT / "intermediate"
LOGS_ROOT = OUTPUT_ROOT / "logs"
PAPER_ROOT = OUTPUT_ROOT / "paper"
PAPER_FIGS = PAPER_ROOT / "figures"
PAPER_TABLES = PAPER_ROOT / "tables"

# (cell, dataset, arch) -> folder under output/experiments/
EXPERIMENT_FOLDERS: dict[tuple[str, str, str], str] = {
    ("full", "nppad", "atsf"): "nppad_atsf_full",
    ("replication", "nppad", "tsnet"): "CellA_NPPAD_TimesNet",
    ("replication", "tep", "atsf"): "CellB_TEP_ATSF",
    ("replication", "tep", "tsnet"): "CellC_TEP_TimesNet",
    ("replication", "paderborn", "atsf"): "CellD_Paderborn_ATSF",
    ("replication", "nppad", "atsf"): "CellO_NPPAD_ATSF",
}

# labels used by run_all_cells / extract_per_seed
CELL_FOLDERS: dict[str, str] = {
    "full": "nppad_atsf_full",
    "O": "CellO_NPPAD_ATSF",
    "A": "CellA_NPPAD_TimesNet",
    "B": "CellB_TEP_ATSF",
    "C": "CellC_TEP_TimesNet",
    "D": "CellD_Paderborn_ATSF",
    "D_bearing": "CellD_Paderborn_ATSF_bearing",
    "n30_O": "n30ext_nppad_atsf",
    "n30_A": "n30ext_nppad_tsnet",
}

# export_results / make_figs_v7 CSV tags
CELL_EXPORT_TAGS: dict[str, str] = {
    "nppad_atsf_full": "nppad_atsf_full",
    "CellO_NPPAD_ATSF": "cellO_key",
    "CellA_NPPAD_TimesNet": "nppad_tsnet",
    "CellB_TEP_ATSF": "tep_atsf",
    "CellC_TEP_TimesNet": "tep_tsnet",
    "CellD_Paderborn_ATSF": "paderborn_atsf",
}


def _has_csv(path: Path) -> bool:
    if not path.is_dir():
        return False
    return next(path.glob("*.csv"), None) is not None or next(
        path.glob("*/*.csv"), None) is not None


def nppad_root() -> Path:
    """CSV tree for NPPAD (usually ``data/.../Operation_csv_data``)."""
    env = os.environ.get("NPPAD_ROOT")
    if env:
        return Path(env)
    preferred = DATA_ROOT / "NuclearPowerPlantAccidentData" / "Operation_csv_data"
    # GitHub zip extracts as *-main; still inside data/, not an outside fallback.
    bases = (
        DATA_ROOT / "NuclearPowerPlantAccidentData",
        DATA_ROOT / "NuclearPowerPlantAccidentData-main",
    )
    for base in bases:
        nested = base / "Operation_csv_data"
        if _has_csv(nested):
            return nested
        if _has_csv(base):
            return base
    return preferred


def tep_root() -> Path:
    env = os.environ.get("TEP_ROOT")
    if env:
        return Path(env)
    return DATA_ROOT / "TEP"


def paderborn_root() -> Path:
    env = os.environ.get("PAD_ROOT")
    if env:
        return Path(env)
    return DATA_ROOT / "Paderborn"


def default_data_root(dataset: str) -> Path:
    dataset = dataset.lower()
    if dataset == "nppad":
        return nppad_root()
    if dataset == "tep":
        return tep_root()
    if dataset == "paderborn":
        return paderborn_root()
    raise ValueError(f"unknown dataset {dataset!r}")


def default_out_dir(cell: str, dataset: str, arch: str) -> str:
    """Relative path ``output/experiments/<name>`` (resolved against cwd)."""
    name = EXPERIMENT_FOLDERS.get(
        (cell, dataset, arch), f"{cell}_{dataset}_{arch}")
    return str(Path("output") / "experiments" / name)


def cell_experiment_dir(label: str) -> Path:
    try:
        name = CELL_FOLDERS[label]
    except KeyError as exc:
        raise KeyError(f"unknown cell {label!r}; expected one of "
                       f"{sorted(CELL_FOLDERS)}") from exc
    return EXPERIMENTS_ROOT / name
