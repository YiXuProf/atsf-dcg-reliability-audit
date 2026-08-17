# Datasets (download into this folder)

This directory is the **only default location** for the three public datasets. After cloning or unzipping this package, put the data in the matching subfolders here — not outside the repository. Do not commit raw data to git.

| Subfolder | Dataset | Used in the paper |
|---|---|---|
| `NuclearPowerPlantAccidentData/` | NPPAD | Cells O / A; Fig. 2–8; Table I–V |
| `TEP/` | Tennessee Eastman Process | Cells B / C |
| `Paderborn/` | Paderborn KAt bearings | Cell D |

Once the files are in place, you do not need to edit code or set environment variables. Use `NPPAD_ROOT`, `TEP_ROOT`, or `PAD_ROOT` only if the data live somewhere else.

For all three datasets: **download the official files and use them as-is.** This package does not convert MDB / R / MATLAB sources into a second on-disk cache. Loaders read the official CSVs / `.RData` / `.mat` files; interpolation, train-split z-score, and windowing happen **in memory**.

---

## 1. NPPAD (nuclear power plant accident simulation)

**What it is:** PCTRAN three-loop PWR simulations, 18 operating classes × several severities, process-variable CSVs.

**Paper / DOI**

- Qi et al., *Scientific Data* 9, 766 (2022)
- https://doi.org/10.1038/s41597-022-01879-1

**Where to get it**

- GitHub (recommended): https://github.com/thu-inet/NuclearPowerPlantAccidentData

**Use the official CSVs as-is.** The GitHub repository already ships `Operation_csv_data/` (one CSV per simulation run). This code reads those CSVs only. You do **not** need the original `.mdb` files, and you do **not** need to run `mdbtocsv` / `Data Processing.py` from the NPPAD repo. After clone or unzip, the CSV tree should already be there.

**How to place it here**

```bash
cd data
git clone https://github.com/thu-inet/NuclearPowerPlantAccidentData.git
```

If you download a zip, extract it into `NuclearPowerPlantAccidentData/` (zips often have a `-main` suffix; rename if needed, or leave the folder under `data/`).

**Layout the loader expects** (present after clone; do not convert MDB):

```
data/NuclearPowerPlantAccidentData/Operation_csv_data/
├── NORM/     1.csv  2.csv  ...     → Normal
├── LOCA/     ...
├── SGATR/    ...
└── ... (18 class folders)
```

Each CSV is one simulation run. Folder names are normalised and matched to
`Normal, ATWS, FLB, LACP, LLB, LOCA, LOCAC, LOF, LR, MD, RI, RW, SGATR, SGBTR, SLBIC, SLBOC, SP, TT`
(`NORM→Normal` and other aliases: `atsf_dcg/data.py`).

---

## 2. TEP (Tennessee Eastman Process)

**What it is:** Tennessee Eastman Process simulations, 52 process variables. This package uses **18 classes**: Normal + faults 1,2,4,5,6,7,8,10–14,16–20 (faults 3, 9, 15 excluded, as in the TEP literature).

**Paper / DOI**

- Rieth et al. (2017), Harvard Dataverse
- https://doi.org/10.7910/DVN/6C3JR1
- Dataset page: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1

**Download these four official `.RData` files and use them as-is**

On the Dataverse page, download the original Rieth files (do not re-export them from MATLAB). This code reads them with `rdata` (already in the repo-root `requirements.txt`). You do **not** need R or MATLAB, and you do not need to convert them to CSV / npy.

| File | Role |
|---|---|
| `TEP_FaultFree_Training.RData` | Used (required) |
| `TEP_Faulty_Training.RData` | Used (required) |
| `TEP_FaultFree_Testing.RData` | Presence check only |
| `TEP_Faulty_Testing.RData` | Presence check only |

Reading `.RData` requires `rdata==0.11.2` (pinned in the repo-root `requirements.txt`).

**Optional fallback: MathWorks `.mat` names** (use them if you already have them; `scipy.io.loadmat` does not fully decode MATLAB table / v7.3 layouts). **Do not** convert `.RData` to `.mat` yourself.

- `faultfreetraining.mat`
- `faultytraining.mat`
- `faultfreetesting.mat`
- `faultytesting.mat`

**Place them here**

```
data/TEP/
├── TEP_FaultFree_Training.RData
├── TEP_Faulty_Training.RData
├── TEP_FaultFree_Testing.RData
└── TEP_Faulty_Testing.RData
```

What the loader does (no extra preprocess script): 18 classes as above; only the two **Training** files are used; faulty runs drop pre-onset samples; run-level split default 98/21/21 per class (from 500 runs); train-only z-score; windows T=128, stride=128.

Check:

```bash
python scripts/probe_tep_mat.py
```

---

## 3. Paderborn (KAt bearings)

**What it is:** University of Paderborn bearing damage data, 32 bearings × 4 operating settings × 20 runs as `.mat` files. This package maps bearing-code letters to 4 classes: K healthy / KA outer ring / KI inner ring / KB both.

**Paper / licence**

- Lessmeier et al., PHM Europe 2016  
  https://doi.org/10.36001/phme.2016.v3i1.1577
- Licence: **CC BY-NC 4.0** (non-commercial research only)
- Data centre: https://groups.uni-paderborn.de/kat/BearingDataCenter/

**What to download (official `.mat`, use as-is)**

32 `.rar` archives (one per bearing). After extraction, each bearing has about 80 `.mat` files (about 2560 files in total). This code reads those `.mat` files directly; do **not** convert them to CSV / npy. The official `KA08` package includes one known-corrupt file `KA08/N15_M01_F10_KA08_2.mat`; the loader skips it — you do not need to delete it by hand.

**Preferred: download into this folder with the repo script** (requires `unrar`, `unar`, or `7z` on the system)

```bash
python scripts/download_paderborn.py
```

You can also open the data-centre page, download each rar by bearing code, and extract to `data/Paderborn/{code}/`.

**Expected layout**

```
data/Paderborn/
├── K001/   N15_M07_F10_K001_1.mat  ...
├── KA01/
├── KI01/
└── ...
```

Size: about **5 GB** after extraction.

What the loader does (no extra preprocess script): 4 classes from the bearing-code letter; 64 kHz channels mean-pooled to 4 kHz; `temp` dropped; run-level 60/20/20 split stratified by (bearing, setting); train-only z-score; windows T=128, stride=128.

Check:

```bash
python scripts/probe_paderborn_mat.py data/Paderborn/K001/N15_M07_F10_K001_1.mat
python scripts/scan_bad_mat.py
```

---

## After the files are in place

```bash
python -m atsf_dcg.run_experiments --dataset nppad --cell full
python -m atsf_dcg.run_experiments --dataset tep --cell replication --arch atsf
python -m atsf_dcg.run_experiments --dataset paderborn --cell replication --arch atsf
```

Without real data, use synthetic smoke tests (not paper numbers):

```bash
python tests/smoke_test.py
```
