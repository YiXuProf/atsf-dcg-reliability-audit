# ATSF-DCG: five-cell reliability audit

Companion code for the manuscript:

> Yi Xu. _Silent Failures of Adaptive Mechanisms in Data-Driven Nuclear Power Plant Transient Diagnosis: A Component-Level Reliability Audit Across Simulated and Measured Data._ (under review)

**Author:** Yi Xu — School of Information Engineering, Hunan Industry Polytechnic, Changsha, Hunan 410208, China  
**Email:** [Yi.Xu.Prof@outlook.com](mailto:Yi.Xu.Prof@outlook.com) · **ORCID:** [0009-0002-3789-8136](https://orcid.org/0009-0002-3789-8136)

This repository contains the model and experiment code, already-run paper artifacts (`output/`), and `main.py` to re-export **Fig. 1–13** and **Table I–VI / S1–S15** (including S8a). The three public datasets are **not** redistributed (copyright / licence). Download them into `data/` as described below.

Figure generators are named by **manuscript number** (`scripts/make_fig01_…` … `make_fig13.py`). Their outputs use the **same stems** as `output/paper/figures/` (e.g. `fig10_early_indicator_trajectories`).

```
.
├── atsf_dcg/          # installable core library (model, loaders, training, CLI)
├── scripts/           # make_figXX.py, table export, Paderborn download, n30 merge
├── tests/             # synthetic smoke tests (not paper numbers)
├── data/              # download the three datasets here (empty by default)
├── output/            # already-run experiment artifacts; main.py writes here too
│   ├── paper/         # manuscript-numbered figures & tables (preferred)
│   └── figures/       # same stems as paper/ (written by make_figXX.py)
├── main.py            # paper figure/table entry point
├── requirements.txt
└── LICENSE            # MIT
```

---

## 1. Environment and dependencies

- Python **3.10+**
- Paper-scale training: **NVIDIA GPU** + CUDA PyTorch. CPU is enough to redraw existing figures and tables.
- Windows / Linux / macOS. Run all commands from the **repository root** (the folder that contains `main.py`).

```bash
git clone https://github.com/YiXuProf/atsf-dcg-reliability-audit.git
cd atsf-dcg-reliability-audit
python -m venv .venv

# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -U pip
pip install -r requirements.txt
```

Versions in `requirements.txt` are pinned (`==`). Do not loosen them to `>=`. The file installs `torch==2.13.0`, `numpy==2.2.6`, `pandas==2.3.3`, `scipy==1.15.3`, `scikit-learn==1.7.2`, `matplotlib==3.10.8`, plus `rdata==0.11.2` / `xarray==2025.6.1` for TEP `.RData` files.

The PyPI wheel for `torch==2.13.0` is CPU. For GPU training, install the **same version** CUDA build from https://pytorch.org; do not change the version number. CPU is enough to redraw existing figures and tables.

Downloading and extracting Paderborn also needs `unrar`, `unar`, or `7z` on the system (Windows: [7-Zip](https://www.7-zip.org/)).

After install, run a smoke test that does not need real data:

```bash
python tests/smoke_test.py
```

A passing run means the code and dependencies work. The smoke test uses synthetic data and **must not** be reported as paper numbers.

---

## 2. Download datasets into `data/`

Do not commit raw data to git, and do not redistribute it with this package. After cloning or unzipping, download the three public datasets into the matching folders under **`data/`**. Layout, aliases, and probe scripts: **[`data/README.md`](data/README.md)**.

| Dataset   | Put it here                           | Used for                         |
| --------- | ------------------------------------- | -------------------------------- |
| NPPAD     | `data/NuclearPowerPlantAccidentData/` | Cells O / A; Fig. 2–8; Table I–V |
| TEP       | `data/TEP/`                           | Cells B / C                      |
| Paderborn | `data/Paderborn/`                     | Cell D                           |

Once the files are in place, you do not need to edit paths or set environment variables. Use `NPPAD_ROOT` / `TEP_ROOT` / `PAD_ROOT` only if the data live somewhere else.

For all three datasets: **download the official files and leave them as they are.** This package does not convert MDB/R/MATLAB sources into a second on-disk cache. Loaders read the official CSVs / `.RData` / `.mat` files and do interpolation, train-split z-score, and windowing **in memory**.

### 2.1 NPPAD (nuclear power plant accident simulation)

- Paper: Qi et al., _Scientific Data_ 9, 766 (2022), https://doi.org/10.1038/s41597-022-01879-1
- GitHub: https://github.com/thu-inet/NuclearPowerPlantAccidentData

**Use the official CSVs as-is.** The GitHub repository already contains `Operation_csv_data/` (one CSV per simulation run). This code reads those files directly. You do **not** need the original `.mdb` files, and you do **not** need to run an MDB→CSV converter (`mdbtocsv` / `Data Processing.py` in the NPPAD repo).

```bash
cd data
git clone https://github.com/thu-inet/NuclearPowerPlantAccidentData.git
cd ..
```

If you download a zip, extract it into `data/NuclearPowerPlantAccidentData/` (zips often have a `-main` suffix; rename if needed). After clone/unzip you should already see:

```
data/NuclearPowerPlantAccidentData/Operation_csv_data/
├── NORM/     1.csv  2.csv  ...
├── LOCA/
└── ... (18 class folders)
```

That tree is the only NPPAD input this package uses. Do not run any extra local preprocess script to “convert” or cache the CSVs.

### 2.2 TEP (Tennessee Eastman Process)

- Rieth et al. (2017), Harvard Dataverse  
  https://doi.org/10.7910/DVN/6C3JR1  
  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1

**Use the official `.RData` files as-is.** On the Dataverse page, download the four original Rieth files (not a custom export). Put them in `data/TEP/`. This code reads them with `rdata` (already in `requirements.txt`). You do **not** need R, MATLAB, or a conversion to CSV/npy.

| File                           | Role                |
| ------------------------------ | ------------------- |
| `TEP_FaultFree_Training.RData` | Used (required)     |
| `TEP_Faulty_Training.RData`    | Used (required)     |
| `TEP_FaultFree_Testing.RData`  | Presence check only |
| `TEP_Faulty_Testing.RData`     | Presence check only |

```
data/TEP/
├── TEP_FaultFree_Training.RData
├── TEP_Faulty_Training.RData
├── TEP_FaultFree_Testing.RData
└── TEP_Faulty_Testing.RData
```

Optional fallback if you already have the MathWorks `.mat` names (`faultfreetraining.mat` etc.): the loader will try them, but `.RData` is the reliable path. Do not convert `.RData` to `.mat` yourself.

What the loader does (you do not run a preprocess script): 18 classes = Normal + faults 1,2,4,5,6,7,8,10–14,16–20 (faults 3, 9, 15 dropped, literature convention); only **training** files are used; faulty runs drop pre-onset samples; run-level split default 98/21/21 of 500 runs per class; train-only z-score; windows T=128, stride=128.

Check:

```bash
python scripts/probe_tep_mat.py
```

### 2.3 Paderborn (KAt bearings, CC BY-NC 4.0)

- Lessmeier et al., PHM Europe 2016, https://doi.org/10.36001/phme.2016.v3i1.1577
- Data centre: https://groups.uni-paderborn.de/kat/BearingDataCenter/
- Licence: **non-commercial research only**. About **5 GB** after extraction.

**Use the official `.mat` files as-is.** Each of the 32 bearings is one `.rar` on the data-centre page. Extract them; this code reads the `.mat` files directly. You do **not** convert them to CSV or npy.

Preferred (download + extract into this repo; needs `unrar` / `unar` / `7z`):

```bash
python scripts/download_paderborn.py
```

Or download the 32 `.rar` archives in a browser and extract to `data/Paderborn/{code}/`. Expected layout:

```
data/Paderborn/
├── K001/   N15_M07_F10_K001_1.mat  ...
├── KA01/
├── KI01/
└── ... (32 bearing codes)
```

The official `KA08` package includes one known-corrupt file `KA08/N15_M01_F10_KA08_2.mat`. The loader skips it; do not delete files by hand.

What the loader does (you do not run a preprocess script): 4 classes from the bearing-code letter (K healthy / KA outer / KI inner / KB both); 64 kHz channels are mean-pooled down to 4 kHz; `temp` is dropped; run-level 60/20/20 split stratified by (bearing, setting); train-only z-score; windows T=128, stride=128.

Check:

```bash
python scripts/probe_paderborn_mat.py data/Paderborn/K001/N15_M07_F10_K001_1.mat
python scripts/scan_bad_mat.py
```

---

## 3. Produce all figures and tables

The entry point is **`main.py`** at the repository root. It **does not train**. It reads `output/experiments/`, `output/tables/`, and `output/intermediate/eval_dump/`, then writes PNG/SVG and CSV files named as in the manuscript.

This package already includes a completed `output/`. After installing dependencies you can redraw figures and tables **without downloading datasets** (except the raw-waveform t-SNE panel of Fig. 6; see below).

```bash
python main.py              # catalog of Fig. 1–13 / Table I–VII / S1–S15
python main.py all          # all figures + tables (skips only Table VII)
python main.py n30          # supplementary n=30 merge + bearing summary (§5)
```

Outputs:

```
output/paper/figures/     fig01_….png / .svg  …
output/paper/tables/      Table_I_….csv  Table_S1_….csv  …
```

Figure scripts also refresh `output/figures/` using the **same filenames** as `output/paper/figures/`. Table export rebuilds `output/tables/` from experiment directories first (skip with `--no-refresh`).

### Emit by id

```bash
python main.py fig 1              # manuscript Fig. 1 only
python main.py fig 1 12 13
python main.py fig 3-7
python main.py fig 10             # early-stop trajectories (D1/D2 labels)
python main.py table I            # Table I only
python main.py table VI S1
python main.py table all
python main.py all --no-refresh   # copy/plot only; do not rebuild summary CSVs
python main.py n30
python main.py n30 --smoke
```

Or call generators directly (stems match paper):

```bash
python scripts/make_fig01_architecture_audit_map.py
python scripts/make_fig02_rho_trajectories.py
python scripts/make_fig03_07_runlevel.py
python scripts/make_fig08_diagnostics.py
python scripts/make_fig09.py
python scripts/make_fig10.py
python scripts/make_fig11.py
python scripts/make_fig12_decision_workflow.py
python scripts/make_fig13.py
```

### Script ↔ manuscript figure

| Fig. | Script                                 | Output stem (paper + `output/figures/`)               |
| ---- | -------------------------------------- | ----------------------------------------------------- |
| 1    | `make_fig01_architecture_audit_map.py` | `fig01_architecture_audit_map`                        |
| 2    | `make_fig02_rho_trajectories.py`       | `fig02_rho_trajectories`                              |
| 3–7  | `make_fig03_07_runlevel.py`            | `fig03_fusion_weight_dist` … `fig07_confusion_matrix` |
| 8    | `make_fig08_diagnostics.py`            | `fig08_diagnostics`                                   |
| 9    | `make_fig09.py`                        | `fig09_ablation_forest`                               |
| 10   | `make_fig10.py`                        | `fig10_early_indicator_trajectories`                  |
| 11   | `make_fig11.py`                        | `fig11_regime_map`                                    |
| 12   | `make_fig12_decision_workflow.py`      | `fig12_decision_workflow`                             |
| 13   | `make_fig13.py`                        | `fig13_degradation_heatmap`                           |

Shared helpers for Figs. 9/10/11/13 live in `scripts/five_cell_figs.py`. The old name `make_figs_v7.py` is a deprecated forwarder only.

### What each item needs

| Id                | Content                                                            | Needs                                                 |
| ----------------- | ------------------------------------------------------------------ | ----------------------------------------------------- |
| Fig. 1            | Architecture / failure-mode map                                    | No data                                               |
| Fig. 2            | ρ training trajectories                                            | `output/experiments/nppad_atsf_full/rho_curve.csv`    |
| Fig. 3–5, 7       | Fusion weights, gates, confusion matrix                            | `output/intermediate/eval_dump/full_seed42/`          |
| Fig. 6            | t-SNE (features + raw waveforms)                                   | Same dump; the raw panel also needs NPPAD on disk     |
| Fig. 8            | Diagnostic four-panel                                              | `nppad_atsf_full` results / remedy tables             |
| Fig. 9–11, 13     | Five-cell ablation / early-stop / regime map / degradation heatmap | `output/tables/`                                      |
| Fig. 12           | D1–D5 screening workflow                                           | No data                                               |
| Table I           | NPPAD per-class run-level split                                    | `nppad_atsf_full/protocol_report.json`                |
| Table II          | Ablation-as-diagnosis (acc / F1 + tests vs full)                   | `nppad_atsf_full` results + significance              |
| Table III         | Remedy evaluation (last rows = controls)                           | same                                                  |
| Table IV          | Mechanism indicators H(α), ρ, S(τ), perm-null z                    | `nppad_atsf_full/remedy_table.csv`                    |
| Table V           | Planned control contrasts / TOST                                   | `output/tables/reviewer_controls/`                    |
| Table VI          | Five-cell recurrence (results + significance)                      | `output/tables/five_cell_summary/`                    |
| Table VII         | D1–D5 rules                                                        | Manuscript text; `main.py` skips it                   |
| Tables S1–S5      | Degradation matrices, Cells O / A / B / C / D                      | five-cell `degradation_table_*.csv` + per-seed finals |
| Table S6          | Cell O v7 rerun vs original-paper grid                             | `nppad_atsf_full` and `CellO_NPPAD_ATSF`              |
| Table S7          | Spearman: epoch-3 indicators vs snr10 drop                         | `epoch_indicators_*.csv` + `finals_*.csv`             |
| Table S8a, S8–S11 | Per-seed Acc / F1 (%) for Cells O / A / B / C / D                  | `output/tables/per_seed_finals/per_seed_raw.csv`      |
| Tables S12–S15    | Full-model per-seed ρ / H(α) / Var_t(α) / perm-null z              | same                                                  |

Fig. 3–7 share one run of `make_fig03_07_runlevel.py`. Without NPPAD locally, Fig. 6 still draws the feature t-SNE and skips the raw-waveform panel.

**Naming note:** remedy codes **R1–R4** and deployment rules **D1–D5** are different. Fig. 10 panel text refers to **D1/D2** (regime / dominance thresholds), not remedies R1/R2.

---

## 4. Train from scratch (optional; to reproduce numbers)

To rerun experiments instead of using the shipped `output/`, download the data (section 2) and train. The paper setting is **10 seeds (42–51)**, early stopping, and sensor-degradation evaluation. The full five-cell pipeline is long and needs a GPU.

All five cells (O / A / B / C / D) in one go:

```bash
python scripts/run_all_cells.py
```

After a partial failure, skip finished cells, e.g. skip A and B:

```bash
python scripts/run_all_cells.py A B
```

Or run datasets separately:

```bash
# NPPAD × ATSF full grid (default: output/experiments/nppad_atsf_full)
python -m atsf_dcg.run_experiments --dataset nppad --cell full --degradation --log-epoch-indicators

# Five-cell replication grids
python -m atsf_dcg.run_experiments --dataset nppad --cell replication --arch tsnet --degradation --log-epoch-indicators
python -m atsf_dcg.run_experiments --dataset tep --cell replication --arch atsf --degradation --log-epoch-indicators
python -m atsf_dcg.run_experiments --dataset tep --cell replication --arch tsnet --degradation --log-epoch-indicators
python -m atsf_dcg.run_experiments --dataset paderborn --cell replication --arch atsf --degradation --log-epoch-indicators
```

Default seeds are `42 43 44 45 46`. To match the paper’s 10-seed setting, add:

```bash
--seeds 42 43 44 45 46 47 48 49 50 51
```

Fig. 3–7 also need an evaluation dump (retrain `full`, seed 42, on NPPAD; writes α / gates / features):

```bash
python -m atsf_dcg.eval_dump --config full --seed 42 --out-dir output/intermediate/eval_dump
```

After training:

```bash
python main.py all
```

This exports tables from the new `output/experiments/` and redraws every figure.

---

## 5. Supplementary experiments (bearing split + n=30)

These are **new** runs. They must not overwrite `output/experiments/CellD_Paderborn_ATSF`, `nppad_atsf_full`, or the other Cell\* folders. **Always** give smoke runs a throwaway `--out-dir` (default NPPAD×ATSF smoke would otherwise write into `nppad_atsf_full`). Pin `torch==2.13.0` (same CUDA wheel as section 1). Stop and report versions if a rerun of an old seed drifts by more than **1 pp**.

Code comments label Paderborn KB as `BothRings`. Do not change that string.

`python main.py all` still exports the **n=10** manuscript tables. After the runs below:

```bash
python main.py n30              # preferred: reanalyze_n30 + bearing summary
python main.py n30 --smoke
```

Writes `output/analysis_n30/` and does **not** rewrite paper Table II–VI.

### 5.1 Cell D bearing-level split

Default `--split-unit run` is the paper Cell D protocol (same bearing in train and test). `--split-unit bearing` holds out whole bearings (19 / 7 / 6, `split_seed=42`). KB is 1/1/1 by design.

```bash
# smoke (dedicated out-dir)
python -m atsf_dcg.run_experiments --cell replication --dataset paderborn --arch atsf \
  --split-unit bearing --synthetic --smoke --configs full \
  --out-dir output/experiments/_smoke_cellD_bearing

# real data: new directory only
python -m atsf_dcg.run_experiments --cell replication --dataset paderborn --arch atsf \
  --split-unit bearing --seeds 42 43 44 45 46 47 48 49 50 51 \
  --degradation --log-epoch-indicators \
  --out-dir output/experiments/CellD_Paderborn_ATSF_bearing
```

The 7 replication configs are `full`, `w/o_spectral`, `w/o_temporal`, `w/o_fusion`, `w/o_gating`, `full_r1`, `full_r2_gumbel`.

### 5.2 Seed expansion (52–71) on Cell O / Cell A

Do **not** pass n=30 priority config names together with `--smoke` on the default full cell (smoke only keeps `full` / `full_r2` / `full_r4_lstm`).

```bash
# Cell O smoke
python -m atsf_dcg.run_experiments --synthetic --smoke --dataset nppad --arch atsf \
  --out-dir output/experiments/_smoke_n30ext_atsf

# Cell O real data
python -m atsf_dcg.run_experiments --dataset nppad --arch atsf \
  --seeds 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 \
  --configs full "w/o_spectral" "w/o_gating" full_r2_gumbel \
  --degradation --log-epoch-indicators \
  --out-dir output/experiments/n30ext_nppad_atsf

# Cell A smoke
python -m atsf_dcg.run_experiments --cell replication --dataset nppad --arch tsnet \
  --synthetic --smoke \
  --out-dir output/experiments/_smoke_n30ext_tsnet

# Cell A real data
python -m atsf_dcg.run_experiments --cell replication --dataset nppad --arch tsnet \
  --seeds 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 \
  --configs full "w/o_spectral" "w/o_gating" full_r2_gumbel \
  --degradation --log-epoch-indicators \
  --out-dir output/experiments/n30ext_nppad_tsnet
```

Optional Cell O extras if compute allows: add `full_r3_sinc` `full_r1_r3`. Do **not** expand `full_fixed_global` / `full_fixed_class`.

Then merge:

```bash
python main.py n30
```

`output/analysis_n30/` holds `tableII_n30.csv`, `tableIII_n30.csv`, `tableIV_n30.csv`, `tableVI_cellO_n30.csv`, `tableVI_cellA_n30.csv`, Holm/TOST family CSV, `d1_d2_recalibration.csv`, and `tier_changes.md`. Bearing vs run note: `output/experiments/CellD_Paderborn_ATSF_bearing/summary_bearing_vs_run.md` when that run exists.

---

## 6. Output layout

See [`output/README.md`](output/README.md) for the already-run artifacts.

```
output/
├── paper/          # manuscript-numbered figures and tables from main.py (use these)
├── figures/        # same stems as paper/ (written by make_figXX.py)
├── tables/         # five-cell summary CSVs
├── analysis_n30/   # supplementary n=30 merge from `python main.py n30`
├── experiments/    # per-cell results_table.csv, diagnostics/*.jsonl, …
├── intermediate/   # eval_dump (α / gates / features; large)
└── logs/
```

Preprocessing is in-memory (interpolation, train-split z-score, windowing). A processed sensor dataset is **not** written under `data/`.

---

## 7. FAQ

**Figures only, no training?**  
Install with `pip install -r requirements.txt`, then `python main.py all`.

**`main.py` says `rho_curve.csv` / `eval_dump` is missing?**  
That experiment or dump is not under `output/`. Use the shipped `output/`, or retrain (section 4) and run `main.py` again.

**TEP will not load?**  
Confirm the four `.RData` filenames match `data/README.md` and that `rdata==0.11.2` is installed from `requirements.txt`.

**Paderborn script: no extractor found?**  
Install 7-Zip / unrar so `7z` or `unrar` is on `PATH`, then `python scripts/download_paderborn.py`.

**Username path contains non-ASCII characters?**  
The code locates `data/` and `output/` from the repository root; you usually do not need to change anything. If the data must live elsewhere, set `NPPAD_ROOT` / `TEP_ROOT` / `PAD_ROOT`.

---

## 8. Licence and citation

**Licence:** this repository’s code is released under the [MIT License](LICENSE) (Copyright © 2026 Yi Xu). Datasets are **not** redistributed; NPPAD / TEP / Paderborn keep their own terms (Paderborn is **CC BY-NC 4.0**, non-commercial). MIT on the code does not change those dataset licences.

**Citation** (manuscript under review):

> Yi Xu. _Silent Failures of Adaptive Mechanisms in Data-Driven Nuclear Power Plant Transient Diagnosis: A Component-Level Reliability Audit Across Simulated and Measured Data._ (under review)

**Contact:** Yi Xu — [Yi.Xu.Prof@outlook.com](mailto:Yi.Xu.Prof@outlook.com) · ORCID [0009-0002-3789-8136](https://orcid.org/0009-0002-3789-8136)
