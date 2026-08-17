# output (paper artifacts)

This folder holds the **already-run** paper artifacts (figures, tables, logs, per-epoch diagnostics, eval dumps). They were copied from the original experiment workspace and were not retrained.

```
output/
├── paper/                      manuscript-numbered copies from main.py (Fig. 1–13 / Table I–VI / S1–S15)
├── figures/
│   ├── fusion/                 rho, fusion weights, gating, t-SNE, confusion matrix
│   ├── diagnostics_panels/     four-panel diagnostic plots
│   ├── five_cell/              degradation heatmap, early-stop trajectories, ablation forest, regime map
│   └── architecture/           Fig. 1 audit map / Fig. 12 D1–D5 workflow (drawn by scripts)
├── tables/
│   ├── five_cell_summary/      accuracy / remedy / significance / degradation
│   ├── epoch_indicators/       training-time H(α), S(τ), ...
│   ├── per_seed_finals/        per-seed finals + per_seed_raw.csv
│   └── reviewer_controls/      paired tests / TOST (Table V)
├── experiments/
│   ├── nppad_atsf_full/        NPPAD × ATSF full grid (20 configs × 10 seeds)
│   ├── CellO_NPPAD_ATSF/
│   ├── CellA_NPPAD_TimesNet/
│   ├── CellB_TEP_ATSF/
│   ├── CellC_TEP_TimesNet/
│   └── CellD_Paderborn_ATSF/
├── intermediate/
│   └── eval_dump/              α / gate / feature dumps for Figs. 3–7
└── logs/                       cloud batch-run logs
```

## figures

| folder | file | content |
|---|---|---|
| `fusion/` | `fig1_rho_trajectories` | ρ training trajectories (manuscript Fig. 2) |
| | `fig2_fusion_weight_dist` | fusion-weight distribution (Fig. 3) |
| | `fig3_fusion_weight_by_class` | per-class fusion weights (Fig. 4) |
| | `fig4_gate_heatmap` | gate heatmap (Fig. 5) |
| | `fig5_tsne` | feature t-SNE (Fig. 6) |
| | `fig6_confusion_matrix` | confusion matrix (Fig. 7) |
| `diagnostics_panels/` | `fig6_diagnostics` / `fig7_diagnostics` | diagnostic four-panel (Fig. 8) |
| `five_cell/` | `fig7_degradation_heatmap` | sensor degradation (Fig. 13) |
| | `fig8_early_indicator_trajectories` | early-stop indicators (Fig. 10) |
| | `fig9_ablation_forest` | five-cell ablation (Fig. 9) |
| | `fig10_regime_map` | failure regime map (Fig. 11) |

Architecture / D1–D5 workflow figures:

```bash
python scripts/make_fig0_architecture_audit_map.py
python scripts/make_fig12_decision_workflow.py
```

They write to `figures/architecture/`.

## tables

- `five_cell_summary/`: `results_table_*.csv`, `remedy_table_*.csv`, `significance_*.csv`, `degradation_table_*.csv` (tags: `cellO_key` / `nppad_tsnet` / `tep_atsf` / `tep_tsnet` / `paderborn_atsf`)
- `epoch_indicators/`: `epoch_indicators_*.csv`
- `per_seed_finals/`: `finals_*.csv`, `per_seed_raw.csv`
- `reviewer_controls/`: `custom_paired.csv`, `custom_paired_eps1pp.csv`, `custom_tost.csv`

To regenerate v7 figures:

```bash
python scripts/make_figs_v7.py
```

defaults: `--csvdir output/tables`, `--out output/figures/five_cell`.

## experiments

Each cell directory typically contains `results_table.csv`, `remedy_table.csv`, `significance.csv`, `degradation_table.csv`, `rho_curve.csv`, `per_class_table.csv`, `perm_null_summary.csv`, `protocol_report.json`, and `diagnostics/{config}_seed{seed}.jsonl`.

`nppad_atsf_full` is the complete NPPAD × ATSF grid (20 configs × 10 seeds, including R1–R4 and fixed-fusion controls).

## intermediate

`intermediate/eval_dump/full_seed42/` (test split) and `full_seed42_train/` (train split) contain `alpha.npy`, `gates.npy`, `features.npz`, `predictions.csv`, `meta.json` for redrawing fusion/gating figures. Large (~162 MB).

## logs

`cellA.log` / `cellB.log` / `cellC.log` / `run_D.log` / `run_all.log`

Datasets are not in this folder. Put them under `data/`; see [`../data/README.md`](../data/README.md).
