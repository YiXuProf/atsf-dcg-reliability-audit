"""Regenerate manuscript Fig. 8 (2x2 diagnostics panel, n=10) from the merged result CSVs.

Usage (from the repository root)::
    python scripts/make_fig08_diagnostics.py
    python scripts/make_fig08_diagnostics.py --results-dir output/experiments/nppad_atsf_full \
        --out output/figures/diagnostics_panels/fig08_diagnostics.png

Reads:
    <results-dir>/results_table.csv   (config, accuracy "mean+-std", macro_f1)
    <results-dir>/remedy_table.csv    (config, accuracy_mean, H_alpha, alpha_tvar,
                                       S_tau, rho_last, perm_null_z)
Writes:
    fig08_diagnostics.png (300 dpi) and .svg  (same stem as paper/)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atsf_dcg.paths import EXPERIMENTS_ROOT, FIGURES_ROOT  # noqa: E402

# ACADEMIC palette (consistent with the other figures)
C_ABLATION = "#6B8CBB"   # light blue  - ablations
C_REMEDY = "#2E4A62"     # dark blue   - remedies
C_FULL = "#7A8B99"       # gray        - full model bar
C_BASE = "#B03A2E"       # dashed red  - full-model baseline line

ABLATIONS = ["w/o_spectral", "w/o_temporal", "w/o_fusion",
             "w/o_dynamic_gating", "w/o_gating"]
REMEDIES = ["full_r1", "full_r3_stft", "full_r3_sinc", "full_r1_r3",
            "full_r2", "full_r2_gumbel", "full_r4_sparsemax",
            "full_r4_entmax", "full_r4_lstm", "full_r1_r2_r3", "full_all"]


def _acc_mean(s: str) -> float:
    return float(str(s).split("±")[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir",
                    default=str(EXPERIMENTS_ROOT / "nppad_atsf_full"),
                    help="directory containing results_table.csv / remedy_table.csv")
    ap.add_argument("--out",
                    default=str(FIGURES_ROOT / "diagnostics_panels" / "fig08_diagnostics.png"))
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    res = pd.read_csv(rdir / "results_table.csv")
    rem = pd.read_csv(rdir / "remedy_table.csv").set_index("config")

    full_acc = _acc_mean(res.loc[res.config == "full", "accuracy"].iloc[0])
    d_acc = {c: (_acc_mean(res.loc[res.config == c, "accuracy"].iloc[0]) - full_acc) * 100
             for c in ABLATIONS + REMEDIES}
    rho = rem["rho_last"].to_dict()
    ent = rem["H_alpha"].to_dict()
    tvar = rem["alpha_tvar"].to_dict()

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))

    # (a) accuracy change vs Full -----------------------------------------
    ax = axes[0, 0]
    order = ABLATIONS + REMEDIES
    labels = [o.replace("w/o_", "w/o ").replace("full_", "") for o in order]
    colors = [C_ABLATION if o in ABLATIONS else C_REMEDY for o in order]
    ax.bar(range(len(order)), [d_acc[o] for o in order], color=colors, width=0.72)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7.5)
    ax.set_ylabel("Δ accuracy vs Full (pp)")
    ax.set_title("(a) Accuracy change under ablation / remedy", fontsize=10)

    # (b)-(d) metric panels: full + remedies -------------------------------
    def panel(ax, metric, ylabel, title):
        keys = ["full"] + REMEDIES
        labs = ["Full"] + [k.replace("full_", "") for k in REMEDIES]
        vals = [metric[k] for k in keys]
        cols = [C_FULL] + [C_REMEDY] * len(REMEDIES)
        ax.bar(range(len(keys)), vals, color=cols, width=0.72)
        ax.axhline(metric["full"], color=C_BASE, lw=1.0, ls="--",
                   alpha=0.85, label="Full baseline")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(labs, rotation=55, ha="right", fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, frameon=False)

    panel(axes[0, 1], rho, "ρ (gradient-norm ratio)", "(b) Gradient balance ρ")
    panel(axes[1, 0], ent, "H(α) (normalized)", "(c) Routing entropy H(α)")
    panel(axes[1, 1], tvar, "Var_t(α)", "(d) Routing temporal variance")

    for a in axes.flat:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.grid(axis="y", alpha=0.25, lw=0.5)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")                # PNG (300 dpi)
    svg = out.with_suffix(".svg")
    fig.savefig(svg, bbox_inches="tight")                          # SVG (vector)
    print(f"[done] wrote {out} and {svg} "
          f"(full acc baseline = {full_acc:.4f})")


if __name__ == "__main__":
    main()
