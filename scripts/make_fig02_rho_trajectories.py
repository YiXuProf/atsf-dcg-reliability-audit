"""Regenerate manuscript Fig. 2 (rho training trajectories, n=10) from rho_curve.csv.

Usage (from the repository root)::
    python scripts/make_fig02_rho_trajectories.py
    python scripts/make_fig02_rho_trajectories.py --results-dir output/experiments/nppad_atsf_full \
        --out output/figures/fusion/fig02_rho_trajectories.png

Reads:
    <results-dir>/rho_curve.csv    (config, seed, epoch, rho)  -- written by
                                   run_experiments.py (training or --merge-only)
Writes:
    fig02_rho_trajectories.png (300 dpi) and .svg  (same stem as paper/)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atsf_dcg.paths import EXPERIMENTS_ROOT, FIGURES_ROOT  # noqa: E402

# ACADEMIC palette (consistent with the other figures)
C_FULL = "#7A8B99"      # gray        - full model
C_R1 = "#4A6FA5"        # blue        - R1 balanced training
C_R1C = "#6B8CBB"       # light blue  - R1-containing combinations
C_SINC = "#B03A2E"      # red         - R3-sinc (aggravated dominance)
C_ALL = "#2E4A62"       # dark blue   - full_all
C_THR = "#B03A2E"       # dashed red  - dominance threshold rho = 1.5

# config -> (colour, linestyle, label, show band)
PANEL_A = {
    "full": (C_FULL, "-", "Full", True),
    "full_r1": (C_R1, "-", "R1", False),
    "full_r1_r3": (C_R1C, "--", "R1+R3", False),
    "full_r1_r2_r3": (C_R1C, "-.", "R1+R2+R3", False),
    "full_all": (C_ALL, ":", "All remedies", False),
    "full_r3_sinc": (C_SINC, "-", "R3-sinc", True),
}
PANEL_B = {
    "full": (C_FULL, "-", "Full", False),
    "w/o_gating": ("#3D5A73", "-", "w/o Gating", False),
    "w/o_dynamic_gating": ("#5C7A99", "--", "w/o Dyn. Gating", False),
    "w/o_fusion": ("#8BA3C7", "-.", "w/o Fusion", False),
    "full_r2": ("#4A6FA5", "-", "R2", False),
    "full_r2_gumbel": ("#6B8CBB", "--", "R2-gumbel", False),
}

THRESHOLD = 1.5


def _mean_curve(df: pd.DataFrame):
    """Per-epoch seed-averaged (mean) trajectory and +/-1 std band."""
    g = df.groupby("epoch")["rho"]
    mean = g.mean()
    std = g.std().fillna(0.0)
    return mean.index.values, mean.values, std.values


def _draw_panel(ax, rho: pd.DataFrame, panel: dict, title: str) -> None:
    for cfg, (colour, ls, label, band) in panel.items():
        sub = rho[rho.config == cfg]
        if sub.empty:
            print(f"[warn] config {cfg!r} not found in rho_curve.csv -- skipped")
            continue
        ep, mean, std = _mean_curve(sub)
        ax.plot(ep, mean, color=colour, ls=ls, lw=1.6, label=label)
        if band:
            ax.fill_between(ep, mean - std, mean + std, color=colour, alpha=0.15, lw=0)
    ax.axhline(THRESHOLD, color=C_THR, ls="--", lw=1.0, alpha=0.8)
    ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 1, THRESHOLD * 1.05,
            r"dominance threshold $\rho = 1.5$", color=C_THR,
            fontsize=8, ha="right", va="bottom")
    ax.axhline(1.0, color="#999999", ls=":", lw=0.8, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"gradient-norm ratio $\rho$ (log scale)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir",
                    default=str(EXPERIMENTS_ROOT / "nppad_atsf_full"),
                    help="directory containing rho_curve.csv")
    ap.add_argument("--out",
                    default=str(FIGURES_ROOT / "fusion" / "fig02_rho_trajectories.png"))
    args = ap.parse_args()

    rho = pd.read_csv(Path(args.results_dir) / "rho_curve.csv")

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    _draw_panel(axes[0], rho, PANEL_A, "(a) Full vs. gradient-balancing remedies")
    _draw_panel(axes[1], rho, PANEL_B, "(b) Full vs. gating variants")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")               # PNG (300 dpi)
    svg = out.with_suffix(".svg")
    fig.savefig(svg, bbox_inches="tight")                        # SVG (vector)
    print(f"[done] wrote {out} and {svg}")

    # ---- stats quoted in the paper -------------------------------------
    # All stats are computed on the SEED-AVERAGED trajectory (per-epoch mean
    # across seeds), matching the Fig. 2 caption definition.
    print("\n[stats] seed-averaged trajectory summary (mean over seeds per epoch):")
    for cfg in sorted(rho.config.unique()):
        sub = rho[rho.config == cfg]
        curve = sub.groupby("epoch")["rho"].mean()
        frac = (curve > THRESHOLD).mean() * 100
        first5 = curve.iloc[:5].mean()
        last5 = curve.iloc[-5:].mean()
        print(f"  {cfg:24s} median={curve.median():6.2f}  "
              f"rho>1.5 in {frac:5.1f}% of epochs  "
              f"first5={first5:5.2f}  last5={last5:5.2f}")


if __name__ == "__main__":
    main()
