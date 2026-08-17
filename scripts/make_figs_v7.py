#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figs_v7.py — five-cell panels. Generator stems stay fig7–fig10;
manuscript numbers are Fig. 13 / 10 / 9 / 11 respectively.

Usage (from the repository root)::
    python scripts/make_figs_v7.py
    python scripts/make_figs_v7.py --csvdir output/tables --out output/figures/five_cell

Required CSVs inside --csvdir (both naming patterns are auto-detected):
    degradation_table_{tag}.csv        significance_{tag}.csv
    epoch_indicators_{tag}.csv         finals_{tag}.csv
    tag in: cellO_key, nppad_tsnet, tep_atsf, tep_tsnet, paderborn_atsf
    (epoch/finals may carry a "replication_" prefix — handled automatically)

Outputs: output/figures/five_cell/fig7_*.png/.svg ... fig10_*.png/.svg  (PNG 300 dpi + SVG)
Only needs: pandas, numpy, matplotlib.
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atsf_dcg.paths import FIGURES_ROOT, TABLES_ROOT  # noqa: E402

CELLS = [  # (short label, display name, csv tag)
    ("O", "NPPAD x ATSF-DCG",      "cellO_key"),
    ("A", "NPPAD x TSF-TimesNet",  "nppad_tsnet"),
    ("B", "TEP x ATSF-DCG",        "tep_atsf"),
    ("C", "TEP x TSF-TimesNet",    "tep_tsnet"),
    ("D", "Paderborn x ATSF-DCG",  "paderborn_atsf"),
]
DEGS = ["gaussian_noise_snr20", "gaussian_noise_snr10", "drift", "bias",
        "stuck", "dropout", "downsample"]
DEG_LBL = {"gaussian_noise_snr20": "noise SNR20", "gaussian_noise_snr10": "noise SNR10",
           "drift": "drift", "bias": "bias", "stuck": "stuck",
           "dropout": "dropout", "downsample": "downsample"}
# muted, low-saturation palette; D emphasised in near-black
CCOLOR = {"O": "#7f7f7f", "A": "#c08a3e", "B": "#5a8f6f", "C": "#a0555f", "D": "#2b2b2b"}

CSVdir = str(TABLES_ROOT)
OUTdir = str(FIGURES_ROOT / "five_cell")


def find_csv(prefix, tag):
    # recursive search: CSVs may sit in five_cell_summary / epoch_indicators / per_seed_finals
    candidates = [f"{prefix}_{tag}.csv", f"{prefix}_replication_{tag}.csv"]
    hits = []
    for dp, _, fns in os.walk(CSVdir):
        for f in fns:
            if f in candidates or (f.startswith(prefix) and tag in f and f.endswith(".csv")):
                hits.append(os.path.join(dp, f))
    if hits:
        exact = [h for h in hits if os.path.basename(h) in candidates]
        return sorted(exact or hits)[0]
    raise FileNotFoundError(f"{prefix}_*{tag}*.csv not found under {CSVdir}")


def pm(x):  # parse "0.7794±0.0090"
    return float(str(x).split("±")[0])


def save(fig, name):
    os.makedirs(OUTdir, exist_ok=True)
    for ext in ("png", "svg"):
        p = os.path.join(OUTdir, f"{name}.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print("  wrote", p)
    plt.close(fig)


# ---------------------------------------------------------------- manuscript Fig. 13 (generator fig7)
def fig7_heatmap():
    drop = np.zeros((5, 7)); gain = np.zeros((5, 7))
    for i, (_, _, tag) in enumerate(CELLS):
        d = pd.read_csv(find_csv("degradation_table", tag)).set_index("degradation")
        clean_full = pm(d.loc["clean", "full"])
        clean_ws   = pm(d.loc["clean", "w/o_spectral"])
        for j, dg in enumerate(DEGS):
            f_, w_ = pm(d.loc[dg, "full"]), pm(d.loc[dg, "w/o_spectral"])
            drop[i, j] = (clean_full - f_) * 100.0      # positive = accuracy lost
            gain[i, j] = (w_ - f_) * 100.0              # positive = removing spectral helps
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    ylabels = [f"Cell {s}: {n}" for s, n, _ in CELLS]
    xticks = [DEG_LBL[d] for d in DEGS]

    ax = axes[0]
    im = ax.imshow(drop, cmap="Oranges", vmin=0, vmax=60, aspect="auto")
    for i in range(5):
        for j in range(7):
            ax.text(j, i, f"{drop[i,j]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if drop[i, j] > 32 else "#333333")
    ax.set_title("(a) Full-model accuracy drop vs clean (pp)", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.85, label="drop (pp)")

    ax = axes[1]
    lim = 50.0
    im = ax.imshow(gain, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(5):
        for j in range(7):
            v = gain[i, j]
            v = 0.0 if abs(v) < 0.05 else v          # avoid "-0.0"
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 30 else "#333333")
    ax.set_title("(b) Gain from removing spectral branch (w/o_spectral - full, pp)", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.85, label="gain (pp)")

    for k, ax in enumerate(axes):
        ax.set_xticks(range(7)); ax.set_xticklabels(xticks, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(5))
        ax.set_yticklabels(ylabels if k == 0 else [""] * 5, fontsize=9)
        ax.set_xticks(np.arange(-.5, 7, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", length=0)
    fig.suptitle("", y=0.98)
    save(fig, "fig7_degradation_heatmap")


# ---------------------------------------------------------------- manuscript Fig. 10 (generator fig8)
def fig8_trajectories():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
    EMAX = 15
    for s, name, tag in CELLS:
        d = pd.read_csv(find_csv("epoch_indicators", tag))
        d = d[(d.config == "full") & (d.epoch <= EMAX)]
        g = d.groupby("epoch")[["rho", "s_tau"]].agg(["mean", "std"])
        x = g.index.values
        for ax, col, ylab in [(axes[0], "rho", r"modality ratio $\rho$"),
                              (axes[1], "s_tau", r"gate responsiveness $S$ (epoch est.)")]:
            m = g[(col, "mean")].values; sd = g[(col, "std")].fillna(0).values
            lw = 2.6 if s == "D" else 1.6
            ax.plot(x, m, color=CCOLOR[s], lw=lw, label=f"Cell {s} ({name})")
            ax.fill_between(x, m - sd, m + sd, color=CCOLOR[s], alpha=0.15, lw=0)
            ax.set_xlabel("epoch", fontsize=10); ax.set_ylabel(ylab, fontsize=10)
            ax.tick_params(labelsize=9)
    axes[0].axhline(1.5, ls="--", lw=1, color="#888888")
    axes[0].set_title("(a) temporal dominance ratio over training (dashed: $\\rho$=1.5, rule R2)",
                      fontsize=10.5)
    axes[1].axhline(0.15, ls="--", lw=1, color="#888888")
    axes[1].text(EMAX - .2, 0.156, "S = 0.15  (regime threshold, rule R1)",
                 ha="right", fontsize=8, color="#666666")
    axes[0].set_ylim(0, 6.8)      # clip early-epoch std overshoot
    axes[1].set_ylim(0.04, 0.32)
    axes[1].set_title("(b) gate responsiveness over training", fontsize=11)
    axes[1].legend(fontsize=8, loc="upper left", framealpha=0.9)
    save(fig, "fig8_early_indicator_trajectories")


# ---------------------------------------------------------------- manuscript Fig. 9 (generator fig9)
def fig9_forest():
    order = ["w/o_spectral", "w/o_temporal", "w/o_fusion", "w/o_dynamic_gating",
             "w/o_gating", "tsnet_vanilla"]
    pcol = {"w/o_spectral": "#a0555f", "w/o_temporal": "#2b2b2b",
            "w/o_fusion": "#5a8f6f", "w/o_gating": "#c08a3e",
            "w/o_dynamic_gating": "#8a7fa0", "tsnet_vanilla": "#7f7f7f"}
    rows = []
    for s, name, tag in CELLS:
        df = pd.read_csv(find_csv("significance", tag))
        for cfg in order:
            r = df[df.config == cfg]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            rows.append(dict(cell=s, cname=name, cfg=cfg,
                             d=r.acc_diff_mean * 100,
                             lo=r.acc_ci95_lo * 100, hi=r.acc_ci95_hi * 100))
        rows.append(None)  # group separator
    if rows and rows[-1] is None:
        rows.pop()
    n = len(rows)
    fig, ax = plt.subplots(figsize=(8.6, 0.34 * n + 1.6))
    ypos = list(range(n))[::-1]
    seen = set()
    for y, r in zip(ypos, rows):
        if r is None:
            continue
        lbl = r["cfg"] if r["cfg"] not in seen else None
        seen.add(r["cfg"])
        ax.plot([r["lo"], r["hi"]], [y, y], color=pcol[r["cfg"]], lw=2, solid_capstyle="round")
        ax.plot(r["d"], y, "o", ms=6, color=pcol[r["cfg"]], label=lbl)
        # significant markers: CI not covering 0
        if r["lo"] > 0 or r["hi"] < 0:
            ax.plot(r["d"], y, "o", ms=11, mfc="none", mec=pcol[r["cfg"]], mew=1.2)
    ax.axvline(0, color="#444444", lw=1)
    ylab = []
    for r in rows:
        if r is None:
            ylab.append("")
        else:
            tag = {"w/o_spectral": "w/o spectral", "w/o_temporal": "w/o temporal",
                   "w/o_fusion": "w/o fusion", "w/o_dynamic_gating": "w/o dyn. gating",
                   "w/o_gating": "w/o gating", "tsnet_vanilla": "backbone only"}[r["cfg"]]
            ylab.append(f"Cell {r['cell']}   {tag}")
    ax.set_yticks(ypos); ax.set_yticklabels(ylab, fontsize=8.5)
    ax.set_xlabel("accuracy difference vs full model (pp, mean ± 95% CI)", fontsize=10)
    ax.set_title("Five-cell ablation audit: open circles = 95% CI excludes zero "
                 "(F1 reverses only on Cell D)", fontsize=10.5)
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.tick_params(axis="x", labelsize=9)
    # legend below the axes: never occludes data points
    handles, labels = ax.get_legend_handles_labels()
    pretty = {"w/o_spectral": "w/o spectral", "w/o_temporal": "w/o temporal",
              "w/o_fusion": "w/o fusion", "w/o_dynamic_gating": "w/o dyn. gating",
              "w/o_gating": "w/o gating", "tsnet_vanilla": "backbone only"}
    labels = [pretty.get(l, l) for l in labels]
    ax.legend(handles, labels, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False)
    save(fig, "fig9_ablation_forest")


# ---------------------------------------------------------------- manuscript Fig. 11 (generator fig10)
def fig10_regime_map():
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for s, name, tag in CELLS:
        e = pd.read_csv(find_csv("epoch_indicators", tag))
        e = e[(e.config == "full") & (e.epoch == 3)]
        f = pd.read_csv(find_csv("finals", tag))
        f = f[f.config == "full"]
        xe, ye = e.rho.mean(), e.s_tau.mean()
        xf, yf = f["final.rho_last"].mean(), f["final.S_tau"].mean()
        ax.add_patch(FancyArrowPatch((xe, ye), (xf, yf), arrowstyle="-|>",
                                     mutation_scale=13, color=CCOLOR[s], lw=1.4, alpha=0.75))
        ax.plot(xe, ye, "o", ms=10, mfc="white", mec=CCOLOR[s], mew=1.8)
        ax.plot(xf, yf, "o", ms=10, color=CCOLOR[s])
        ax.annotate(f"Cell {s}", (xf, yf), textcoords="offset points", xytext=(10, 4),
                    fontsize=10, color=CCOLOR[s], fontweight="bold")
    ax.axvline(1.5, ls="--", lw=1, color="#999999")
    ax.axhline(0.15, ls="--", lw=1, color="#999999")
    ax.text(3.9, 0.02, "failure regime\n(F1 recurs: spectral detrimental,\ngate hyporesponsive)",
            fontsize=8.5, ha="center", color="#7a4a4f")
    ax.text(3.9, 0.27, "responsive regime\n(F1 reverses: spectral beneficial)",
            fontsize=8.5, ha="center", color="#2b2b2b")
    ax.text(0.35, 0.02, "balanced regime\n(weak dominance)", fontsize=8.5, ha="center", color="#5a8f6f")
    ax.set_xlabel(r"modality ratio $\rho$ (temporal / spectral)", fontsize=11)
    ax.set_ylabel(r"gate responsiveness $S(0.9)$", fontsize=11)
    ax.set_title("Regime map: open marker = epoch-3 estimate, filled = final;\n"
                 "the failure regime is identifiable within the first 3 epochs", fontsize=10.5)
    ax.grid(ls=":", alpha=0.4)
    ax.tick_params(labelsize=9)
    ax.set_xlim(0, 4.8); ax.set_ylim(0, 0.34)
    save(fig, "fig10_regime_map")


def main():
    global CSVdir, OUTdir
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvdir", default=str(TABLES_ROOT),
                    help="folder containing the result/diag CSVs (walked recursively)")
    ap.add_argument("--out", default=str(FIGURES_ROOT / "five_cell"),
                    help="output folder for figures")
    ap.add_argument("--only", default="", help="e.g. '7,8' to build a subset")
    args = ap.parse_args()
    CSVdir = os.path.abspath(args.csvdir)
    OUTdir = args.out
    if not os.path.isdir(CSVdir):
        sys.exit(f"csvdir not found: {CSVdir}")
    todo = set(args.only.split(",")) if args.only else {"7", "8", "9", "10"}
    print(f"reading CSVs from: {CSVdir}\nwriting figures to: {os.path.abspath(OUTdir)}")
    if "7" in todo:
        print("Fig.7 degradation heatmap");          fig7_heatmap()
    if "8" in todo:
        print("Fig.8 early-indicator trajectories"); fig8_trajectories()
    if "9" in todo:
        print("Fig.9 ablation forest plot");         fig9_forest()
    if "10" in todo:
        print("Fig.10 regime map");                  fig10_regime_map()
    print("done.")


if __name__ == "__main__":
    main()
