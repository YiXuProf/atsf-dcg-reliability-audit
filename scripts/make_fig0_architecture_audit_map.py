# -*- coding: utf-8 -*-
"""
Fig. 1 — Architecture under audit and its failure-mode map (final geometry v3).
- Spectral encoder label split over two lines, box taller.
- Adaptive-fusion box widened.
- All four back-half arrows (encoder->fusion elbow tips, fusion->gate,
  gate->head, head->classes) are the SAME SHORT length (2.6 units:
  arrowhead plus a sliver of shaft).
- Every connector endpoint sits exactly on a box-edge midpoint.

Run from anywhere::

    python scripts/make_fig0_architecture_audit_map.py
Outputs:
    output/figures/architecture/fig0_architecture_audit_map.png   (300 dpi)
    output/figures/architecture/fig0_architecture_audit_map.svg
"""
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atsf_dcg.paths import FIGURES_ROOT  # noqa: E402

OUT_DIR = FIGURES_ROOT / "architecture"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = str(OUT_DIR / "fig0_architecture_audit_map")

C_TEMP   = "#dce6f1"   # temporal encoder fill (pale steel blue)
C_TEMP_E = "#7c9ab8"
C_SPEC   = "#efe0da"   # spectral encoder fill (pale terracotta)
C_SPEC_E = "#b98a7a"
C_BOX    = "#f5f3ee"
C_BOX_E  = "#8a8578"
C_F1     = "#b3543f"   # muted red
C_F2     = "#7d6b8f"   # muted purple
C_F3     = "#6f9068"   # muted green
C_TEXT   = "#3a3a3a"
C_LINE   = "#555555"

fig, ax = plt.subplots(figsize=(10.2, 4.3))
ax.set_xlim(0, 158); ax.set_ylim(0, 67); ax.axis("off")

def block(x, y, w, h, lines, fc=C_BOX, ec=C_BOX_E, fs=14, sfs=12.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1.6))
    cy = y + h / 2
    if len(lines) == 1:
        ax.text(x + w / 2, cy, lines[0], ha="center", va="center", fontsize=fs, color=C_TEXT)
    else:
        ax.text(x + w / 2, cy + h * 0.17, lines[0], ha="center", va="center", fontsize=fs, color=C_TEXT)
        ax.text(x + w / 2, cy - h * 0.22, lines[1], ha="center", va="center", fontsize=sfs,
                color=C_TEXT, style="italic")

def block3(x, y, w, h, title, sub1, sub2, fc, ec, fs=14, sfs=11.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1.6))
    cy = y + h / 2
    ax.text(x + w / 2, cy + 5.4, title, ha="center", va="center", fontsize=fs, color=C_TEXT)
    ax.text(x + w / 2, cy, sub1, ha="center", va="center", fontsize=sfs, color=C_TEXT, style="italic")
    ax.text(x + w / 2, cy - 5.4, sub2, ha="center", va="center", fontsize=sfs - 1, color=C_TEXT, style="italic")  # one size smaller

def elbow(x1, y1, xm, y2, x2):
    """horizontal -> vertical -> horizontal elbow connector with arrowhead."""
    ax.plot([x1, xm, xm], [y1, y1, y2], lw=1.8, color=C_LINE, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch((xm, y2), (x2, y2), arrowstyle="-|>", mutation_scale=18,
                                 lw=1.8, color=C_LINE))

def straight(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
                                 lw=1.8, color=C_LINE))

def badge(x, y, label, color, ty):
    ax.plot([x, x], [y - 3.6, ty], ls=(0, (4, 3)), lw=1.8, color=color, zorder=1)
    ax.add_patch(Circle((x, y), 3.4, fc=color, ec="white", lw=1.6, zorder=5))
    ax.text(x, y, label, ha="center", va="center", fontsize=13, fontweight="bold",
            color="white", zorder=6)

# ---- blocks (visible edges = x-0.6 / x+w+0.6; connector heights = box centers) ----
block(3, 24, 16, 12, ["Input", "window"])                                     # cy=30
block3(30, 38, 26, 17, "Spectral encoder", r"$E_f$", "(magnitude-FFT view)",
       C_SPEC, C_SPEC_E)                                                      # cy=46.5
block(30, 10, 26, 13, ["Temporal encoder", r"$E_t$"], fc=C_TEMP, ec=C_TEMP_E)  # cy=16.5
block(62, 24, 26, 13, ["Adaptive fusion", r"(weight $\alpha$)"])               # cy=30.5, edges 61.4/88.6
block(91.8, 24, 22, 13, ["Dynamic", "channel gate $G$"])                      # edges 91.2/114.4
block(117.6, 24, 23, 13, ["Classifier head", "(BiLSTM + attention)"], sfs=11)  # edges 117.0/141.2
block(144.4, 26.5, 12, 8, ["18 classes"], fs=11)                              # edges 143.8/157.0

# ---- front elbows (input fan-out) ----
elbow(19.6, 30, 25, 46.5, 29.4)    # input right-mid -> spectral left-mid
elbow(19.6, 30, 25, 16.5, 29.4)    # input right-mid -> temporal left-mid

# ---- back half: ALL four arrows are 2.6 units (arrowhead + sliver) ----
elbow(56.6, 46.5, 58.8, 30.5, 61.4)  # spectral right-mid -> fusion left-mid (tip 58.8->61.4)
elbow(56.6, 16.5, 58.8, 30.5, 61.4)  # temporal right-mid -> fusion left-mid (tip 58.8->61.4)
straight(88.6, 30.5, 91.2, 30.5)     # fusion right-mid -> gate left-mid
straight(114.4, 30.5, 117.0, 30.5)   # gate right-mid   -> head left-mid
straight(141.2, 30.5, 143.8, 30.5)   # head right-mid   -> classes left-mid

# ---- failure-mode badges: dashed stems land on box top midpoints ----
badge(43,    62.5, "F1", C_F1, 55.7)   # F1 -> spectral encoder top-mid
badge(79,    62.5, "F2", C_F2, 37.7)   # F2 -> fusion top-mid
badge(102.8, 62.5, "F3", C_F3, 37.7)   # F3 -> gate top-mid

plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
fig.savefig(OUT + ".png", dpi=300)
fig.savefig(OUT + ".svg")
print("saved:", OUT + ".png", "and", OUT + ".svg")
