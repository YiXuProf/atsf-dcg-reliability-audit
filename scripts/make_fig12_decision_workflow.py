# -*- coding: utf-8 -*-
"""
Fig. 12 — The D1–D5 screening-and-routing workflow.
Top lane: training-time screening (epoch 2–3 indicators -> D1 regime -> D2 dominance
-> D3 vulnerability ranking). Bottom lane: deployment routing (degradation-type
detector -> D4/D5 -> head routing with zero retraining).

Run from anywhere::

    python scripts/make_fig12_decision_workflow.py
Outputs:
    output/figures/architecture/fig12_decision_workflow.png   (300 dpi)
    output/figures/architecture/fig12_decision_workflow.svg
"""
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atsf_dcg.paths import FIGURES_ROOT  # noqa: E402

OUT_DIR = FIGURES_ROOT / "architecture"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = str(OUT_DIR / "fig12_decision_workflow")

C_TEMP   = "#dce6f1"   # D3 / spectral-free default (pale steel blue)
C_TEMP_E = "#7c9ab8"
C_SPEC   = "#efe0da"   # routing block (pale terracotta)
C_SPEC_E = "#b98a7a"
C_OK     = "#e6efe4"   # responsive-regime block (pale green)
C_OK_E   = "#6f9068"
C_BOX    = "#f5f3ee"
C_BOX_E  = "#8a8578"
C_TEXT   = "#3a3a3a"
C_LINE   = "#555555"

fig, ax = plt.subplots(figsize=(10.2, 3.3))
ax.set_xlim(0, 162); ax.set_ylim(0, 52); ax.axis("off")

def block(x, y, w, h, lines, fc=C_BOX, ec=C_BOX_E, fs=12.5, sfs=11.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1.6))
    cy = y + h / 2
    if len(lines) == 1:
        ax.text(x + w / 2, cy, lines[0], ha="center", va="center", fontsize=fs, color=C_TEXT)
    else:
        ax.text(x + w / 2, cy + h * 0.17, lines[0], ha="center", va="center", fontsize=fs, color=C_TEXT)
        ax.text(x + w / 2, cy - h * 0.22, lines[1], ha="center", va="center", fontsize=sfs,
                color=C_TEXT, style="italic")

def diamond(cx, cy, w, h, lines, fc="#fdf3e6", ec="#c9a25e"):
    ax.add_patch(Polygon([(cx - w / 2, cy), (cx, cy + h / 2), (cx, cy - h / 2), (cx + w / 2, cy)],
                         closed=True, fc=fc, ec=ec, lw=1.6))
    ax.text(cx, cy + h * 0.16, lines[0], ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=C_TEXT)
    ax.text(cx, cy - h * 0.20, lines[1], ha="center", va="center", fontsize=10.5,
            color=C_TEXT, style="italic")

def straight(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
                                 lw=1.8, color=C_LINE))

# ---- lane labels and divider ----
ax.text(1.5, 49.5, "TRAINING (epoch 2–3)", ha="left", va="center", fontsize=11,
        fontweight="bold", color=C_TEXT)
ax.text(1.5, 25.5, "DEPLOYMENT", ha="left", va="center", fontsize=11,
        fontweight="bold", color=C_TEXT)
ax.plot([0, 162], [28.5, 28.5], ls=(0, (6, 4)), lw=1.2, color="#999999", zorder=0)

# ---- top lane: training-time screening (all text measured to fit its shape) ----
block(3, 34, 24, 12, ["Train 2–3 epochs", r"log $s_\tau$, $\rho$, $H(\alpha)$"], fs=12, sfs=11)  # edges 2.4/27.6
diamond(40, 40, 21, 13, ["D1", r"$s_\tau>0.15$?"])                  # vertices 29.5/50.5
diamond(63.5, 40, 21, 13, ["D2", r"$\rho>1.5$?"])                   # vertices 53/74
block(77, 34, 27, 12, ["Dominance verdict", "expect F1/F3 silent"])  # edges 76.4/104.6
block(107.5, 34, 36, 12, ["D3: vulnerability ranking", "default: spectral-free head"],
      fc=C_TEMP, ec=C_TEMP_E)                                        # edges 106.9/144.1

# ---- bottom lane: deployment routing ----
block(28.5, 2, 32, 12, ["Responsive regime", "retain adaptive modules"], fc=C_OK, ec=C_OK_E)  # 27.9/61.1
block(64, 2, 26, 12, ["Degradation-type", "detector at runtime"])      # edges 63.4/90.6
diamond(103, 8, 21, 13, ["D4/D5", "deg. type?"])                       # vertices 92.5/113.5
block(116, 2, 38.5, 12, ["Route head, zero retraining", "spectral-free / spectral-only"],
      fc=C_SPEC, ec=C_SPEC_E)                                          # edges 115.4/155.1

# ---- top-lane arrows (all endpoints on box/diamond edge midpoints) ----
straight(27.6, 40, 29.5, 40)                       # train -> D1
straight(50.5, 40, 53, 40)                         # D1 -> D2 ("no": keep screening)
ax.text(51.75, 42.2, "no", ha="center", va="center", fontsize=10.5, color=C_TEXT, style="italic")
straight(74, 40, 76.4, 40)                         # D2 -> dominance verdict
straight(104.6, 40, 106.9, 40)                     # dominance -> D3

# D1 "yes": drop to the responsive-regime block (vertical, lands on top edge)
straight(40, 33.5, 40, 14.6)
ax.text(41.5, 24, "yes", ha="left", va="center", fontsize=10.5, color=C_TEXT, style="italic")

# ---- bottom-lane arrows ----
straight(61.1, 8, 63.4, 8)                         # responsive regime -> detector
straight(90.6, 8, 92.5, 8)                         # detector -> D4/D5
straight(113.5, 8, 115.4, 8)                       # D4/D5 -> route head

# D3 default drops to the routing block (vertical, lands on top edge)
straight(125.5, 33.4, 125.5, 14.6)
ax.text(127, 24, "default", ha="left", va="center", fontsize=10.5, color=C_TEXT, style="italic")

plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
fig.savefig(OUT + ".png", dpi=300)
fig.savefig(OUT + ".svg")
print("saved:", OUT + ".png", "and", OUT + ".svg")