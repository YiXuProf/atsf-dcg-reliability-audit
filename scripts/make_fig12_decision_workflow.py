# -*- coding: utf-8 -*-
"""
Fig. 12 — The D1–D5 screening-and-routing workflow (v2 redesign).

Top lane: training-time screening (epoch 2–3 indicators -> D1 regime -> D2
dominance band -> D3 vulnerability ranking). Bottom lane: deployment routing
(degradation-type detector -> D4/D5 -> head routing with zero retraining).

D2 is three-way (Table 6 / §4.9):
    yes (rho > 3)        -> dominance verdict -> D3
    uncertain (0.8, 3)   -> ten-seed ablation (§4.8) -> D3
    no  (rho < 0.8)      -> balanced, no dominance call -> D3

v2 changes vs. first draft (geometry fixes, same style/palette/canvas):
- Ablation box moved directly under D2; the uncertain stem is a single
  straight arrow into its top edge (no elbow next to the D2 stem label).
- The "no (< 0.8)" rail now runs BELOW the ablation box and enters D3's
  bottom at x = 124.5, while ablation -> D3 enters at x = 115: two clearly
  separated parallel verticals instead of a merged diagonal at D3's corner.
- D4/D5 degradation -> head mapping moved INSIDE the route-head box
  (self-contained; wording follows Table 6 rows D4/D5 and the §4.10 drift
  counter-example, including D5's responsive-regime condition).
- D3 -> route drop moved to x = 134 with the default/override note on its
  right, clear of the D2 rails and the lane divider.

Run from anywhere::

    python scripts/make_fig12_decision_workflow.py
Outputs:
    output/figures/architecture/fig12_decision_workflow.png   (300 dpi)
    output/figures/architecture/fig12_decision_workflow.svg
"""
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

C_TEMP   = "#dce6f1"
C_TEMP_E = "#7c9ab8"
C_SPEC   = "#efe0da"
C_SPEC_E = "#b98a7a"
C_OK     = "#e6efe4"
C_OK_E   = "#6f9068"
C_BOX    = "#f5f3ee"
C_BOX_E  = "#8a8578"
C_TEXT   = "#3a3a3a"
C_LINE   = "#555555"

fig, ax = plt.subplots(figsize=(10.2, 3.3))
ax.set_xlim(0, 162); ax.set_ylim(0, 52); ax.axis("off")


def block(x, y, w, h, lines, fc=C_BOX, ec=C_BOX_E, fs=12.5, sfs=11.5, tfs=9.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1.6))
    cy = y + h / 2
    n = len(lines)
    if n == 1:
        ax.text(x + w / 2, cy, lines[0], ha="center", va="center", fontsize=fs, color=C_TEXT)
    elif n == 2:
        ax.text(x + w / 2, cy + h * 0.17, lines[0], ha="center", va="center",
                fontsize=fs, color=C_TEXT)
        ax.text(x + w / 2, cy - h * 0.22, lines[1], ha="center", va="center",
                fontsize=sfs, color=C_TEXT, style="italic")
    else:
        ax.text(x + w / 2, cy + h * 0.28, lines[0], ha="center", va="center",
                fontsize=fs, color=C_TEXT)
        ax.text(x + w / 2, cy + h * 0.02, lines[1], ha="center", va="center",
                fontsize=sfs, color=C_TEXT, style="italic")
        ax.text(x + w / 2, cy - h * 0.28, lines[2], ha="center", va="center",
                fontsize=tfs, color=C_TEXT, style="italic")


def diamond(cx, cy, w, h, lines, fc="#fdf3e6", ec="#c9a25e"):
    ax.add_patch(Polygon(
        [(cx - w / 2, cy), (cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2)],
        closed=True, fc=fc, ec=ec, lw=1.6))
    ax.text(cx, cy + h * 0.16, lines[0], ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=C_TEXT)
    ax.text(cx, cy - h * 0.20, lines[1], ha="center", va="center", fontsize=10.5,
            color=C_TEXT, style="italic")


def straight(x1, y1, x2, y2, ms=18):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
                                 lw=1.8, color=C_LINE))


# ---- lane labels and divider ----
ax.text(1.5, 50.4525, "TRAINING (epoch 2–3)", ha="left", va="center", fontsize=11,
        fontweight="bold", color=C_TEXT)  # +1 px (990 px / 52 units = 19.04 px/unit)
ax.text(1.5, 22.6, "DEPLOYMENT", ha="left", va="center", fontsize=11,
        fontweight="bold", color=C_TEXT)
ax.plot([0, 162], [24.8, 24.8], ls=(0, (6, 4)), lw=1.2, color="#999999", zorder=0)

# ---- top lane ----
block(2.5, 38.5, 21, 10.5, ["Train 2–3 epochs", r"log $s_\tau$, $\rho$, $H(\alpha)$"],
      fs=11.5, sfs=10.5)                                   # R edge 24.1 (incl. pad)
diamond(36.5, 43.8, 17, 10.5, ["D1", r"$s_\tau>0.15$?"])   # L28 / R45 / B38.55
diamond(57.0, 43.8, 18, 10.5, ["D2", r"$\rho$ band?"])     # L48 / R66 / B38.55
block(74, 38.5, 23, 10.5, ["Dominance verdict", "expect F1/F3 silent"],
      fs=11.5, sfs=10.5)                                   # L73.4 / R97.6 (shifted right:
                                                           # room for the D2 yes-label)
block(101, 38.5, 38, 10.5, ["D3: vulnerability ranking", "default: spectral-free head"],
      fc=C_TEMP, ec=C_TEMP_E, fs=11.5, sfs=10.5)           # L100.4 / R139.6 / bottom 37.9

# Ablation box directly under D2 (D2 bottom vertex x = 57)
block(44, 28.5, 26, 7.0, ["Run ten-seed ablation", "(§4.8)"],
      fs=11, sfs=10)                                       # L43.4 / R70.6 / top 36.1

# ---- bottom lane ----
block(22, 1.8, 34, 13.5,
      ["Responsive regime", "retain adaptive modules",
       "flag elevated degradation vulnerability"],
      fc=C_OK, ec=C_OK_E, fs=11.5, sfs=10.5, tfs=8.2)      # R56.6 / top 15.9
block(60, 3.2, 24, 11, ["Degradation-type", "detector at runtime"],
      fs=11.5, sfs=10.5)                                   # R84.6
diamond(97, 8.7, 17, 10.5, ["D4/D5", "deg. type?"])        # L88.5 / R105.5

# Route-head box: mapping moved INSIDE (self-contained, Table 6 D4/D5 + §4.10)
RX, RY, RW, RH = 112, 1.0, 48, 15.5                        # L111.4 / top 17.1
ax.add_patch(FancyBboxPatch((RX, RY), RW, RH, boxstyle="round,pad=0.6,rounding_size=1.2",
                            fc=C_SPEC, ec=C_SPEC_E, lw=1.6))
ax.text(RX + RW / 2, 14.2, "Route head, zero retraining", ha="center", va="center",
        fontsize=11.5, color=C_TEXT)
ax.text(RX + RW / 2, 11.8, "spectral-free / spectral-only", ha="center", va="center",
        fontsize=10.5, color=C_TEXT, style="italic")
ax.plot([115, 157], [10.6, 10.6], lw=0.8, color=C_SPEC_E)  # inner rule
map_lines = [
    "noise / dropout → spectral-free head",
    "bias + responsive regime → spectral-only head",
    "bias + low-response regime → retain full",
    "drift → retain full + warning",
]
for i, line in enumerate(map_lines):
    ax.text(RX + RW / 2, 9.1 - i * 2.0, line, ha="center", va="center",
            fontsize=7.2, color=C_TEXT)

# ---- top arrows ----
straight(24.1, 43.8, 28.0, 43.8)
straight(45.0, 43.8, 47.6, 43.8)
ax.text(46.3, 45.5, "no", ha="center", va="center", fontsize=10, color=C_TEXT,
        style="italic")

# D2 → Dominance: yes (> 3); two stacked lines — a single line is ~11 units
# wide and cannot fit the 7.4-unit gap between the D2 vertex and the box edge
straight(66.0, 43.8, 73.4, 43.8)
ax.text(69.7, 44.9, "yes\n(> 3)", ha="center", va="bottom", fontsize=8,
        color=C_TEXT, style="italic")
straight(97.6, 43.8, 100.4, 43.8)

# D2 uncertain (0.8, 3) → ablation: single straight stem into the box top
straight(57.0, 38.55, 57.0, 36.1, ms=16)
ax.text(58.5, 37.2, r"uncertain $(0.8,\ 3)$", ha="left", va="center",
        fontsize=7.8, color=C_TEXT, style="italic")

# ablation → D3: right at y=32, then up into D3 bottom at x=115
ax.plot([70.6, 115.0], [32.0, 32.0], lw=1.8, color=C_LINE, solid_capstyle="round")
straight(115.0, 32.0, 115.0, 37.9, ms=16)

# D2 no (< 0.8) → D3: exit bottom-left edge point, elbow LEFT around the
# ablation box (a straight drop at x=52 would pierce its top edge), rail
# BELOW the box, up into D3 bottom at x=124.5 (separate from the ablation
# vertical at x=115)
ax.plot([52.0, 52.0, 40.5, 40.5, 124.5], [41.4, 38.2, 38.2, 26.5, 26.5],
        lw=1.8, color=C_LINE, solid_capstyle="round")
straight(124.5, 26.5, 124.5, 37.9, ms=16)
ax.text(50.8, 39.6, r"no ($< 0.8$)", ha="right", va="center", fontsize=8.2,
        color=C_TEXT, style="italic")
ax.text(88.0, 27.5, "balanced: no dominance call", ha="center", va="bottom",
        fontsize=7.2, color=C_TEXT, style="italic")

# D1 yes → responsive
straight(36.5, 38.55, 36.5, 15.9)
ax.text(37.7, 23.5, "yes", ha="left", va="center", fontsize=10, color=C_TEXT,
        style="italic")

# ---- bottom arrows ----
straight(56.6, 8.7, 59.4, 8.7)
straight(84.6, 8.7, 88.5, 8.7)
straight(105.5, 8.7, 111.4, 8.7)

# D3 → route head (drop at x=134, note to its right, clear of the rails)
straight(134.0, 37.9, 134.0, 16.9)
ax.text(135.5, 27.5, "D3: default; D4/D5: override\nif degradation detected",
        ha="left", va="center", fontsize=7.0, color=C_TEXT, style="italic")

plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
fig.savefig(OUT + ".png", dpi=300)
fig.savefig(OUT + ".svg")
print("saved:", OUT + ".png", "and", OUT + ".svg")
