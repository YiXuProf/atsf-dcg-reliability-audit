#!/usr/bin/env python3
"""Manuscript Fig. 13 — sensor-degradation heatmap.

Writes ``output/figures/five_cell/fig13_degradation_heatmap.{png,svg}``
(same stem as ``output/paper/figures/``).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import five_cell_figs as f  # noqa: E402

if __name__ == "__main__":
    f.render(["13"])
