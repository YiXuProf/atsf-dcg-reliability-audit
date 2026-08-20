#!/usr/bin/env python3
"""Deprecated alias — use make_fig09/10/11/13.py (manuscript numbering).

Forwards to ``five_cell_figs.py`` so old docs/commands keep working.
"""
from __future__ import annotations

import sys
from pathlib import Path

print("[warn] scripts/make_figs_v7.py is deprecated; "
      "prefer scripts/make_fig09.py … make_fig13.py", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import five_cell_figs as f  # noqa: E402

if __name__ == "__main__":
    f.main(sys.argv[1:])
