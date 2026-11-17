#!/usr/bin/env python
"""Rebuild every table (results/tables) and figure (results/figures) from results/raw."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fuguard_repro.reports import build_tables, build_figures  # noqa: E402

for name, df in build_tables().items():
    print(f"\n=== {name} ===\n{df}")
print("\nfigures:", *[str(p) for p in build_figures()], sep="\n  ")
