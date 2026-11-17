#!/usr/bin/env python
"""Build every table (results/tables) and figure (results/figures) from results/raw."""
import argparse

import _bootstrap  # noqa: F401
from fedup_repro.paths import ensure_dirs
from fedup_repro.reports import make_all

ap = argparse.ArgumentParser()
ap.add_argument("--main", default="main", help="experiment folder holding the main-results runs (e.g. smoke)")
ap.add_argument("--cross", default="cross")
a = ap.parse_args()
ensure_dirs()
out = make_all(main=a.main, cross=a.cross)
for k, v in out.items():
    print(f"{k}: {0 if v is None else len(v)} rows")
