#!/usr/bin/env python
"""Build (and cache) the federated splits for the benchmarks and seeds: Dirichlet partition + proxy hold-out.
Needed once per (benchmark, scenario type, seed); experiments call it automatically when the cache is missing."""
import argparse

import _bootstrap  # noqa: F401
from fedup_repro.config import PAPER_BENCHMARKS, make_config
from fedup_repro.data import build_fed_data
from fedup_repro.paths import ensure_dirs

ap = argparse.ArgumentParser()
ap.add_argument("--benchmarks", nargs="+", default=PAPER_BENCHMARKS)
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
ap.add_argument("--preset", default="repo")
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()
ensure_dirs()
for b in a.benchmarks:
    for sc in ("client", "class"):                    # 'client' merges train+test; 'class'/'sample' share one split
        cfg = make_config(b, sc, preset=a.preset, quick=a.quick)
        for s in a.seeds:
            fed = build_fed_data(cfg, s)
            print(f"{b:20s} {sc:7s} seed {s}: client sizes {fed['meta']['sizes']}")
