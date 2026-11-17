#!/usr/bin/env python
"""Everything, in order: smoke test -> data -> main results -> ablations -> cross-protocol -> reports.
    python run_all.py --quick          # tiny settings, checks the whole chain
    python run_all.py                  # the full grid (hours; use --gpus 2 on Kaggle T4 x2)"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--gpus", type=int, default=None)
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
ap.add_argument("--skip", nargs="*", default=[], help="experiments to skip")
a = ap.parse_args()


def sh(*cmd):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


py = sys.executable
sh(py, "-m", "fedup_repro.smoke")
common = ["--seeds", *map(str, a.seeds)] + (["--quick"] if a.quick else []) + (["--gpus", str(a.gpus)] if a.gpus else [])
if not a.quick:
    sh(py, "scripts/download_datasets.py")
for exp in ("main", "ablation_dim", "ablation_alpha", "ablation_dp", "ablation_centroid", "cross"):
    if exp not in a.skip:
        sh(py, "scripts/run_experiment.py", exp, *common)
sh(py, "scripts/make_reports.py")
