#!/usr/bin/env python
"""Command-line runner (for servers / long jobs). Same code path as notebooks/02_experiments.
Examples:
  python scripts/run_experiment.py table2 --datasets SVHN CIFAR10 --seeds 0 1
  python scripts/run_experiment.py table3
  python scripts/run_experiment.py ablation
  python scripts/run_experiment.py sensitivity --datasets SVHN
  python scripts/run_experiment.py scalability
  python scripts/run_experiment.py awe
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fuguard_repro.config import IMAGE_DATASETS, METHODS, PROTOCOLS, ot_variant, lr_variant  # noqa: E402
from fuguard_repro.aggregate import SENS_LAMBDAS  # noqa: E402
from fuguard_repro.runner import run_grid, standard_jobs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("experiment", choices=["table2", "table3", "ablation", "sensitivity", "scalability", "awe"])
ap.add_argument("--datasets", nargs="+", default=None)
ap.add_argument("--methods", nargs="+", default=METHODS)
ap.add_argument("--seeds", nargs="+", type=int, default=None)
ap.add_argument("--device", default=None)
a = ap.parse_args()

if a.experiment in ("table2", "table3"):
    p = PROTOCOLS[a.experiment]
    run_grid(a.experiment, a.datasets or IMAGE_DATASETS, standard_jobs(a.methods), a.seeds or p["seeds"],
             p["recovery_rounds"], device=a.device)
elif a.experiment == "ablation":
    run_grid("ablation", a.datasets or ["SVHN"], [(ot_variant(0.0), "fuguard", {"ot_lambda": 0.0})],
             a.seeds or PROTOCOLS["table2"]["seeds"], 5, device=a.device)
elif a.experiment == "sensitivity":
    jobs = [(ot_variant(l), "fuguard", {"ot_lambda": l}) for l in SENS_LAMBDAS if abs(l - 0.1) > 1e-12]
    run_grid("sensitivity", a.datasets or IMAGE_DATASETS, jobs, a.seeds or PROTOCOLS["table2"]["seeds"], 5, device=a.device)
elif a.experiment == "scalability":
    run_grid("scalability", ["SVHN"], standard_jobs(a.methods), a.seeds or PROTOCOLS["table2"]["seeds"], 5,
             base_overrides={"num_clients": 100, "tag": "SVHN-100"}, device=a.device)
elif a.experiment == "awe":
    jobs = [("retrain", "retrain", {})]
    for lr in (0.1, 0.05, 0.01):
        jobs += [(lr_variant("fedsga", lr), "fedsga", {"forgetting_lr": lr}),
                 (lr_variant("fuguard", lr), "fuguard", {"forgetting_lr": lr})]
    run_grid("awe", ["AWE"], jobs, a.seeds or [0], 50, kind="awe", device=a.device)
