#!/usr/bin/env python
"""Run one experiment plan.  python scripts/run_experiment.py main --seeds 0 1 [--gpus 2] [--quick] [--preset paper]

experiments: main | ablation_dim | ablation_alpha | ablation_dp | ablation_centroid | cross | smoke
Finished runs are skipped on a restart. With --gpus N (default: all visible GPUs) one worker process per GPU is started."""
import argparse
import ast

import _bootstrap  # noqa: F401
from fedup_repro import experiments as E
from fedup_repro import multigpu
from fedup_repro.paths import ensure_dirs

ap = argparse.ArgumentParser()
ap.add_argument("experiment", choices=E.EXPERIMENTS)
ap.add_argument("--benchmarks", nargs="+", default=None)
ap.add_argument("--methods", nargs="+", default=None)
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
ap.add_argument("--preset", default=None, choices=["repo", "paper", "fuguard"])
ap.add_argument("--quick", action="store_true", help="tiny settings for a fast pipeline check")
ap.add_argument("--force", action="store_true", help="recompute finished runs")
ap.add_argument("--device", default=None)
ap.add_argument("--gpus", type=int, default=None, help="worker processes (default: all visible GPUs)")
ap.add_argument("--skip-errors", action="store_true")
ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                help="override any config key for every job, e.g. --set rounds_per_epoch=1 --set filter_epochs=100 (repeatable)")
ap.add_argument("--worker", action="store_true", help="internal: run only the given --units in this process")
ap.add_argument("--units", default=None, help="internal: comma separated benchmark:seed pairs")
a = ap.parse_args()

ensure_dirs()
jobs = E.plan(a.experiment, a.benchmarks, tuple(a.seeds), a.preset, a.quick, a.methods)
extra = {}
for kv in a.set:
    k, _, v = kv.partition("=")
    try:
        extra[k.strip()] = ast.literal_eval(v)
    except (ValueError, SyntaxError):
        extra[k.strip()] = v                       # plain string such as sgd_momentum
for j in jobs:                                      # the sweep value of an ablation job wins over --set
    j["overrides"] = {**extra, **j["overrides"]}
if a.units:
    keep = {(u.split(":")[0], int(u.split(":")[1])) for u in a.units.split(",")}
    jobs = [j for j in jobs if (j["benchmark"], j["seed"]) in keep]
print(f"[plan] {a.experiment}: {len(jobs)} jobs")

n = a.gpus if a.gpus is not None else multigpu.count_gpus()
if not a.worker and n > 1 and len(jobs) > 1:
    groups = multigpu.split_units(jobs, n)
    if len(groups) > 1:
        passthrough = ["--seeds", *map(str, a.seeds)]
        for flag, val in (("--benchmarks", a.benchmarks), ("--methods", a.methods)):
            if val:
                passthrough += [flag, *val]
        if a.preset:
            passthrough += ["--preset", a.preset]
        for kv in a.set:
            passthrough += ["--set", kv]
        passthrough += (["--quick"] if a.quick else []) + (["--force"] if a.force else []) + (["--skip-errors"] if a.skip_errors else [])
        multigpu.spawn_workers(a.experiment, groups, passthrough)
        raise SystemExit(0)
done, failed = E.run_jobs(jobs, device=a.device, force=a.force, skip_errors=a.skip_errors)
print(f"[done] {len(done)} jobs, {len(failed)} failed")
