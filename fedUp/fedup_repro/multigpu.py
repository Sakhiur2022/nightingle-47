"""Use every GPU: one worker process per GPU (a single PyTorch process only ever uses one). Jobs are grouped by
(benchmark, seed) so the cached FedAvg checkpoint of a unit is written by exactly one worker."""
import os
import subprocess
import sys
import time
from pathlib import Path

from .paths import LOGS_DIR, ROOT

UNIT_COST = {"CIFAR100-ResNet34": 4.0, "CIFAR10-Transformer": 3.0, "AGNews-TinyBERT": 3.0, "CIFAR10-ResNet18": 2.5,
             "CIFAR10-ConvNet": 1.0, "CIFAR100-ConvNet": 1.0, "SVHN-ConvNet": 1.2, "EuroSAT-ConvNet": 0.6, "MNIST-LeNet5": 0.5, "SYNTHETIC-Tiny": 0.1}


def count_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
        return sum(1 for line in out.splitlines() if line.startswith("GPU "))
    except Exception:  # noqa: BLE001
        return 0


def split_units(jobs, n):
    """Greedy longest-processing-time split of the (benchmark, seed) units of `jobs` into at most n groups."""
    weight = {}
    for j in jobs:
        weight[(j["benchmark"], j["seed"])] = weight.get((j["benchmark"], j["seed"]), 0) + UNIT_COST.get(j["benchmark"], 2.0)
    units = sorted(weight, key=lambda u: -weight[u])
    bins, load = [[] for _ in range(min(n, len(units)) or 1)], [0.0] * (min(n, len(units)) or 1)
    for u in units:
        i = load.index(min(load))
        bins[i].append(u)
        load[i] += weight[u]
    return [b for b in bins if b]


def spawn_workers(experiment, groups, passthrough, poll=60):
    """Start `python scripts/run_experiment.py <experiment> --worker --units b:s,...` per group with CUDA_VISIBLE_DEVICES=k."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    procs, t0 = [], time.time()
    for k, g in enumerate(groups):
        units = ",".join(f"{b}:{s}" for b, s in g)
        log = LOGS_DIR / f"{experiment}_gpu{k}.log"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(k), "PYTHONUNBUFFERED": "1"}
        cmd = [sys.executable, str(Path(ROOT) / "scripts" / "run_experiment.py"), experiment, "--worker", "--units", units, *passthrough]
        procs.append((subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT), log))
        print(f"[multigpu] GPU {k}: {units}  -> {log.name}")
    try:
        while any(p.poll() is None for p, _ in procs):
            time.sleep(poll)
            for k, (p, log) in enumerate(procs):
                last = subprocess.run(["tail", "-n", "1", str(log)], capture_output=True, text=True).stdout.strip()
                print(f"[multigpu | gpu{k} | {int(time.time() - t0)}s | {'running' if p.poll() is None else 'exit %s' % p.returncode}] {last[:140]}")
    except KeyboardInterrupt:
        for p, _ in procs:
            p.terminate()
        raise
    bad = [(k, log) for k, (p, log) in enumerate(procs) if p.returncode != 0]
    if bad:
        tails = "\n".join(f"--- gpu{k} ({log}) ---\n" + subprocess.run(["tail", "-n", "25", str(log)], capture_output=True, text=True).stdout
                          for k, log in bad)
        raise RuntimeError("worker(s) failed:\n" + tails)
    print(f"[multigpu] done in {int(time.time() - t0)}s")
