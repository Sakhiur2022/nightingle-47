#!/usr/bin/env python
"""One command to reproduce the FuGuard results locally:  install -> smoke test -> data -> experiments -> tables/figures.

    python run_all.py --dry-run                 # print every command, run nothing
    python run_all.py --quick                   # pilot: CIFAR-10, seed 0, FedSGA + FuGuard only
    python run_all.py                           # everything except AWE (long!)
    python run_all.py --with-awe clone          # also AWE (git clone, ~2 GB); or --with-awe C:\\path\\to\\zip_or_folder
    python run_all.py --stages data table2 reports --datasets CIFAR10 SVHN
    python run_all.py --skip-install --device cuda:0

Every stage is resumable (finished runs are skipped), so after an interruption just run the same command again.
Stages: install smoke data table2 table3 ablation sensitivity scalability awe reports
Logs: logs/<stage>.log.  Nothing here has been run on a GPU machine by the author - start with --quick.
"""
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
ALL_STAGES = ["install", "smoke", "data", "table2", "table3", "ablation", "sensitivity", "scalability", "awe", "reports"]
IMAGE_DATASETS = ["SVHN", "EuroSAT", "CIFAR10", "CIFAR100"]


def torchvision_version_for(torch_version):
    """torch 2.N -> torchvision 0.(N+15), the official pairing for the 2.x series (2.5->0.20, 2.8->0.23, 2.9->0.24)."""
    major, minor = (int(x) for x in torch_version.split("+")[0].split(".")[:2])
    if major != 2:
        raise ValueError(f"unsupported torch {torch_version}; install torchvision manually")
    return f"0.{minor + 15}.0"


def missing_torchvision_cmd():
    """If torch is installed but torchvision is not: pip command for the matching version WITHOUT touching torch."""
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torchvision") is not None:
        return None
    from importlib.metadata import version
    tv = torchvision_version_for(version("torch"))
    return [PY, "-m", "pip", "install", f"torchvision=={tv}", "--no-deps"]


def requirements_file(yes):
    """requirements.txt without torch/torchvision when torch is already installed (keeps your CUDA build)."""
    req = ROOT / "requirements.txt"
    if importlib.util.find_spec("torch") is None:
        print("torch is NOT installed. Plain `pip install torch` gives a CPU-only build on Windows. For a CUDA build, install torch "
              "from https://pytorch.org first and re-run with --skip-install.")
        if not yes and input("Install the default PyPI torch anyway? [y/N] ").strip().lower() != "y":
            sys.exit("aborted - install torch first")
        return req
    keep = [l for l in req.read_text().splitlines() if not l.strip().lower().startswith(("torch", "torchvision"))]
    tmp = Path(tempfile.gettempdir()) / "fuguard_requirements_no_torch.txt"
    tmp.write_text("\n".join(keep) + "\n")
    print("torch already installed -> leaving torch/torchvision untouched")
    return tmp


def build_plan(a):
    ds, seeds = a.datasets, [str(s) for s in a.seeds]
    meth = ["--methods", *a.methods] if a.methods else []
    dev = ["--device", a.device] if a.device else []
    run = lambda *x: [PY, str(ROOT / "scripts" / "run_experiment.py"), *x]  # noqa: E731
    plan = {}
    plan["install"] = [[PY, "-m", "pip", "install", "-r", "@REQ@"]]
    tv = missing_torchvision_cmd()
    if tv:                                   # torch present, torchvision absent -> add the matching one (no torch upgrade)
        plan["install"].append(tv)
    plan["smoke"] = [[PY, "-c", "from fuguard_repro.smoke import run_smoke_test; print(run_smoke_test().to_string())"]]
    prep = [PY, str(ROOT / "scripts" / "prepare_data.py"), "--datasets", *ds, "--seeds", *seeds]
    if "scalability" in a.stages and "SVHN" in ds:
        prep.append("--clients100")
    plan["data"] = [prep]
    plan["table2"] = [run("table2", "--datasets", *ds, "--seeds", *seeds, *meth, *dev)]
    plan["table3"] = [run("table3", "--datasets", *ds, *meth, *dev)]                  # paper: fixed seed 0 only
    plan["ablation"] = [run("ablation", "--datasets", "SVHN", "--seeds", *seeds, *dev)] if "SVHN" in ds else []
    plan["sensitivity"] = [run("sensitivity", "--datasets", *ds, "--seeds", *seeds, *dev)]
    plan["scalability"] = [run("scalability", "--seeds", *seeds, *meth, *dev)] if "SVHN" in ds else []
    awe = []
    if a.with_awe:
        src = "None" if a.with_awe == "clone" else repr(a.with_awe)
        awe.append([PY, "-c", f"from fuguard_repro.colab import fetch_awe_dataset; fetch_awe_dataset({src})"])
        awe.append([PY, str(ROOT / "scripts" / "prepare_data.py"), "--awe"])
        awe.append(run("awe", *dev))
    plan["awe"] = awe
    plan["reports"] = [[PY, str(ROOT / "scripts" / "make_reports.py")]]
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", choices=ALL_STAGES, default=None)
    ap.add_argument("--datasets", nargs="+", choices=IMAGE_DATASETS, default=IMAGE_DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--methods", nargs="+", default=None, help="subset of retrain fedsga fu not fastfedul fuguard")
    ap.add_argument("--device", default=None, help="e.g. cuda:0 (default: auto)")
    ap.add_argument("--with-awe", metavar="clone|PATH", default=None, help="also run AWE; 'clone' = git clone, or a zip/folder path")
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--quick", action="store_true", help="pilot: CIFAR10, seed 0, fedsga+fuguard, table2 only")
    ap.add_argument("--keep-going", action="store_true", help="continue with the next stage if one fails")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-y", "--yes", action="store_true", help="don't ask questions")
    a = ap.parse_args()

    if a.quick:
        a.datasets, a.seeds, a.methods = ["CIFAR10"], [0], ["fedsga", "fuguard"]
        a.stages = a.stages or ["install", "smoke", "data", "table2", "reports"]
    a.stages = a.stages or [s for s in ALL_STAGES if s != "awe" or a.with_awe]
    if a.skip_install and "install" in a.stages:
        a.stages.remove("install")
    if "awe" in a.stages and not a.with_awe:
        sys.exit("stage 'awe' needs --with-awe clone|PATH")

    free = shutil.disk_usage(ROOT).free / 1e9
    print(f"project: {ROOT}\nstages : {a.stages}\ndatasets={a.datasets} seeds={a.seeds} methods={a.methods or 'all'} | free disk {free:.0f} GB")
    if free < 15 and not a.dry_run:
        print("WARNING: less than 15 GB free; the full run needs roughly 10 GB for data plus checkpoints/results.")

    plan = build_plan(a)
    (ROOT / "logs").mkdir(exist_ok=True)
    timings, failed = [], []
    for stage in a.stages:
        cmds = plan[stage]
        if not cmds:
            print(f"\n[{stage}] nothing to do for this selection (needs SVHN / AWE options)")
            continue
        for cmd in cmds:
            if "@REQ@" in cmd:
                cmd = [str(requirements_file(a.yes)) if c == "@REQ@" else c for c in cmd] if not a.dry_run else cmd
            print(f"\n=== [{stage}] " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
            if a.dry_run:
                continue
            t0, log = time.time(), ROOT / "logs" / f"{stage}.log"
            with open(log, "a", encoding="utf-8") as lf:
                env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}   # Windows consoles choke on lambda/± otherwise
                p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1)
                for line in p.stdout:          # stream to console and log
                    sys.stdout.write(line)
                    lf.write(line)
                rc = p.wait()
            timings.append((stage, time.time() - t0))
            if rc != 0:
                failed.append(stage)
                print(f"\n!!! stage '{stage}' failed (exit {rc}); see {log}")
                if not a.keep_going:
                    sys.exit(rc)
                break
    print("\n--- summary ---")
    for s, t in timings:
        print(f"{s:<12}{t / 60:7.1f} min")
    print("failed stages:", failed or "none", "\nresults: results/tables, results/figures, results/raw")


if __name__ == "__main__":
    main()
