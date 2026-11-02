#!/usr/bin/env python
"""Download datasets to a folder of your choice, one at a time or all at once.

  python scripts/download_datasets.py --dest E:\\fuguard_data --only cifar10
  python scripts/download_datasets.py --dest E:\\fuguard_data --only svhn          # official site, then Hugging Face mirror
  python scripts/download_datasets.py --dest E:\\fuguard_data --only svhn --source hf   # skip the (blocked?) official site
  python scripts/download_datasets.py --dest E:\\fuguard_data --only cifar100 --source hf
  python scripts/download_datasets.py --dest E:\\fuguard_data --only eurosat
  python scripts/download_datasets.py --dest E:\\fuguard_data --only awe --awe-source C:\\Users\\Sakhiur\\Downloads\\NGIT-2022-AWE-Industrial-Dataset-master.zip
  python scripts/download_datasets.py --dest E:\\fuguard_data --check               # what is already there
  python scripts/download_datasets.py --dest E:\\fuguard_data                      # everything

Afterwards tell the project where the data is:   set FUGUARD_DATA_DIR=E:\\fuguard_data   (PowerShell: $env:FUGUARD_DATA_DIR="E:\\fuguard_data")
"""
import argparse
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dest", required=True, help="folder to download into (created if missing)")
ap.add_argument("--only", nargs="+", choices=["cifar10", "svhn", "cifar100", "eurosat", "awe"], default=None)
ap.add_argument("--source", choices=["auto", "official", "hf"], default="auto",
                help="svhn/cifar100: auto = official site first, then Hugging Face mirror; hf = mirror only")
ap.add_argument("--awe-source", default=None, help="zip or folder you already have (default: git clone)")
ap.add_argument("--timeout", type=int, default=30, help="seconds before an official-site download gives up")
ap.add_argument("--check", action="store_true", help="only report what is present")
a = ap.parse_args()

dest = Path(a.dest).expanduser().resolve()
dest.mkdir(parents=True, exist_ok=True)
os.environ["FUGUARD_DATA_DIR"] = str(dest)                       # must be set before importing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fuguard_repro import downloaders as dl  # noqa: E402

if a.check:
    print(dest, dl.status(dest))
    sys.exit(0)
todo = a.only or ["cifar10", "svhn", "cifar100", "eurosat", "awe"]
failed = []
for name in todo:
    print(f"\n=== {name} -> {dest}")
    try:
        if name == "cifar10":
            dl.get_cifar10(dest)
        elif name == "svhn":
            dl.get_svhn(dest, a.source, a.timeout)
        elif name == "cifar100":
            dl.get_cifar100(dest, a.source, a.timeout)
        elif name == "eurosat":
            dl.get_eurosat(dest)
        elif name == "awe":
            dl.get_awe(dest, a.awe_source)
    except Exception as e:  # noqa: BLE001
        failed.append(name)
        print(f"!!! {name} failed: {e!r}")
print("\nstatus:", dl.status(dest))
print(f'\nNext:  set FUGUARD_DATA_DIR={dest}    (PowerShell: $env:FUGUARD_DATA_DIR="{dest}")')
sys.exit(1 if failed else 0)
