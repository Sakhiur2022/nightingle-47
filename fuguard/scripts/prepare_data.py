#!/usr/bin/env python
"""Prepare data from the command line.  python scripts/prepare_data.py --datasets CIFAR10 --seeds 0 1 [--clients100] [--awe]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fuguard_repro.config import IMAGE_DATASETS, image_config, awe_config  # noqa: E402
from fuguard_repro.data_image import prepare_image_data  # noqa: E402
from fuguard_repro.data_awe import prepare_awe_data  # noqa: E402
from fuguard_repro.paths import ensure_dirs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="*", default=[])
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
ap.add_argument("--clients100", action="store_true", help="also prepare 100-client SVHN (Fig. 9)")
ap.add_argument("--awe", action="store_true")
a = ap.parse_args()
ensure_dirs()
for ds in a.datasets:
    for s in a.seeds:
        prepare_image_data(image_config(ds), s)
if a.clients100:
    for s in a.seeds:
        prepare_image_data(image_config("SVHN", num_clients=100, tag="SVHN-100"), s)
if a.awe:
    prepare_awe_data(awe_config())
