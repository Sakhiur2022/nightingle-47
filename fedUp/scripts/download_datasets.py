#!/usr/bin/env python
"""Download / cache the raw datasets under data/raw (MNIST, CIFAR-10, CIFAR-100, AG News tokenised with the text model; SVHN and EuroSAT
are needed only for the FuGuard-protocol comparison and are fetched with --only svhn eurosat)."""
import argparse

import _bootstrap  # noqa: F401
from fedup_repro import downloaders
from fedup_repro.config import BASE
from fedup_repro.paths import RAW_DATA_DIR, ensure_dirs

ap = argparse.ArgumentParser()
ap.add_argument("--only", nargs="+", choices=["mnist", "cifar10", "cifar100", "svhn", "eurosat", "agnews"], default=["mnist", "cifar10", "cifar100", "agnews"])
ap.add_argument("--text-model", default=BASE["text_model"])
a = ap.parse_args()
ensure_dirs()
for name in a.only:
    if name == "agnews":
        tx, ty, ex, ey = downloaders.load_agnews_arrays(a.text_model, BASE["max_len"])
    else:
        tx, ty, ex, ey = downloaders.load_image_arrays(name)
    print(f"{name:9s} train {tx.shape} test {ex.shape}  classes {len(set(ty.tolist()))}  -> {RAW_DATA_DIR}")
