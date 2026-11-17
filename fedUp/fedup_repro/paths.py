"""Project-relative paths.

Override the root with FEDUP_ROOT, the data folder with FEDUP_DATA_DIR and the artifacts folder with FEDUP_ARTIFACTS_DIR
(e.g. point them at /kaggle/working/... or a mounted Drive so they survive a recycled runtime).
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("FEDUP_ROOT", Path(__file__).resolve().parent.parent)).resolve()
DATA_DIR = Path(os.environ.get("FEDUP_DATA_DIR", ROOT / "data")).resolve()
RAW_DATA_DIR = DATA_DIR / "raw"                  # downloaded datasets (MNIST, CIFAR, AG News, tokenised arrays)
PROCESSED_DIR = DATA_DIR / "processed"           # federated splits (Dirichlet partition + proxy hold-out), per seed
ARTIFACTS_DIR = Path(os.environ.get("FEDUP_ARTIFACTS_DIR", ROOT / "artifacts")).resolve()
CKPT_DIR = ARTIFACTS_DIR / "checkpoints"         # cached FedAvg models (pre-unlearning) and Retrain models
FEATURE_DIR = ARTIFACTS_DIR / "features"         # cached per-client feature arrays (optional)
FILTER_DIR = ARTIFACTS_DIR / "filters"           # trained FedUP plug-in filters (small, removable = reversible unlearning)
RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"                    # one .pkl (+ .json summary) per run
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = ROOT / "logs"


def ensure_dirs():
    for p in (RAW_DATA_DIR, PROCESSED_DIR, CKPT_DIR, FEATURE_DIR, FILTER_DIR, RAW_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR):
        p.mkdir(parents=True, exist_ok=True)
