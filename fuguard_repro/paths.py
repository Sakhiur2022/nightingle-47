"""Project-relative paths. Override the root with the FUGUARD_ROOT environment variable."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("FUGUARD_ROOT", Path(__file__).resolve().parent.parent)).resolve()
DATA_DIR = ROOT / "data"                      # raw downloads (torchvision, EuroSAT, AWE json)
PROCESSED_DIR = DATA_DIR / "processed"        # federated splits + backdoored data
CKPT_DIR = ROOT / "artifacts" / "checkpoints" # shared pre-unlearning training checkpoints
RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"                 # one .pkl (+ .json summary) per run
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"


def ensure_dirs():
    for p in (DATA_DIR, PROCESSED_DIR, CKPT_DIR, RAW_DIR, TABLES_DIR, FIGURES_DIR):
        p.mkdir(parents=True, exist_ok=True)
