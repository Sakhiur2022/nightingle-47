"""AWE task loader (torch). All preparation code lives in awe_prep.py (numpy only) and is re-exported here."""
import numpy as np
import torch

from .awe_prep import (awe_dir, prepare_awe_data, find_awe_root, list_dataset, add_spike_trigger,  # noqa: F401
                       preprocess, load_group, group_file, state_dir)
from .paths import PROCESSED_DIR
from .tasks import AWETask
from .utils import get_device


def load_awe_task(cfg, device=None, root=PROCESSED_DIR):
    device = get_device(device) if not isinstance(device, torch.device) else device
    out = awe_dir(cfg, root)
    if not (out / "global_test.npz").exists():
        raise FileNotFoundError(f"{out} missing -> run the AWE data-preparation block (01b) first")
    clients = {}
    for i in range(cfg["num_clients"]):
        d = np.load(out / f"client_{i}_train.npz")
        clients[f"client_{i}"] = (torch.from_numpy(d["train_X"].astype(np.float32)),
                                  torch.from_numpy(d["train_y"].astype(np.int64)))
    g = np.load(out / "global_test.npz")
    t = lambda a, dt: torch.from_numpy(a.astype(dt))  # noqa: E731
    return AWETask(cfg, device, clients, (t(g["X_clean"], np.float32), t(g["y_clean"], np.int64)),
                   (t(g["X_backdoor"], np.float32), t(g["y_backdoor"], np.int64)))
