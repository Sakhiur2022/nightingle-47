import contextlib
import os
import random
import time

import numpy as np

STD_DDOF = int(os.environ.get("FEDUP_STD_DDOF", 0))     # 0 = population std (as in the FuGuard reproduction); 1 = sample std


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device=None):
    import torch
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextlib.contextmanager
def timer(device=None):
    """with timer(dev) as t: ...; t.s -> seconds (GPU-synchronised)"""
    class T:
        s = 0.0
    t = T()
    if device is not None:
        sync(device)
    t0 = time.perf_counter()
    try:
        yield t
    finally:
        if device is not None:
            sync(device)
        t.s = time.perf_counter() - t0


def tensor_bytes(obj):
    """Bytes of a tensor, a state_dict or an iterable of them (as stored/transmitted)."""
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(v) for v in obj)
    return 0


def mean_std(values, ddof=None):
    a = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))], dtype=float)
    if len(a) == 0:
        return float("nan"), float("nan")
    d = STD_DDOF if ddof is None else ddof
    return float(a.mean()), (float(a.std(ddof=d)) if len(a) > d else 0.0)
