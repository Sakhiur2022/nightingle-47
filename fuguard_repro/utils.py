import random
import numpy as np
import torch


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer=None):
    if prefer is not None:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def iterate_batches(x, y, bs, shuffle):
    """Mini-batch iterator over tensors that already live on the compute device."""
    n = len(x)
    order = torch.randperm(n, device=x.device) if shuffle else torch.arange(n, device=x.device)
    for i in range(0, n, bs):
        idx = order[i:i + bs]
        yield x[idx], y[idx]


def clone_state(model_or_state):
    sd = model_or_state.state_dict() if hasattr(model_or_state, "state_dict") else model_or_state
    return {k: v.detach().clone() for k, v in sd.items()}


def state_to(state, device):
    return {k: v.to(device) for k, v in state.items()}
