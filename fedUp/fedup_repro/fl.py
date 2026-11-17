"""Federated training engine (FedAvg) with a cached pre-unlearning checkpoint."""
import copy
import time

import torch
import torch.nn.functional as F

from .config import n_rounds, train_key
from .data import cat_clients, iterate
from .models import build_model
from .paths import CKPT_DIR
from .utils import set_seed, tensor_bytes


def _optimizer(model, cfg):
    lr, kind = cfg["lr"], cfg["optimizer"]
    if kind == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if kind == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])


def _scaler(cfg, device):
    on = bool(cfg["amp"]) and device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=on)
    except Exception:  # noqa: BLE001  (older torch)
        return torch.cuda.amp.GradScaler(enabled=on)


def train_epochs(model, x, y, cfg, device, epochs, poison=None, trigger=None, seed=0, bs=None, opt=None):
    """Local / central training on (x, y) for `epochs` epochs. Returns the number of samples seen per epoch."""
    model.train()
    opt = opt or _optimizer(model, cfg)
    scaler = _scaler(cfg, device)
    gen = torch.Generator().manual_seed(seed)
    use_amp = bool(cfg["amp"]) and device.type == "cuda"
    for _ in range(epochs):
        for xb, yb in iterate(x, y, bs or cfg["local_batch"], cfg, device, shuffle=True, poison=poison, trigger=trigger,
                              gen=gen, drop_single=True):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = F.cross_entropy(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
    return len(y)


def fedavg(states, sizes, mode="uniform"):
    """Average client state_dicts. mode 'uniform' = plain mean [official code]; 'size' = weighted by |D_n| [paper]."""
    n = len(states)
    w = [1.0 / n] * n if mode == "uniform" else [s / float(sum(sizes)) for s in sizes]
    out = {}
    for k, v0 in states[0].items():
        if v0.dtype.is_floating_point:
            out[k] = sum(wi * s[k].float() for wi, s in zip(w, states)).to(v0.dtype)
        else:                                            # e.g. BatchNorm num_batches_tracked
            out[k] = (sum(s[k].long() for s in states) // n).to(v0.dtype)
    return out


@torch.no_grad()
def predict(model, x, cfg, device, bs=None, poison=None, trigger=None):
    model.eval()
    outs = []
    for xb, _ in iterate(x, torch.zeros(len(x), dtype=torch.long), bs or cfg["eval_batch"], cfg, device, poison=poison,
                         trigger=trigger):
        outs.append(model(xb).float().cpu())
    return torch.cat(outs) if outs else torch.zeros(0)


def pooled_acc(model, x, y, cfg, device, n_max=2000):
    if len(y) == 0:
        return float("nan")
    k = min(len(y), n_max)
    return float((predict(model, x[:k], cfg, device).argmax(1) == y[:k]).float().mean())


def monitor_set(view, n_max=2000, seed=0):
    """Held-out data used ONLY for the progress line printed during FedAvg (never for the reported metrics).
    Client scenario: the train+test pool is merged and partitioned, so the official test set is part of the training data;
    we monitor on the retained clients' own test splits instead. Other scenarios: the official test set."""
    if view.get("merged"):
        ids = [i for i in range(len(view["clients"])) if i not in view["forget_clients"]]
        x, y, _ = cat_clients(view, ids, "test")
    else:
        x, y = view["global_test"]
    idx = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed))[:n_max]
    return x[idx], y[idx]


def train_federated(view, cfg, seed, device, client_ids=None, rounds=None, init_state=None, save_local=(), tag="",
                    verbose=True):
    """FedAvg over `client_ids` (default: all) for `rounds` rounds (default: global_epochs * rounds_per_epoch).
    Returns dict(state, history, time_s, comm_bytes, rounds, local_states {client: state} for `save_local`)."""
    ids = list(range(len(view["clients"]))) if client_ids is None else list(client_ids)
    R = rounds if rounds is not None else n_rounds(cfg)
    set_seed(seed)
    model = build_model(cfg)
    if init_state is not None:
        model.load_state_dict(init_state)
    model.to(device)
    g_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model_bytes = tensor_bytes(g_state)
    history, local_keep = [], {}
    t0 = time.perf_counter()
    gx, gy = monitor_set(view)
    for r in range(R):
        states, sizes = [], []
        for c in ids:
            model.load_state_dict(g_state)
            cl = view["clients"][c]
            n = train_epochs(model, cl["x"], cl["y"], cfg, device, cfg["local_epochs"], poison=cl["poison"],
                             trigger=view["trigger"], seed=seed * 100003 + r * 101 + c)
            states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            sizes.append(n)
            if r == R - 1 and c in save_local:
                local_keep[c] = states[-1]
        g_state = fedavg(states, sizes, cfg["aggregation"])
        if (r + 1) % max(1, int(cfg["eval_every"])) == 0 or r == R - 1:
            model.load_state_dict(g_state)
            acc = pooled_acc(model, gx, gy, cfg, device)
            history.append((r + 1, acc))
            if verbose:
                print(f"[fl{tag}] round {r + 1}/{R}  held-out acc {acc:.4f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    t = time.perf_counter() - t0
    return dict(state=g_state, history=history, time_s=t, comm_bytes=2 * model_bytes * len(ids) * R, rounds=R,
                local_states=local_keep)


def train_or_load(view, cfg, seed, device, verbose=True):
    """Cached FedAvg model of the scenario's training data (shared by all unlearning methods of that seed)."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = CKPT_DIR / f"{train_key(cfg, seed)}.pt"
    if path.exists():
        if verbose:
            print(f"[fl] cached pre-unlearning model: {path.name}")
        return torch.load(path, weights_only=False)
    res = train_federated(view, cfg, seed, device, save_local=tuple(view["forget_clients"]), verbose=verbose)
    torch.save(res, path)
    return res


def load_into(cfg, state, device):
    m = build_model(cfg)
    m.load_state_dict(state)
    return m.to(device).eval()
