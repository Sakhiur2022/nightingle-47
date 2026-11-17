"""Membership-inference privacy risk after unlearning (our implementation of the official design).
Attack: an MLP on the model's logits that tries to tell FORGOTTEN samples (label 0) from RETAINED samples (label 1).
It is trained on a shadow model (the original model fine-tuned on proxy data without the forgotten part) and tested on the
real unlearned model. ~0.5 = cannot tell the two apart. Forgotten / retained sets are balanced 1:1 (the official code
keeps up to 2x retained samples). The official script does not return its number, so the exact paper value cannot be
recovered: treat this metric as comparable between methods, not with the paper's absolute values."""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import cat_clients
from .fl import predict, train_epochs


def _sets(view, cfg, source):
    """-> ((xf, pf), (xr, pr)): forgotten / retained inputs and optional poison masks (trigger applied at batch time)."""
    sc, n = view["scenario"], len(view["clients"])
    fg = view["forget_clients"]
    ids_f, ids_r = [i for i in range(n) if i in fg], [i for i in range(n) if i not in fg]
    which = "train" if source == "train" else "proxy"
    if sc in ("client", "client_backdoor"):
        xf, yf, pf = cat_clients(view, ids_f, which)
        xr, yr, pr = cat_clients(view, ids_r, which)
        if which == "proxy" and sc == "client_backdoor":          # proxy data is clean: give the forgotten part its trigger
            pf = torch.ones(len(xf), dtype=torch.bool)
        return (xf, pf), (xr, pr)
    x, y, p = cat_clients(view, list(range(n)), which)
    if sc == "class":
        m = torch.tensor([int(v) in set(int(c) for c in cfg["forget_classes"]) for v in y])
        return (x[m], None), (x[~m], None)
    # sample: forgotten = triggered samples, retained = clean samples
    if which == "train":
        return (x[p], torch.ones(int(p.sum()), dtype=torch.bool)), (x[~p], None)
    return (x, torch.ones(len(x), dtype=torch.bool)), (x, None)


def _logits(model, xs, ps, view, cfg, device, n_max, rng):
    x, p = xs
    if len(x) == 0:
        return torch.zeros(0, cfg["num_classes"])
    idx = torch.from_numpy(rng.permutation(len(x))[:n_max])
    return predict(model, x[idx], cfg, device, poison=(p[idx] if p is not None else None), trigger=view["trigger"])


def _attack_set(model, view, cfg, device, source, rng):
    f, r = _sets(view, cfg, source)
    n_max = int(cfg["mia_max"])
    lf = _logits(model, f, None, view, cfg, device, n_max, rng)
    if len(lf) == 0:
        return None
    lr = _logits(model, r, None, view, cfg, device, len(lf), rng)
    k = min(len(lf), len(lr))
    if k == 0:
        return None
    return torch.cat([lf[:k], lr[:k]]), torch.cat([torch.zeros(k), torch.ones(k)]).long()


class AttackNet(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(k, 20), nn.ReLU(), nn.Linear(20, 20), nn.ReLU(), nn.Linear(20, 2))

    def forward(self, x):
        return self.net(x)


def run_mia(target_model, base_model, view, cfg, device, seed=0):
    """-> attack accuracy in [0, 1] (nan when there is nothing to forget)."""
    rng = np.random.RandomState(4000 + seed)
    tgt = _attack_set(target_model, view, cfg, device, "train", rng)
    if tgt is None:
        return float("nan")
    # shadow: original model fine-tuned on retained proxy data (centrally, a few epochs)
    shadow = copy.deepcopy(base_model).to(device)
    sc, n = view["scenario"], len(view["clients"])
    keep_ids = [i for i in range(n) if i not in view["forget_clients"]] if sc in ("client", "client_backdoor") else list(range(n))
    xp, yp, _ = cat_clients(view, keep_ids, "proxy")
    if sc == "class":
        m = torch.tensor([int(v) not in set(int(c) for c in cfg["forget_classes"]) for v in yp])
        xp, yp = xp[m], yp[m]
    if len(yp) > 1:
        train_epochs(shadow, xp, yp, cfg, device, int(cfg["mia_shadow_epochs"]), seed=seed)
    sh = _attack_set(shadow, view, cfg, device, "proxy", rng)
    if sh is None:
        return float("nan")
    torch.manual_seed(seed)
    net = AttackNet(cfg["num_classes"]).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    xs, ys = sh[0].to(device), sh[1].to(device)
    for _ in range(int(cfg["mia_attack_epochs"])):
        perm = torch.randperm(len(ys), device=device)
        for i in range(0, len(ys), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            F.cross_entropy(net(xs[b]), ys[b]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return float((net(tgt[0].to(device)).argmax(1).cpu() == tgt[1]).float().mean())
