"""Metrics: R-A / F-A (client, class), 0A / PS (sample), ACC / ASR (client_backdoor). All values are fractions in [0, 1]."""
import numpy as np
import torch

from .fl import predict


def _cell_acc(pred, y):
    """{class: accuracy} for the classes present in y."""
    return {int(c): float((pred[y == c] == c).float().mean()) for c in torch.unique(y)}


def _mean(xs):
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def eval_client(model, view, cfg, device):
    """R-A: accuracy on the retained clients' test splits; F-A: on the forgotten client's (label-flipped) test split.
    'class_mean' (official): mean over (client, class) cells; 'pooled': plain accuracy."""
    ra_cells, fa_cells, ra_hit, ra_n, fa_hit, fa_n = [], [], 0, 0, 0, 0
    for i, c in enumerate(view["clients"]):
        if len(c["yt"]) == 0:
            continue
        pred = predict(model, c["xt"], cfg, device).argmax(1)
        cells = _cell_acc(pred, c["yt"])
        hit = int((pred == c["yt"]).sum())
        if i in view["forget_clients"]:
            fa_cells += list(cells.values()); fa_hit += hit; fa_n += len(c["yt"])
        else:
            ra_cells += list(cells.values()); ra_hit += hit; ra_n += len(c["yt"])
    if cfg["r_a_style"] == "class_mean":
        return dict(R_A=_mean(ra_cells), F_A=_mean(fa_cells))
    return dict(R_A=ra_hit / max(ra_n, 1), F_A=(fa_hit / fa_n) if fa_n else float("nan"))


def eval_class(model, view, cfg, device):
    x, y = view["global_test"]
    pred = predict(model, x, cfg, device).argmax(1)
    cells = _cell_acc(pred, y)
    fc = set(int(c) for c in cfg["forget_classes"])
    if cfg["r_a_style"] == "class_mean":
        return dict(R_A=_mean(a for c, a in cells.items() if c not in fc), F_A=_mean(a for c, a in cells.items() if c in fc))
    m = torch.tensor([int(v) in fc for v in y])
    return dict(R_A=float((pred[~m] == y[~m]).float().mean()), F_A=float((pred[m] == y[m]).float().mean()) if m.any() else float("nan"))


def eval_sample(model, view, cfg, device):
    """Triggered copy of the whole test set with ORIGINAL labels: 0A = recall of class 0, PS = precision of class 0
    (both drop when triggered inputs are sent to class 0, i.e. when the backdoor works). R_A = clean test accuracy."""
    x, y = view["global_test"]
    t = int(cfg["sample_target"])
    clean = predict(model, x, cfg, device).argmax(1)
    trig = predict(model, x, cfg, device, poison=torch.ones(len(y), dtype=torch.bool), trigger=view["trigger"]).argmax(1)
    tp = int(((trig == t) & (y == t)).sum())
    n_pred = int((trig == t).sum())
    non = y != t
    return dict(R_A=float((clean == y).float().mean()), R_A_cells=_mean(_cell_acc(clean, y).values()),
                **{"0A": float((trig[y == t] == t).float().mean()) if (y == t).any() else float("nan"),
                   "PS": (tp / n_pred) if n_pred else 0.0,
                   "ASR0": float((trig[non] == t).float().mean())})


def eval_backdoor_client(model, view, cfg, device):
    """FuGuard-style: ACC on the clean test set; ASR on triggered NON-target test images (strict) and
    'ASR_incl_target' = FuGuard's definition (fraction of the whole triggered test set predicted as the target)."""
    x, y = view["global_test"]
    t = int(cfg["bd_target"])
    clean = predict(model, x, cfg, device).argmax(1)
    non = y != t
    trig = predict(model, x[non], cfg, device, poison=torch.ones(int(non.sum()), dtype=torch.bool),
                   trigger=view["trigger"]).argmax(1)
    asr = float((trig == t).float().mean())
    n_t = int((~non).sum())
    hit_t = int((clean[~non] == t).sum())
    return dict(ACC=float((clean == y).float().mean()), ASR=asr,
                ASR_incl_target=(int((trig == t).sum()) + hit_t) / len(y), ACC_cells=_mean(_cell_acc(clean, y).values()))


def evaluate(model, view, cfg, device):
    sc = view["scenario"]
    return {"client": eval_client, "class": eval_class, "sample": eval_sample,
            "client_backdoor": lambda *a: {**eval_backdoor_client(*a), **eval_client(*a)}}[sc](model, view, cfg, device)
