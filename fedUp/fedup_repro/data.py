"""Federated data: raw loading -> (optional static augmentation) -> Dirichlet partition -> per-client proxy/train/test,
cached per seed under data/processed. Plus the scenario views (attacks), triggers and the batch iterator."""
import hashlib
import json

import numpy as np
import torch

from . import downloaders
from .config import BENCHMARKS
from .partition import dirichlet_partition, split_proxy_main, train_test_split_idx
from .paths import PROCESSED_DIR

CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)


# FuGuard's per-dataset statistics and its released trigger constants (normalised space), for the FuGuard-protocol benchmarks
FUGUARD_NORM = {"CIFAR10": ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
                "CIFAR100": ([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]),
                "SVHN": ([0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]),
                "EuroSAT": ([0.3443, 0.3809, 0.4082], [0.1504, 0.1238, 0.1086])}
FUGUARD_TRIGGER = {"CIFAR10": [2.0591, 2.1294, 2.1167], "SVHN": [2.8333, 2.7716, 2.6761],
                   "CIFAR100": [1.8456, 2.0050, 2.0221], "EuroSAT": [2.1872, 2.0016, 2.1832]}


def norm_stats(cfg):
    if cfg.get("norm_key"):
        return FUGUARD_NORM[cfg["norm_key"]]
    if cfg["dataset"] == "mnist":
        return MNIST_MEAN, MNIST_STD
    if cfg["dataset"] == "synthetic":
        return (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
    return CIFAR_MEAN, CIFAR_STD            # [repo] CIFAR-100 also uses the CIFAR-10 statistics


def load_raw(cfg):
    d = cfg["dataset"]
    if d == "agnews":
        return downloaders.load_agnews_arrays(cfg["text_model"], cfg["max_len"])
    if d == "synthetic":
        return downloaders.synthetic_arrays(cfg["num_classes"])
    return downloaders.load_image_arrays(d)


def static_augment(x, rng, pad=4):
    """One-time random crop (pad 4) + horizontal flip of every image, as the official code bakes it into the dataset
    (train AND test). x: uint8 [N,C,H,W]."""
    n, c, h, w = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    oy = rng.randint(0, 2 * pad + 1, size=n)
    ox = rng.randint(0, 2 * pad + 1, size=n)
    flip = rng.rand(n) < 0.5
    out = np.empty_like(x)
    for i in range(n):
        img = xp[i, :, oy[i]:oy[i] + h, ox[i]:ox[i] + w]
        out[i] = img[:, :, ::-1] if flip[i] else img
    return out


def _fed_path(cfg, seed):
    key = {k: cfg.get(k) for k in ["dataset", "num_classes", "num_clients", "alpha", "min_client_size", "proxy_frac",
                                   "train_ratio_client", "train_ratio_other", "data_fraction", "augment", "text_model",
                                   "max_len"]}
    key["merge"] = cfg["scenario"] == "client"
    h = hashlib.md5(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return PROCESSED_DIR / cfg["benchmark"] / f"fed_s{seed}_{h}.pt"


def _t(a):
    return torch.from_numpy(np.ascontiguousarray(a))


def build_fed_data(cfg, seed, force=False):
    """-> dict(clients=[dict(x_train,y_train,x_test,y_test,x_proxy,y_proxy)], global_test=(x,y), meta=...)"""
    path = _fed_path(cfg, seed)
    if path.exists() and not force:
        return torch.load(path, weights_only=False)
    rng = np.random.RandomState(1000 + seed)
    tx, ty, ex, ey = load_raw(cfg)
    if cfg["data_fraction"] < 1.0:
        def sub(x, y, f):
            k = max(int(len(y) * f), cfg["num_clients"] * 20)
            i = rng.permutation(len(y))[:k]
            return x[i], y[i]
        tx, ty = sub(tx, ty, cfg["data_fraction"])
        ex, ey = sub(ex, ey, cfg["data_fraction"])
    if cfg["modality"] == "image" and cfg["augment"] == "static" and cfg["dataset"] in ("cifar10", "cifar100"):
        tx, ex = static_augment(tx, rng), static_augment(ex, rng)
    merge = cfg["scenario"] == "client"
    px, py = (np.concatenate([tx, ex]), np.concatenate([ty, ey])) if merge else (tx, ty)
    parts = dirichlet_partition(py, cfg["num_clients"], cfg["num_classes"], cfg["alpha"], rng, cfg["min_client_size"])
    ratio = cfg["train_ratio_client"] if merge else cfg["train_ratio_other"]
    clients = []
    for idx in parts:
        main, proxy = split_proxy_main(idx, py, cfg["proxy_frac"], rng)
        tr, te = train_test_split_idx(main, ratio, rng)
        clients.append(dict(x_train=_t(px[tr]), y_train=_t(py[tr]), x_test=_t(px[te]), y_test=_t(py[te]),
                            x_proxy=_t(px[proxy]), y_proxy=_t(py[proxy])))
    fed = dict(clients=clients, global_test=(_t(ex), _t(ey)),
               meta=dict(sizes=[len(c["y_train"]) for c in clients], merge=merge, seed=seed))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fed, path)
    return fed


# ============================================================================== triggers and scenario views
def make_trigger(cfg, scenario=None):
    """Callable acting on a prepared batch (normalised float images [B,C,H,W], or int64 text [B,2,L])."""
    scenario = scenario or cfg["scenario"]
    if cfg["modality"] == "text":
        tid = int(cfg["text_trigger_id"])

        def trig_text(xb):
            xb = xb.clone()
            xb[:, 0, -1] = tid                # trigger token at the end of the sequence [paper/repo]
            xb[:, 1, -1] = 1
            return xb
        return trig_text
    mean, std = norm_stats(cfg)
    white = ((1.0 - torch.tensor(mean)) / torch.tensor(std)).view(1, -1, 1, 1)
    if scenario == "client_backdoor":
        p = int(cfg["bd_patch"])
        if cfg.get("bd_trigger") == "repo" and cfg.get("norm_key"):          # FuGuard's released constants (EuroSAT: gray)
            white = torch.tensor(FUGUARD_TRIGGER[cfg["norm_key"]]).view(1, -1, 1, 1)

        def trig_patch(xb):
            xb = xb.clone()
            xb[:, :, -p:, -p:] = white.to(xb.device, xb.dtype)
            return xb
        return trig_patch

    def trig_pixel(xb):                        # sample scenario: one bottom-right pixel
        xb = xb.clone()
        if cfg["trigger_mode"] == "repo":
            xb[:, :, -1, -1] = 255.0          # [repo] 255 written into the NORMALISED tensor (a very large value)
        else:
            xb[:, :, -1, -1] = white.to(xb.device, xb.dtype).view(1, -1)
        return xb
    return trig_pixel


def make_view(cfg, fed, seed):
    """Apply the scenario's attack to a copy of the labels. Returns dict(clients=[dict(x,y,poison,xt,yt,xp,yp)],
    global_test, forget_clients, trigger, scenario)."""
    sc, K = cfg["scenario"], cfg["num_classes"]
    rng = np.random.RandomState(2000 + seed)
    forget = list(cfg["forget_clients"]) if sc in ("client", "client_backdoor") else []
    clients = []
    for i, c in enumerate(fed["clients"]):
        v = dict(x=c["x_train"], y=c["y_train"].clone(), poison=None, xt=c["x_test"], yt=c["y_test"].clone(),
                 xp=c["x_proxy"], yp=c["y_proxy"].clone())
        if sc == "client" and i in forget:                       # Byzantine label flipping (+1 mod K) on train, test, proxy
            for k in ("y", "yt", "yp"):
                v[k] = (v[k] + 1) % K
        elif sc == "sample":                                     # backdoor a fraction of every client's samples -> class 0
            n = len(v["y"])
            m = torch.zeros(n, dtype=torch.bool)
            m[torch.from_numpy(rng.permutation(n)[:int(round(cfg["backdoor_frac"] * n))])] = True
            v["poison"] = m
            v["y"][m] = int(cfg["sample_target"])
        elif sc == "client_backdoor" and i in forget:            # FuGuard protocol: patch + relabel all non-target samples
            non_t = (v["y"] != int(cfg["bd_target"]))
            m = non_t & (torch.from_numpy(rng.rand(len(v["y"]))) < float(cfg["bd_ratio"]))
            v["poison"] = m
            v["y"][m] = int(cfg["bd_target"])
        clients.append(v)
    return dict(clients=clients, global_test=fed["global_test"], forget_clients=forget, merged=bool(fed["meta"]["merge"]),
                trigger=make_trigger(cfg), scenario=sc, num_classes=K)


# ============================================================================== batches
def prep(xb, cfg, device):
    """uint8 images -> normalised float; text stays int64."""
    xb = xb.to(device, non_blocking=True)
    if cfg["modality"] == "text":
        return xb.long()
    mean, std = norm_stats(cfg)
    m = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    s = torch.tensor(std, device=device).view(1, -1, 1, 1)
    return (xb.float() / 255.0 - m) / s


def iterate(x, y, bs, cfg, device, shuffle=False, poison=None, trigger=None, gen=None, drop_single=False):
    n = len(y)
    order = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        idx = order[i:i + bs]
        if drop_single and len(idx) < 2:
            continue
        xb = prep(x[idx], cfg, device)
        yb = y[idx].to(device)
        if poison is not None and trigger is not None:
            m = poison[idx].to(device)
            if bool(m.any()):
                xb = xb.clone()
                xb[m] = trigger(xb[m])
        yield xb, yb


def cat_clients(view, ids, which="train"):
    """Concatenate the train (or test / proxy) arrays of the given clients, with the poison masks."""
    xs, ys, ps = [], [], []
    for i in ids:
        c = view["clients"][i]
        xs.append(c["x"] if which == "train" else c["xt"] if which == "test" else c["xp"])
        ys.append(c["y"] if which == "train" else c["yt"] if which == "test" else c["yp"])
        if which == "train":
            ps.append(c["poison"] if c["poison"] is not None else torch.zeros(len(c["y"]), dtype=torch.bool))
    return torch.cat(xs), torch.cat(ys), (torch.cat(ps) if ps else None)
