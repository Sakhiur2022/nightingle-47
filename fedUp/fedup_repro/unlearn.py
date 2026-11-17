"""Unlearning methods. Each takes a context (cfg, view, device, seed, model = pre-unlearning FedAvg model, base = checkpoint)
and returns dict(model=<module used for evaluation>, time_s, comm_bytes, info, ...).

 fedup        the paper's method: clients upload DP class centroids once, the server trains a plug-in filter
 retrain      FedAvg from scratch on the retained data (gold standard, expensive)
 eraseclient  Halimi et al. reference-model + projected gradient ascent (ported from FuGuard's FU baseline), client scenarios only
 federaser / exactfun / fused   documented stubs: the paper compares against them but the released FedUP code does not
                                include them and no reference code is available here (see docs/FIXES_AND_DEVIATIONS.md)
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import train_key
from .data import iterate
from .features import extract_by_class, make_centroids, upload_bytes
from .filter import train_filter
from .fl import load_into, train_federated
from .models import build_model
from .paths import CKPT_DIR
from .utils import tensor_bytes, timer


# ------------------------------------------------------------------------------------------------ FedUP
def _relabel_forgotten(feats, cfg, rng):
    """Class unlearning: the features of every forgotten class are given a random OTHER label (one draw per client/class),
    so the filter learns to send them away from their true class."""
    K, fc = cfg["num_classes"], set(int(c) for c in cfg["forget_classes"])
    out = {}
    for c, f in feats.items():
        lab = c
        if c in fc:
            lab = int(rng.choice([k for k in range(K) if k not in fc]))
        out[lab] = torch.cat([out[lab], f]) if lab in out else f
    return out


def run_fedup(ctx):
    cfg, view, dev, base_model, seed = ctx.cfg, ctx.view, ctx.device, ctx.model, ctx.seed
    sc = view["scenario"]
    rng = np.random.RandomState(3000 + seed)
    n = len(view["clients"])
    ids = [i for i in range(n) if i not in view["forget_clients"]]       # the forgotten client uploads nothing
    cents, t_clients = {}, []
    for i in ids:
        c = view["clients"][i]
        keep = (~c["poison"]) if sc == "sample" else None                 # sample unlearning: drop the backdoored samples
        with timer(dev) as t:
            f = extract_by_class(base_model, c["x"], c["y"], cfg, dev, keep=keep)
            if sc == "class":
                f = _relabel_forgotten(f, cfg, rng)
            cents[i] = make_centroids(f, cfg["rho"], cfg["centroid_method"], cfg["dp_sigma"], dev, seed=seed * 1009 + i)
        t_clients.append(t.s)
    out = train_filter(base_model, cents, cfg, dev, seed=seed, tag=f"{train_key(cfg, seed)}_{sc}_fedup")
    t_client = max(t_clients) if t_clients else 0.0                        # clients work in parallel
    return dict(model=out["filtered"], time_s=t_client + out["time_s"], time_filter_s=out["time_s"], time_client_s=t_client,
                comm_bytes=upload_bytes(cents),
                info=dict(n_centroids=out["n_centroids"], filter_kb=out["filter_bytes"] / 1024,
                          final_filter_loss=out["history"][-1][1], final_filter_acc=out["history"][-1][2]))


# ------------------------------------------------------------------------------------------------ Retrain
def retained_view(view, cfg):
    sc, fg = view["scenario"], set(view["forget_clients"])
    fc = torch.tensor([int(c) for c in cfg["forget_classes"]])
    clients = []
    for i, c in enumerate(view["clients"]):
        if i in fg:
            continue
        v = dict(c)
        if sc == "class":
            m = ~torch.isin(c["y"], fc)
            v.update(x=c["x"][m], y=c["y"][m], poison=None)
        elif sc == "sample":
            m = ~c["poison"]
            v.update(x=c["x"][m], y=c["y"][m], poison=None)
        clients.append(v)
    out = dict(view)
    out.update(clients=clients, forget_clients=[])
    return out


def run_retrain(ctx):
    cfg, view, dev, seed = ctx.cfg, ctx.view, ctx.device, ctx.seed
    suffix = "_c" + "-".join(map(str, cfg["forget_classes"])) if view["scenario"] == "class" else ""
    path = CKPT_DIR / f"{train_key(cfg, seed)}_retrain{suffix}.pt"
    if path.exists():
        res = torch.load(path, weights_only=False)
    else:
        res = train_federated(retained_view(view, cfg), cfg, seed, dev, tag="-retrain")
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(res, path)
    return dict(model=load_into(cfg, res["state"], dev), time_s=res["time_s"], comm_bytes=res["comm_bytes"],
                info=dict(rounds=res["rounds"]))


# ------------------------------------------------------------------------------------------------ EraseClient
def _vec(m):
    return nn.utils.parameters_to_vector(m.parameters())


def _dist(a, b):
    return torch.norm(_vec(a) - _vec(b))


def run_eraseclient(ctx):
    cfg, view, dev, seed, base = ctx.cfg, ctx.view, ctx.device, ctx.seed, ctx.base
    if view["scenario"] not in ("client", "client_backdoor"):
        raise ValueError("eraseclient only supports the client scenarios")
    f = view["forget_clients"][0]
    N = len(view["clients"])
    g_model, l_model = load_into(cfg, base["state"], dev), load_into(cfg, base["local_states"][f], dev)
    ref = load_into(cfg, base["state"], dev)
    with timer(dev) as t:
        ref_vec = N / (N - 1) * _vec(g_model).detach() - 1 / (N - 1) * _vec(l_model).detach()   # Halimi et al. reference model
        nn.utils.vector_to_parameters(ref_vec.clone(), ref.parameters())
        rand_cfg = dict(cfg, pretrained=False)
        rand = [_dist(ref, build_model(rand_cfg).to(dev)).item() for _ in range(3)]
        thr = float(np.mean(rand)) / 3                                  # as in IBM's federated-unlearning notebook
        model = copy.deepcopy(ref).train()
        opt = torch.optim.SGD(model.parameters(), lr=cfg["ec_lr"])
        c = view["clients"][f]
        gen = torch.Generator().manual_seed(seed)
        stop = False
        for _ in range(int(cfg["ec_epochs"])):
            if stop:
                break
            for xb, yb in iterate(c["x"], c["y"], cfg["local_batch"], cfg, dev, shuffle=True, poison=c["poison"],
                                  trigger=view["trigger"], gen=gen, drop_single=True):
                opt.zero_grad()
                (-F.cross_entropy(model(xb), yb)).backward()             # gradient ASCENT on the forgotten client's data
                opt.step()
                with torch.no_grad():
                    if _dist(model, ref) > thr:                          # project back into the ball around the reference
                        dv = _vec(model) - _vec(ref)
                        nrm = torch.norm(dv)
                        if nrm > 1e-8:
                            dv = dv / nrm * float(np.sqrt(thr))
                        nn.utils.vector_to_parameters(_vec(ref) + dv, model.parameters())
                    if _dist(model, l_model) > cfg["ec_distance_threshold"]:
                        stop = True
                        break
    t_unl = t.s
    retained = [i for i in range(N) if i != f]
    rec = train_federated(view, cfg, seed, dev, client_ids=retained, rounds=int(cfg["ec_recovery_rounds"]),
                          init_state={k: v.detach().cpu() for k, v in model.state_dict().items()}, tag="-ec-recovery")
    return dict(model=load_into(cfg, rec["state"], dev), time_s=t_unl + rec["time_s"],
                comm_bytes=rec["comm_bytes"] + 2 * tensor_bytes(base["state"]), info=dict(threshold=thr, unlearn_s=t_unl))


def _stub(name):
    def run(ctx):
        raise NotImplementedError(
            f"'{name}' is compared in the FedUP paper, but the released FedUP code does not include it and no reference "
            f"implementation is bundled here. Implement it against its original paper, then register it in "
            f"fedup_repro.unlearn.REGISTRY (see docs/FIXES_AND_DEVIATIONS.md).")
    return run


REGISTRY = {"fedup": run_fedup, "retrain": run_retrain, "eraseclient": run_eraseclient,
            "federaser": _stub("FedEraser"), "exactfun": _stub("Exact-Fun"), "fused": _stub("FUSED")}
