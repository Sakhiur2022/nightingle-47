"""The six unlearning methods, ported from the official scripts (one function each).
Every method receives an UnlearnContext and modifies ctx.global_model IN PLACE."""
import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import iterate_batches, state_to


@dataclass
class UnlearnContext:
    task: object
    cfg: dict
    global_model: nn.Module
    target: str
    target_data: tuple              # (x, y) of the forgotten client on the compute device
    active: list                    # remaining clients
    local_states: dict              # last pre-unlearning local models {client: state_dict (cpu)}
    global_prev: dict               # global state at the START of the last pre-unlearning round (cpu)
    device: torch.device
    info: dict = field(default_factory=dict)


# ------------------------------------------------------------------ Retrain
def retrain(ctx):
    """Re-initialise the global model; the recovery rounds ARE the retraining (target client excluded)."""
    ctx.global_model.load_state_dict(ctx.task.make_model().state_dict())


# ------------------------------------------------------------------ FedSGA (Wu et al.)
def fedsga(ctx):
    cfg, model = ctx.cfg, ctx.global_model
    x, y = ctx.target_data
    opt = torch.optim.SGD(model.parameters(), lr=cfg["forgetting_lr"])
    model.train()
    for _ in range(cfg["forgetting_epoch"]):
        for xb, yb in iterate_batches(x, y, cfg["batch_size"], True):
            opt.zero_grad()
            (-F.cross_entropy(model(xb), yb)).backward()
            opt.step()


# ------------------------------------------------------------------ NoT (Khalil et al.)
def not_(ctx):
    with torch.no_grad():
        for p in ctx.global_model.parameters():
            p.mul_(-1.0)


# ------------------------------------------------------------------ Fast-FedUL (Huynh et al.)
def fastfedul(ctx):
    """Subtract the target client's last recorded update (local_model - global_at_that_round)."""
    scale = ctx.cfg["fastfedul_scale"]
    local = state_to(ctx.local_states[ctx.target], ctx.device)
    gprev = state_to(ctx.global_prev, ctx.device)
    state = ctx.global_model.state_dict()
    with torch.no_grad():
        for k in state:
            if state[k].is_floating_point():
                state[k].sub_(scale * (local[k] - gprev[k]))
    ctx.global_model.load_state_dict(state)


# ------------------------------------------------------------------ FU (Halimi et al.)
def _vec(model):
    return nn.utils.parameters_to_vector(model.parameters())


def _dist(a, b):
    with torch.no_grad():
        return torch.square(torch.norm(_vec(a) - _vec(b)))


def fu(ctx):
    cfg, g = ctx.cfg, ctx.global_model
    x, y = ctx.target_data
    n_parties = len(ctx.active) + 1
    client_model = ctx.task.make_model()
    client_model.load_state_dict(state_to(ctx.local_states[ctx.target], ctx.device))
    ref_vec = n_parties / (n_parties - 1) * _vec(g).detach() - 1 / (n_parties - 1) * _vec(client_model).detach()
    model_ref = ctx.task.make_model()
    nn.utils.vector_to_parameters(ref_vec.clone(), model_ref.parameters())
    rand = [_dist(model_ref, ctx.task.make_model()).item() for _ in range(10)]
    threshold = float(np.mean(rand)) / 3          # as in IBM's federated-unlearning notebook
    model = copy.deepcopy(model_ref)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg["forgetting_lr"])
    stop = False
    for _ in range(cfg["forgetting_epoch"]):
        if stop:
            break
        for xb, yb in iterate_batches(x, y, cfg["batch_size"], True):
            opt.zero_grad()
            (-F.cross_entropy(model(xb), yb)).backward()
            opt.step()
            with torch.no_grad():
                if _dist(model, model_ref) > threshold:
                    dv = _vec(model) - _vec(model_ref)
                    nrm = torch.norm(dv)
                    if nrm > 1e-8:
                        dv = dv / nrm * float(np.sqrt(threshold))
                    nn.utils.vector_to_parameters(_vec(model_ref) + dv, model.parameters())
            if _dist(model, client_model) > cfg["distance_threshold"]:
                stop = True
                break
    g.load_state_dict(model.state_dict())


# ------------------------------------------------------------------ FuGuard (this paper)
def fuguard(ctx):
    """Generative proxy (VAE + PCA shift) -> stochastic gradient ASCENT on the proxy,
    regularised by a Sinkhorn divergence between current and frozen pre-unlearning embeddings (Eq. 16)."""
    cfg, model = ctx.cfg, ctx.global_model
    lam = cfg["ot_lambda"]
    xp, yp = ctx.task.make_proxy(ctx.target_data[0], ctx.target_data[1], cfg)
    ctx.info["proxy_size"] = len(xp)
    bs = cfg["unlearn_bs"]
    old = copy.deepcopy(model).eval()
    with torch.no_grad():
        z_old = [old.embed(xp[i:i + bs]) for i in range(0, len(xp), bs)]
    del old
    sinkhorn = None
    if lam != 0:
        from geomloss import SamplesLoss
        sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=cfg["sinkhorn_blur"])
    opt = torch.optim.SGD(model.parameters(), lr=cfg["forgetting_lr"])
    model.train()
    for _ in range(cfg["forgetting_epoch"]):
        for j, i in enumerate(range(0, len(xp), bs)):
            xb, yb = xp[i:i + bs], yp[i:i + bs]
            opt.zero_grad()
            ce = F.cross_entropy(model(xb), yb)
            loss = -ce
            if sinkhorn is not None:
                loss = loss + lam * sinkhorn(model.embed(xb), z_old[j])
            loss.backward()
            opt.step()


REGISTRY = {"retrain": retrain, "fedsga": fedsga, "not": not_, "fastfedul": fastfedul,
            "fu": fu, "fuguard": fuguard}
