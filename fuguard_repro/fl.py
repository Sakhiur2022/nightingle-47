"""Federated training / unlearning / recovery engine (replaces the six near-identical official scripts).

Round semantics follow the official code:
  * rounds 1 .. unlearn_round-1 : plain FedAvg with all clients                      (shared across methods -> cached)
  * round  unlearn_round        : the unlearning request fires; NO aggregation that round; the model is evaluated
                                  right after unlearning ("only unlearn" numbers of Table IV)
  * rounds unlearn_round+1 .. + recovery_rounds : FedAvg over the remaining clients (the recovery phase)
Differences from the repo (documented in docs/FIXES_AND_DEVIATIONS.md): data lives on the GPU, the shared
pre-unlearning phase is trained once per (dataset, seed) and cached, clients do not waste a discarded training
pass in the unlearning round."""
import copy
import hashlib
import json
import time

import numpy as np
import torch
import torch.nn as nn

from .methods import REGISTRY, UnlearnContext
from .metrics import evaluate_acc, evaluate_asr, mia_loss_attack
from .paths import CKPT_DIR
from .utils import clone_state, iterate_batches, setup_seed, state_to

_CKPT_KEYS = ["dataset_name", "tag", "num_clients", "dirichlet_alpha", "backdoor_client", "target_label",
              "unlearn_round", "local_epochs", "batch_size", "lr", "optimizer", "inject_ratio", "num_classes",
              "data_seed", "trigger_amplitude"]


def _optimizer(params, cfg):
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(params, lr=cfg["lr"])
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(params, lr=cfg["lr"])
    raise ValueError(cfg["optimizer"])


def local_train(model, data, cfg):
    model.train()
    opt = _optimizer(model.parameters(), cfg)
    crit = nn.CrossEntropyLoss()
    total, steps = torch.zeros((), device=data[0].device), 0
    for _ in range(cfg["local_epochs"]):
        for xb, yb in iterate_batches(data[0], data[1], cfg["batch_size"], True):
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.detach()
            steps += 1
    return float(total / steps) if steps else 0.0


def fedavg_states(states, sizes):
    tot = float(sum(sizes))
    out = {}
    for k in states[0]:
        if states[0][k].is_floating_point():
            out[k] = sum(s[k] * (sz / tot) for s, sz in zip(states, sizes))
        else:
            out[k] = states[0][k].clone()
    return out


def _fl_round(task, cfg, global_model, local_model, active):
    """One FedAvg round. Returns (global state before aggregation, {client: local state}, mean loss)."""
    g_before = clone_state(global_model)
    states, sizes, names, losses = [], [], [], []
    for name in active:
        data = task.client_data(name)
        if len(data[0]) == 0:        # Dirichlet(alpha=0.1) can leave a client empty
            continue
        local_model.load_state_dict(g_before)
        losses.append(local_train(local_model, data, cfg))
        states.append(clone_state(local_model))
        sizes.append(len(data[0]))
        names.append(name)
    if states:
        global_model.load_state_dict(fedavg_states(states, sizes))
    return g_before, dict(zip(names, states)), float(np.mean(losses)) if losses else 0.0


def _evaluate(task, model):
    return (evaluate_acc(model, task.clean_test), evaluate_asr(model, task.poison_test, task.target_label))


def ckpt_path(cfg, seed):
    blob = json.dumps({k: cfg.get(k) for k in _CKPT_KEYS} | {"seed": seed}, sort_keys=True, default=str)
    h = hashlib.md5(blob.encode()).hexdigest()[:10]
    return CKPT_DIR / cfg["tag"] / f"seed{seed}_{h}.pt"


def train_phase(task, cfg, seed, use_cache=True, verbose=True):
    """FedAvg rounds 1..unlearn_round-1 (identical for every unlearning method). Cached on disk."""
    path = ckpt_path(cfg, seed)
    if use_cache and path.exists():
        if verbose:
            print(f"[cache] pre-unlearning checkpoint {path.name}")
        return torch.load(path, map_location="cpu", weights_only=False)
    n_rounds = cfg["unlearn_round"] - 1
    if n_rounds < 1:
        raise ValueError("unlearn_round must be >= 2")
    setup_seed(seed)
    g, local = task.make_model(), task.make_model()
    hist = {"acc": [], "asr": [], "loss": []}
    names = task.client_names
    for r in range(1, n_rounds + 1):
        t0 = time.time()
        g_before, states, loss = _fl_round(task, cfg, g, local, names)
        acc, asr = _evaluate(task, g)
        hist["acc"].append(acc); hist["asr"].append(asr); hist["loss"].append(loss)
        if verbose:
            print(f"[train r{r:>3}/{n_rounds}] acc={acc:.4f} asr={asr:.4f} loss={loss:.4f} ({time.time()-t0:.1f}s)")
    ckpt = {"global": state_to(clone_state(g), "cpu"), "global_prev": state_to(g_before, "cpu"),
            "local_states": {k: state_to(v, "cpu") for k, v in states.items()}, "history": hist, "seed": seed}
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, path)
    return ckpt


def unlearn_and_recover(task, cfg, method, ckpt, seed, verbose=True):
    if method not in REGISTRY:
        raise ValueError(f"unknown method {method}; choose from {list(REGISTRY)}")
    dev = task.device
    setup_seed(seed + 12345)                         # same post-checkpoint RNG stream for every method
    target, ur, R = cfg["backdoor_client"], cfg["unlearn_round"], cfg["recovery_rounds"]
    g = task.make_model()
    g.load_state_dict(state_to(ckpt["global"], dev))
    hist = {k: list(v) for k, v in ckpt["history"].items()}
    acc_before, asr_before = hist["acc"][-1], hist["asr"][-1]
    tx, ty = task.client_data(target)
    if len(tx) == 0:
        raise ValueError(f"{target} has no data for seed {seed}; pick another backdoor_client")
    mia_before = mia_loss_attack(g, (tx, ty), task.clean_test) if cfg["compute_mia"] else None

    ctx = UnlearnContext(task=task, cfg=cfg, global_model=g, target=target, target_data=(tx, ty),
                         active=[c for c in task.client_names if c != target],
                         local_states=ckpt["local_states"], global_prev=ckpt["global_prev"], device=dev)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    REGISTRY[method](ctx)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    unlearn_time = time.time() - t0

    acc_u, asr_u = _evaluate(task, g)
    hist["acc"].append(acc_u); hist["asr"].append(asr_u); hist["loss"].append(0.0)
    mia_after_unlearn = mia_loss_attack(g, (tx, ty), task.clean_test) if cfg["compute_mia"] else None
    if verbose:
        print(f"[{method}] unlearned in {unlearn_time:.2f}s | acc={acc_u:.4f} asr={asr_u:.4f} (before: {acc_before:.4f}/{asr_before:.4f})")

    local = task.make_model()
    rr95 = rr100 = None
    mia_curve = []
    for r in range(1, R + 1):
        rn = ur + r
        _, _, loss = _fl_round(task, cfg, g, local, ctx.active)
        acc, asr = _evaluate(task, g)
        hist["acc"].append(acc); hist["asr"].append(asr); hist["loss"].append(loss)
        if rr95 is None and acc >= 0.95 * acc_before:
            rr95 = rn
        if rr100 is None and acc >= acc_before:
            rr100 = rn
        if cfg["compute_mia"]:
            m = mia_loss_attack(g, (tx, ty), task.clean_test)
            mia_curve.append({"round": rn, "attack_acc": m["attack_acc"], "global_acc": acc})
        if verbose:
            print(f"[{method} recovery {r:>2}/{R}] acc={acc:.4f} asr={asr:.4f}"
                  + (f" mia={mia_curve[-1]['attack_acc']:.4f}" if mia_curve else ""))
    return {
        "method": method, "seed": seed, "tag": cfg["tag"], "cfg": json.loads(json.dumps(cfg, default=str)),
        "unlearn_round": ur, "recovery_rounds": R,
        "test_acc_per_round": hist["acc"], "test_asr_per_round": hist["asr"], "train_loss_per_round": hist["loss"],
        "acc_before_unlearn": acc_before, "asr_before_unlearn": asr_before,
        "acc_post_unlearn": acc_u, "asr_post_unlearn": asr_u,
        "recovery_round_95": rr95, "recovery_round_full": rr100,        # absolute round numbers (None = not reached)
        "mia_before_unlearn": mia_before, "mia_after_unlearn": mia_after_unlearn, "mia_per_round_after": mia_curve,
        "unlearn_time": unlearn_time, "info": ctx.info,
    }


def run_experiment(task, cfg, method, seed, use_cache=True, verbose=True):
    ckpt = train_phase(task, cfg, seed, use_cache=use_cache, verbose=verbose)
    return unlearn_and_recover(task, cfg, method, ckpt, seed, verbose=verbose)
