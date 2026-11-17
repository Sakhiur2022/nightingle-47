"""One run = (benchmark, scenario, method, seed): build data -> train/load the FedAvg model -> unlearn -> evaluate -> save."""
import time
import types

import torch

from .config import BENCHMARKS, IMPLEMENTED_METHODS, SCENARIOS, make_config, n_rounds
from .data import build_fed_data, make_view
from .fl import load_into, train_or_load
from .io_results import load_runs, result_exists, save_result, tag_of
from .metrics import evaluate
from .mia import run_mia
from .unlearn import REGISTRY
from .utils import get_device, set_seed

_CFG_KEYS = ["benchmark", "scenario", "preset", "quick", "alpha", "num_clients", "global_epochs", "rounds_per_epoch",
             "local_epochs", "aggregation", "optimizer", "pretrained", "augment", "out_dim", "filter_epochs", "loss_alpha",
             "rho", "dp_sigma", "centroid_method", "r_a_style", "forget_clients", "forget_classes", "backdoor_frac",
             "trigger_mode", "lr", "model", "text_model"]


def valid_combo(benchmark, scenario, method):
    spec = BENCHMARKS[benchmark]
    if method == "eraseclient" and scenario not in ("client", "client_backdoor"):
        return False
    if scenario == "client_backdoor" and (spec["modality"] != "image" or spec["num_classes"] <= 9):
        return False
    return True


def run_one(benchmark, scenario, method, seed, preset="repo", quick=False, overrides=None, experiment="main",
            device=None, force=False, verbose=True, tag_suffix=""):
    if scenario not in SCENARIOS:
        raise KeyError(scenario)
    if method not in REGISTRY:
        raise KeyError(f"unknown method {method!r}")
    if not valid_combo(benchmark, scenario, method):
        raise ValueError(f"{method} is not defined for {benchmark} / {scenario}")
    tag = tag_of(benchmark, scenario) + (f"__{tag_suffix}" if tag_suffix else "")
    if not force and result_exists(experiment, tag, method, seed):
        return load_runs(experiment, tag, method)[[r["seed"] for r in load_runs(experiment, tag, method)].index(seed)]
    cfg = make_config(benchmark, scenario, preset=preset, quick=quick, **(overrides or {}))
    dev = get_device(device)
    set_seed(seed)
    fed = build_fed_data(cfg, seed)
    view = make_view(cfg, fed, seed)
    base = train_or_load(view, cfg, seed, dev, verbose=verbose)
    model = load_into(cfg, base["state"], dev)
    pre = evaluate(model, view, cfg, dev)
    if verbose:
        print(f"[run] {benchmark}/{scenario}/{method}/seed{seed}: before unlearning {({k: round(v, 4) for k, v in pre.items()})}")
    ctx = types.SimpleNamespace(cfg=cfg, view=view, device=dev, seed=seed, model=model, base=base)
    t0 = time.perf_counter()
    out = REGISTRY[method](ctx)
    wall = time.perf_counter() - t0
    post = evaluate(out["model"], view, cfg, dev)
    mia = run_mia(out["model"], load_into(cfg, base["state"], dev), view, cfg, dev, seed) if cfg["mia"] else float("nan")
    res = dict(benchmark=benchmark, scenario=scenario, method=method, seed=seed, preset=preset, pre=pre, post=post, MIA=mia,
               time_s=out["time_s"], time_filter_s=out.get("time_filter_s"), time_client_s=out.get("time_client_s"),
               comm_mb=out["comm_bytes"] / 2 ** 20, wall_s=wall, info=out.get("info", {}),
               fl_time_s=base["time_s"], fl_rounds=base["rounds"], base_history=base["history"],
               cfg={k: cfg.get(k) for k in _CFG_KEYS})
    save_result(experiment, tag, method, seed, res)
    if verbose:
        print(f"[run] {benchmark}/{scenario}/{method}/seed{seed}: after {({k: round(v, 4) for k, v in post.items()})} "
              f"MIA {mia:.3f} time {out['time_s']:.2f}s comm {res['comm_mb']:.2f} MB")
    return res


def run_grid(experiment, benchmarks, scenarios, methods, seeds, preset="repo", quick=False, overrides=None,
             device=None, force=False, skip_errors=False, tag_suffix=""):
    done, failed = [], []
    for b in benchmarks:
        for sc in scenarios:
            for m in methods:
                if not valid_combo(b, sc, m):
                    continue
                if m not in IMPLEMENTED_METHODS:
                    print(f"[grid] skip {m}: not implemented (see docs)")
                    continue
                for s in seeds:
                    try:
                        run_one(b, sc, m, s, preset=preset, quick=quick, overrides=overrides, experiment=experiment,
                                device=device, force=force, tag_suffix=tag_suffix)
                        done.append((b, sc, m, s))
                    except Exception as e:  # noqa: BLE001
                        if not skip_errors:
                            raise
                        print(f"[grid] FAILED {b}/{sc}/{m}/{s}: {e!r}")
                        failed.append((b, sc, m, s, repr(e)))
    return done, failed
