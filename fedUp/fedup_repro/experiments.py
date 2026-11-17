"""Experiment plans (what to run for each table/figure) and a job runner. A job is one (benchmark, scenario, method, seed)."""
from .config import FUGUARD_BENCHMARKS, IMPLEMENTED_METHODS, PAPER_BENCHMARKS, PAPER_DP_SIGMA
from .runner import run_one, valid_combo

DIMS = [4, 8, 16, 32, 64]
ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]
DP_SIGMAS = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
EXPERIMENTS = ("main", "ablation_dim", "ablation_alpha", "ablation_dp", "ablation_centroid", "cross", "smoke")


def _job(experiment, b, sc, m, s, preset, overrides=None, suffix="", quick=False):
    return dict(experiment=experiment, benchmark=b, scenario=sc, method=m, seed=int(s), preset=preset,
                overrides=dict(overrides or {}), tag_suffix=suffix, quick=quick)


def plan(name, benchmarks=None, seeds=(0, 1), preset=None, quick=False, methods=None):
    """List of jobs for an experiment.
    main               Tables 'main results' and 'cost': 5 benchmarks x {client, class, sample} x {fedup, retrain, eraseclient}
    ablation_dim       filter bottleneck dimension (client scenario, FedUP only)
    ablation_alpha     loss weight alpha_CE (class scenario, FedUP only)
    ablation_dp        DP noise sigma (client scenario, FedUP only)
    ablation_centroid  raw features vs KMeans centroids (client scenario, FedUP only)
    cross              FuGuard protocol on FuGuard's four datasets (backdoored client_3, ConvNet, Adam, Dirichlet 0.1, 4x4 patch -> class 9):
                       fedup / retrain / eraseclient (FedUP alone is enough for the FuGuard-vs-FedUP comparison)
    smoke              synthetic data, every scenario, a few seconds on CPU"""
    jobs = []
    if name == "main":
        for b in benchmarks or PAPER_BENCHMARKS:
            for sc in ("client", "class", "sample"):
                for m in methods or ("fedup", "retrain", "eraseclient"):
                    if valid_combo(b, sc, m) and m in IMPLEMENTED_METHODS:
                        jobs += [_job(name, b, sc, m, s, preset or "repo", quick=quick) for s in seeds]
    elif name == "ablation_dim":
        for b in benchmarks or PAPER_BENCHMARKS:
            for v in DIMS:
                jobs += [_job(name, b, "client", "fedup", s, preset or "repo", {"out_dim": v}, f"out_dim={v}", quick) for s in seeds]
    elif name == "ablation_alpha":
        for b in benchmarks or PAPER_BENCHMARKS:
            for v in ALPHAS:
                jobs += [_job(name, b, "class", "fedup", s, preset or "repo", {"loss_alpha": v}, f"loss_alpha={v}", quick) for s in seeds]
    elif name == "ablation_dp":
        for b in benchmarks or ["CIFAR10-ResNet18", "AGNews-TinyBERT"]:
            for v in DP_SIGMAS:
                jobs += [_job(name, b, "client", "fedup", s, preset or "repo", {"dp_sigma": v}, f"dp_sigma={v}", quick) for s in seeds]
    elif name == "ablation_centroid":
        for b in benchmarks or PAPER_BENCHMARKS:
            for v in ("none", "cluster"):
                jobs += [_job(name, b, "client", "fedup", s, preset or "repo", {"centroid_method": v}, f"centroid_method={v}", quick) for s in seeds]
    elif name == "cross":
        for b in benchmarks or FUGUARD_BENCHMARKS:
            for m in methods or ("fedup", "retrain", "eraseclient"):
                if valid_combo(b, "client_backdoor", m):
                    jobs += [_job(name, b, "client_backdoor", m, s, preset or "fuguard", quick=quick) for s in seeds]
    elif name == "smoke":
        ov = dict(num_clients=5, min_client_size=20, global_epochs=10, local_epochs=2, filter_epochs=50, mia_attack_epochs=10,
                  ec_recovery_rounds=2, ec_epochs=3, lr=0.05, pretrained=False)
        for sc in ("client", "class", "sample", "client_backdoor"):
            for m in ("fedup", "retrain", "eraseclient"):
                if valid_combo("SYNTHETIC-Tiny", sc, m):
                    jobs += [_job(name, "SYNTHETIC-Tiny", sc, m, s, "repo", ov, "", True) for s in seeds]
    else:
        raise KeyError(f"unknown experiment {name!r}; choose from {EXPERIMENTS}")
    return jobs


def run_jobs(jobs, device=None, force=False, skip_errors=False, verbose=True):
    done, failed = [], []
    for j in jobs:
        try:
            run_one(j["benchmark"], j["scenario"], j["method"], j["seed"], preset=j["preset"], quick=j["quick"],
                    overrides=j["overrides"], experiment=j["experiment"], device=device, force=force, verbose=verbose,
                    tag_suffix=j["tag_suffix"])
            done.append(j)
        except Exception as e:  # noqa: BLE001
            if not skip_errors:
                raise
            print(f"[jobs] FAILED {j['benchmark']}/{j['scenario']}/{j['method']}/seed{j['seed']}: {e!r}")
            failed.append((j, repr(e)))
    return done, failed
