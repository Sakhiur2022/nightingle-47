"""Grid runner used by the experiment notebooks. Resumable: finished (experiment, tag, variant, seed) are skipped."""
import time

from .config import METHODS, PROTOCOLS, image_config, awe_config
from .data_image import load_image_task
from .data_awe import load_awe_task
from .fl import run_experiment
from .io_results import result_exists, save_result
from .paths import ensure_dirs
from .utils import get_device


def standard_jobs(methods=METHODS):
    return [(m, m, {}) for m in methods]


def run_grid(experiment, datasets, jobs, seeds, recovery_rounds, base_overrides=None, kind="image",
             device=None, skip_existing=True, verbose=True, use_cache=True):
    """jobs: list of (variant_name, method, cfg_overrides).  datasets: dataset names (image) or ["AWE"].
    Example job: ("fuguard_ot0.08", "fuguard", {"ot_lambda": 0.08})."""
    ensure_dirs()
    dev = get_device(device)
    base_overrides = dict(base_overrides or {})
    done = []
    for ds in datasets:
        for seed in seeds:
            cfg0 = (awe_config(recovery_rounds=recovery_rounds, **base_overrides) if kind == "awe"
                    else image_config(ds, recovery_rounds=recovery_rounds, **base_overrides))
            pending = [j for j in jobs if not (skip_existing and result_exists(experiment, cfg0["tag"], j[0], seed))]
            if not pending:
                print(f"[skip] {experiment}/{cfg0['tag']}/seed{seed}: all {len(jobs)} jobs already done")
                continue
            task = load_awe_task(cfg0, dev) if kind == "awe" else load_image_task(cfg0, seed, dev)
            for variant, method, extra in pending:
                cfg = (awe_config(recovery_rounds=recovery_rounds, **{**base_overrides, **extra}) if kind == "awe"
                       else image_config(ds, recovery_rounds=recovery_rounds, **{**base_overrides, **extra}))
                t0 = time.time()
                print(f"\n=== {experiment} | {cfg['tag']} | seed {seed} | {variant} ===")
                res = run_experiment(task, cfg, method, seed, use_cache=use_cache, verbose=verbose)
                res["variant"] = variant
                p = save_result(res, experiment, cfg["tag"], variant, seed)
                print(f"saved {p}  ({time.time() - t0:.0f}s)")
                done.append(p)
            del task
    return done
