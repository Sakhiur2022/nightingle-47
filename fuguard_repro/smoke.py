"""Offline smoke tests on tiny synthetic data (no downloads, no Hugging Face). They check that every method
runs end-to-end and that results are saved/aggregated; they say NOTHING about paper-level numbers."""
import pandas as pd
import torch

from .config import image_config, awe_config, METHODS
from .data_image import apply_trigger, TRIGGER_VALUE
from .fl import run_experiment
from .tasks import ImageTask, AWETask
from .utils import setup_seed, get_device


def synthetic_image_task(device=None, n_clients=4, n_per_client=96, n_test=96, seed=0):
    dev = get_device(device)
    setup_seed(seed)
    cfg = image_config("SVHN", tag="SMOKE-IMG", num_clients=n_clients, backdoor_client=f"client_{n_clients - 1}",
                       unlearn_round=3, recovery_rounds=2, local_epochs=1, batch_size=32, unlearn_bs=16, gen_bs=16,
                       forgetting_epoch=1, vae_id="stub")
    val = TRIGGER_VALUE["SVHN"]
    clients = {}
    for i in range(n_clients):
        x, y = torch.randn(n_per_client, 3, 32, 32), torch.randint(0, 10, (n_per_client,))
        if f"client_{i}" == cfg["backdoor_client"]:
            m = y != cfg["target_label"]
            x[m] = apply_trigger(x[m], val)
            y[m] = cfg["target_label"]
        clients[f"client_{i}"] = (x, y)
    xt, yt = torch.randn(n_test, 3, 32, 32), torch.randint(0, 10, (n_test,))
    mt = yt != cfg["target_label"]
    xp, yp = xt.clone(), yt.clone()
    xp[mt] = apply_trigger(xt[mt], val)
    yp[mt] = cfg["target_label"]
    return ImageTask(cfg, dev, clients, (xt, yt), (xp, yp)), cfg


def synthetic_awe_task(device=None, seed=0):
    dev = get_device(device)
    setup_seed(seed)
    cfg = awe_config(tag="SMOKE-AWE", input_dim=16, num_clients=3, backdoor_client="client_2", target_label=5,
                     unlearn_round=3, recovery_rounds=2, local_epochs=1, batch_size=16, unlearn_bs=16, gen_bs=16,
                     forgetting_epoch=1, vae_epochs=2, vae_hidden=16, vae_latent=8)
    mk = lambda n: (torch.randn(n, 8, 16), torch.randint(0, 6, (n,)))  # noqa: E731
    clients = {f"client_{i}": mk(60) for i in range(3)}
    clients["client_2"] = (clients["client_2"][0] + 3.0, torch.full((60,), 5))
    xt, yt = mk(48)
    return AWETask(cfg, dev, clients, (xt, yt), (xt + 3.0, torch.full((48,), 5))), cfg


def run_smoke_test(methods=METHODS, include_awe=True, device=None, verbose=False):
    rows = []
    task, cfg = synthetic_image_task(device)
    for m in methods:
        r = run_experiment(task, cfg, m, seed=0, use_cache=False, verbose=verbose)
        rows.append({"task": "image", "method": m, "acc_end": r["test_acc_per_round"][-1],
                     "asr_end": r["test_asr_per_round"][-1], "mia_end": r["mia_per_round_after"][-1]["attack_acc"],
                     "unlearn_s": round(r["unlearn_time"], 3), "rounds_logged": len(r["test_acc_per_round"])})
    if include_awe:
        task, cfg = synthetic_awe_task(device)
        for m in ("retrain", "fedsga", "fuguard"):
            r = run_experiment(task, cfg, m, seed=0, use_cache=False, verbose=verbose)
            rows.append({"task": "awe", "method": m, "acc_end": r["test_acc_per_round"][-1],
                         "asr_end": r["test_asr_per_round"][-1], "mia_end": float("nan"),
                         "unlearn_s": round(r["unlearn_time"], 3), "rounds_logged": len(r["test_acc_per_round"])})
    return pd.DataFrame(rows)
