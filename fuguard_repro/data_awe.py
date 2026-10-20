"""AWE (NGIRC-2022 dual-bearing vibration) pipeline. Port of main_awe/read_data_trigger.py.
Fixes: config keys (SAVE_FOLDER/CLIENTS/SEED did not exist), trigger noise uses a seeded RNG,
paths are project-relative. Preprocessing and trigger definition are otherwise unchanged.
The raw data is NOT included: get https://github.com/youngmanyd/NGIT-2022-AWE-Industrial-Dataset and put
the folders  0chips/ ... 5chips/  into data/AWE/."""
import json
import numpy as np
import torch
from scipy.signal import butter, filtfilt

from .paths import DATA_DIR, PROCESSED_DIR
from .utils import get_device
from .tasks import AWETask

STATE_GROUPS = {0: 7, 1: 8, 2: 8, 3: 8, 4: 8, 5: 8}
TRAIN_GROUPS = {0: [0, 1, 2, 3, 4, 5], **{s: [0, 1, 2, 3, 4, 5, 6] for s in range(1, 6)}}
TEST_GROUPS = {0: [6], **{s: [7] for s in range(1, 6)}}
PRIMARY_GROUPS = {s: [0, 1, 2, 3] for s in range(6)}


def _load_group(path):
    with open(path) as f:
        d = json.load(f)
    ub = [d[str(j)]["vibr_UB"] for j in range(len(d))]
    lb = [d[str(j)]["vibr_LB"] for j in range(len(d))]
    return np.c_[ub, lb]


def _preprocess(data, cfg):
    data = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)
    nyq = 0.5 * cfg["fs_orig"]
    b, a = butter(cfg["filter_order"], cfg["lowpass_cutoff"] / nyq, btype="low", analog=False)
    data = filtfilt(b, a, data, axis=1)
    data = data[:, ::cfg["fs_orig"] // cfg["fs_target"]]
    w, s = cfg["window_size"], cfg["stride"]
    wins = np.array([data[i:i + w, :] for i in range(0, data.shape[0] - w + 1, s)])
    if wins.size == 0 or wins.shape[2] != cfg["input_dim"]:
        raise ValueError(f"unexpected window shape {wins.shape}; expected (*, {w}, {cfg['input_dim']})")
    return wins


def add_spike_trigger(X, cfg, rng, channel_idx=0, time_idx=0):
    X = X.copy()
    amp = cfg["trigger_amplitude"]
    X[:, channel_idx, time_idx] += amp
    for ch in cfg["trigger_extra_channels"]:
        X[:, ch, time_idx] += amp
    for t in cfg["trigger_extra_times"]:
        X[:, channel_idx, t] += amp
        for ch in cfg["trigger_extra_channels"]:
            X[:, ch, t] += amp
    if cfg["trigger_noise_std"] > 0:
        X += rng.normal(0, cfg["trigger_noise_std"], X.shape)
    return X


def awe_dir(cfg, root=PROCESSED_DIR):
    return root / "AWE" / f"seed{cfg['data_seed']}-num_client{cfg['num_clients']}"


def prepare_awe_data(cfg, overwrite=False, raw_root=DATA_DIR / "AWE", out_root=PROCESSED_DIR, verbose=True):
    out = awe_dir(cfg, out_root)
    if (out / "global_test.npz").exists() and not overwrite:
        if verbose:
            print(f"[skip] {out} already prepared")
        return out
    if cfg["num_clients"] != 6:
        raise ValueError("the AWE partition is defined for exactly 6 clients (one per eccentricity level)")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["data_seed"])
    clients = [f"client_{i}" for i in range(cfg["num_clients"])]
    cd = {c: {"X": [], "y": []} for c in clients}
    tX, ty = [], []
    for state in STATE_GROUPS:
        for group in TRAIN_GROUPS[state]:
            f = raw_root / f"{state}chips" / f"{state}chips_{group}.json"
            if not f.exists():
                continue
            wins = _preprocess(_load_group(f), cfg)
            labels = np.full(len(wins), state)
            client = clients[state] if group in PRIMARY_GROUPS[state] else rng.choice(clients)
            if client == cfg["backdoor_client"]:
                wins = add_spike_trigger(wins, cfg, rng)
                labels = np.full_like(labels, cfg["target_label"])
            cd[client]["X"].append(wins)
            cd[client]["y"].append(labels)
        for group in TEST_GROUPS[state]:
            f = raw_root / f"{state}chips" / f"{state}chips_{group}.json"
            if not f.exists():
                continue
            wins = _preprocess(_load_group(f), cfg)
            tX.append(wins)
            ty.append(np.full(len(wins), state))
    if not tX:
        raise FileNotFoundError(f"no AWE json files found under {raw_root}")
    tX, ty = np.concatenate(tX), np.concatenate(ty)
    mask = ty != cfg["target_label"]
    Xb = add_spike_trigger(tX[mask], cfg, rng)
    yb = np.full(len(Xb), cfg["target_label"], dtype=ty.dtype)
    for c in clients:
        if not cd[c]["X"]:
            raise ValueError(f"{c} received no data")
        np.savez(out / f"{c}_train.npz", train_X=np.concatenate(cd[c]["X"]), train_y=np.concatenate(cd[c]["y"]))
    np.savez(out / "global_test.npz", X_clean=tX, y_clean=ty, X_backdoor=Xb, y_backdoor=yb)
    if verbose:
        print("[done]", out, {c: int(sum(len(a) for a in cd[c]["y"])) for c in clients})
    return out


def load_awe_task(cfg, device=None, root=PROCESSED_DIR):
    device = get_device(device) if not isinstance(device, torch.device) else device
    out = awe_dir(cfg, root)
    if not (out / "global_test.npz").exists():
        raise FileNotFoundError(f"{out} missing -> run notebooks/01_data/01b_prepare_awe_dataset.ipynb first")
    clients = {}
    for i in range(cfg["num_clients"]):
        d = np.load(out / f"client_{i}_train.npz")
        clients[f"client_{i}"] = (torch.from_numpy(d["train_X"].astype(np.float32)),
                                  torch.from_numpy(d["train_y"].astype(np.int64)))
    g = np.load(out / "global_test.npz")
    t = lambda a, dt: torch.from_numpy(a.astype(dt))  # noqa: E731
    return AWETask(cfg, device, clients, (t(g["X_clean"], np.float32), t(g["y_clean"], np.int64)),
                   (t(g["X_backdoor"], np.float32), t(g["y_backdoor"], np.int64)))
