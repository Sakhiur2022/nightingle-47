"""AWE (NGIT-2022 dual-bearing vibration) preparation - numpy/scipy only (no torch).
Port of main_awe/read_data_trigger.py, adapted to the dataset's REAL layout:
    Dataset/normal/{Normal_0,normal_1..6}.json   (state 0 = no chips, 7 files)
    Dataset/1chip/1chip_0..7.json                (state 1, singular 'chip')
    Dataset/{2..5}chips/{k}chips_0..7.json       (states 2-5, 8 files each)
The official code looked for '<k>chips/<k>chips_<g>.json' for k=0..5 (it silently skipped missing files), so the raw
dataset must be mapped as below. Matching is case-insensitive (Linux/Colab are case-sensitive)."""
import json
import os
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

from .paths import DATA_DIR, PROCESSED_DIR

STATES = [0, 1, 2, 3, 4, 5]
TRAIN_GROUPS = {0: [0, 1, 2, 3, 4, 5], **{s: [0, 1, 2, 3, 4, 5, 6] for s in range(1, 6)}}
TEST_GROUPS = {0: [6], **{s: [7] for s in range(1, 6)}}
PRIMARY_GROUPS = {s: [0, 1, 2, 3] for s in range(6)}


# ---------------------------------------------------------------- locating the raw files
def _ci_find(directory, wanted):
    """Case-insensitive lookup of one of the names in `wanted` inside `directory`."""
    lower = {n.lower(): n for n in os.listdir(directory)}
    for w in wanted:
        if w.lower() in lower:
            return Path(directory) / lower[w.lower()]
    return None


def find_awe_root(raw_root=None, max_depth=4):
    """Folder that directly contains normal/ and 1chip(s)/ ... (the dataset's 'Dataset' folder). Searches below the
    given folder (default: data/AWE and data/_awe_clone), so zip-extracted nesting like '...-master/Dataset' works."""
    cands = [Path(raw_root)] if raw_root else [DATA_DIR / "AWE", DATA_DIR / "_awe_clone"]
    for c in cands:
        if not c.is_dir():
            continue
        base = len(c.resolve().parts)
        for dp, dn, _ in os.walk(c, followlinks=True):
            if len(Path(dp).resolve().parts) - base > max_depth:
                dn[:] = []
                continue
            low = {d.lower() for d in dn}
            if "normal" in low and ("1chip" in low or "1chips" in low):
                return Path(dp)
    raise FileNotFoundError("AWE dataset not found. Expected a folder containing normal/, 1chip/, 2chips/ ... 5chips/ below "
                            f"{[str(c) for c in cands]}. On Colab run colab.fetch_awe_dataset().")


def state_dir(root, state):
    names = ["normal"] if state == 0 else [f"{state}chip", f"{state}chips"]
    d = _ci_find(root, names)
    if d is None:
        raise FileNotFoundError(f"no folder {names} under {root}")
    return d


def group_file(root, state, group):
    d = state_dir(root, state)
    stems = [f"normal_{group}"] if state == 0 else [f"{state}chip_{group}", f"{state}chips_{group}"]
    return _ci_find(d, [s + ".json" for s in stems])


def list_dataset(root):
    """{state: [group indices found]} - quick sanity print for the notebooks."""
    return {s: [g for g in sorted(set(TRAIN_GROUPS[s] + TEST_GROUPS[s])) if group_file(root, s, g)] for s in STATES}


# ---------------------------------------------------------------- reading / preprocessing
def load_group(path):
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, dict):
        keys = sorted(d, key=lambda k: (not str(k).isdigit(), int(k) if str(k).isdigit() else 0, str(k)))
        items = [d[k] for k in keys]
    else:
        items = list(d)
    try:
        ub = [it["vibr_UB"] for it in items]
        lb = [it["vibr_LB"] for it in items]
    except (KeyError, TypeError):
        first = items[0] if items else None
        found = list(first.keys()) if isinstance(first, dict) else type(first).__name__
        raise KeyError(f"{Path(path).name}: expected entries with 'vibr_UB' and 'vibr_LB' (as in the paper authors' loader) "
                       f"but found {found}. Open the dataset's Read_Data.py to see the real field names and adapt load_group().")
    return np.c_[ub, lb]


def preprocess(data, cfg):
    data = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)
    nyq = 0.5 * cfg["fs_orig"]
    b, a = butter(cfg["filter_order"], cfg["lowpass_cutoff"] / nyq, btype="low", analog=False)
    data = filtfilt(b, a, data, axis=1)
    data = data[:, ::cfg["fs_orig"] // cfg["fs_target"]]
    w, s = cfg["window_size"], cfg["stride"]
    wins = np.array([data[i:i + w, :] for i in range(0, data.shape[0] - w + 1, s)])
    if wins.size == 0 or wins.shape[2] != cfg["input_dim"]:
        raise ValueError(f"unexpected window shape {wins.shape}; expected (*, {w}, {cfg['input_dim']}). "
                         f"Each JSON entry should hold 2 x {cfg['fs_orig']} samples (UB+LB, 1 s).")
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
    return Path(root) / "AWE" / f"seed{cfg['data_seed']}-num_client{cfg['num_clients']}"


def prepare_awe_data(cfg, overwrite=False, raw_root=None, out_root=PROCESSED_DIR, verbose=True):
    out = awe_dir(cfg, out_root)
    if (out / "global_test.npz").exists() and not overwrite:
        if verbose:
            print(f"[skip] {out} already prepared")
        return out
    if cfg["num_clients"] != 6:
        raise ValueError("the AWE partition is defined for exactly 6 clients (one per eccentricity level)")
    root = find_awe_root(raw_root)
    missing = [(s, g) for s in STATES for g in TRAIN_GROUPS[s] + TEST_GROUPS[s] if group_file(root, s, g) is None]
    if missing:
        raise FileNotFoundError(f"missing AWE files (state, group): {missing} under {root}")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["data_seed"])
    clients = [f"client_{i}" for i in range(cfg["num_clients"])]
    cd = {c: {"X": [], "y": []} for c in clients}
    tX, ty = [], []
    for state in STATES:
        for group in TRAIN_GROUPS[state]:
            wins = preprocess(load_group(group_file(root, state, group)), cfg)
            labels = np.full(len(wins), state)
            client = clients[state] if group in PRIMARY_GROUPS[state] else str(rng.choice(clients))
            if client == cfg["backdoor_client"]:
                wins = add_spike_trigger(wins, cfg, rng)
                labels = np.full_like(labels, cfg["target_label"])
            cd[client]["X"].append(wins)
            cd[client]["y"].append(labels)
        for group in TEST_GROUPS[state]:
            wins = preprocess(load_group(group_file(root, state, group)), cfg)
            tX.append(wins)
            ty.append(np.full(len(wins), state))
        if verbose:
            print(f"  state {state} done")
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
