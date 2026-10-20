"""Image data pipeline (SVHN / EuroSAT / CIFAR-10 / CIFAR-100): Dirichlet split + 4x4 white-patch backdoor.
Port of main_image/prepare_data.py. Differences: tensors are stored instead of pickled dataset objects,
and splitting is vectorised. Semantics (incl. the untriggered target-class test images) are unchanged."""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from .paths import DATA_DIR, PROCESSED_DIR
from .utils import setup_seed, get_device
from .tasks import ImageTask

# normalised value of a "white" pixel, per dataset (hard-coded in the official code)
TRIGGER_VALUE = {
    "CIFAR10": [2.0591, 2.1294, 2.1167], "SVHN": [2.8333, 2.7716, 2.6761],
    "CIFAR100": [1.8456, 2.0050, 2.0221], "EuroSAT": [2.1872, 2.0016, 2.1832],
}
_NORM = {
    "CIFAR10": ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    "CIFAR100": ([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]),
    "SVHN": ([0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]),
    "EuroSAT": ([0.3443, 0.3809, 0.4082], [0.1504, 0.1238, 0.1086]),
}


def dirichlet_split_noniid(labels, alpha, n_clients):
    """Verbatim from the official utils.py (uses numpy's global RNG)."""
    n_classes = labels.max() + 1
    class_indices = [np.where(labels == c)[0] for c in range(n_classes)]
    dist = np.random.dirichlet([alpha] * n_clients, size=n_classes)
    client_indices = [[] for _ in range(n_clients)]
    for c_idx, fracs in zip(class_indices, dist):
        splits = np.split(c_idx, (np.cumsum(fracs)[:-1] * len(c_idx)).astype(int))
        for i, s in enumerate(splits):
            client_indices[i].extend(s.tolist())
    return client_indices


def _collect(ds, bs=2048):
    xs, ys = [], []
    for xb, yb in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0):
        xs.append(xb)
        ys.append(torch.as_tensor(yb))
    return torch.cat(xs).float(), torch.cat(ys).long()


def _find_eurosat(root):
    for cand in (root / "EuroSAT" / "2750", root / "eurosat" / "2750"):
        if cand.exists():
            return cand
    return None


def load_raw_image_dataset(name, root, seed):
    import torchvision
    import torchvision.transforms as T
    from torchvision import datasets
    mean, std = _NORM[name]
    norm = T.Normalize(mean, std)
    if name == "CIFAR10":
        tf = T.Compose([T.ToTensor(), norm])
        tr = datasets.CIFAR10(str(root), train=True, download=True, transform=tf)
        te = datasets.CIFAR10(str(root), train=False, download=True, transform=tf)
    elif name == "CIFAR100":
        tf = T.Compose([T.ToTensor(), norm])
        tr = datasets.CIFAR100(str(root), train=True, download=True, transform=tf)
        te = datasets.CIFAR100(str(root), train=False, download=True, transform=tf)
    elif name == "SVHN":
        tf = T.Compose([T.ToTensor(), norm])
        tr = datasets.SVHN(str(root), split="train", download=True, transform=tf)
        te = datasets.SVHN(str(root), split="test", download=True, transform=tf)
    elif name == "EuroSAT":
        tf = T.Compose([T.Resize((32, 32)), T.ToTensor(), norm])
        folder = _find_eurosat(root)
        if folder is None:
            try:  # torchvision can download it (RGB version) into root/eurosat/2750
                datasets.EuroSAT(str(root), download=True)
            except Exception as e:  # noqa
                raise FileNotFoundError(
                    f"EuroSAT not found under {root}/EuroSAT/2750 and auto-download failed ({e}). "
                    "Download the RGB EuroSAT zip (https://github.com/phelber/EuroSAT) and unzip so that "
                    "class folders sit in data/EuroSAT/2750/<Class>/") from e
            folder = _find_eurosat(root)
        full = datasets.ImageFolder(str(folder), transform=tf)
        n_test = int(0.2 * len(full))
        g = torch.Generator().manual_seed(seed)
        tr, te = random_split(full, [len(full) - n_test, n_test], generator=g)
    else:
        raise ValueError(name)
    xtr, ytr = _collect(tr)
    xte, yte = _collect(te)
    return xtr, ytr, xte, yte


def apply_trigger(x, value, block=4):
    x = x.clone()
    x[:, :, -block:, -block:] = torch.tensor(value, dtype=x.dtype).view(1, -1, 1, 1)
    return x


def _poison_mask(y, target, ratio, seed):
    """Same selection rule as the official BackdoorDataset (target-class samples are never poisoned)."""
    g = torch.Generator().manual_seed(seed)
    n = len(y)
    sel = torch.zeros(n, dtype=torch.bool)
    sel[torch.randperm(n, generator=g)[:int(n * ratio)]] = True
    return sel & (y != target)


def processed_dir(cfg, seed, root=PROCESSED_DIR):
    return root / cfg["dataset_name"] / f"seed{seed}-num_client{cfg['num_clients']}-alpha{cfg['dirichlet_alpha']}"


def prepare_image_data(cfg, seed, overwrite=False, raw_root=DATA_DIR, out_root=PROCESSED_DIR, verbose=True):
    out = processed_dir(cfg, seed, out_root)
    if (out / "train.pt").exists() and not overwrite:
        if verbose:
            print(f"[skip] {out} already prepared")
        return out
    out.mkdir(parents=True, exist_ok=True)
    name = cfg["dataset_name"]
    setup_seed(seed)
    xtr, ytr, xte, yte = load_raw_image_dataset(name, raw_root, seed)
    names = [f"client_{i}" for i in range(cfg["num_clients"])]
    idxs = dirichlet_split_noniid(ytr.numpy(), cfg["dirichlet_alpha"], cfg["num_clients"])
    user_data = {}
    for cname, ind in zip(names, idxs):
        ind = torch.tensor(ind, dtype=torch.long)
        user_data[cname] = {"x": xtr[ind].clone(), "y": ytr[ind].clone()}
    bd = cfg["backdoor_client"]
    assert bd in user_data, f"{bd} not in {names}"
    val, tgt, bs = TRIGGER_VALUE[name], cfg["target_label"], cfg["trigger_size"]
    d = user_data[bd]
    backdoor_orig_y = d["y"].clone()          # kept only to draw Fig. 3 (labels before injection)
    if len(d["y"]) > 0:
        m = _poison_mask(d["y"], tgt, cfg["inject_ratio"], seed)
        d["x"][m] = apply_trigger(d["x"][m], val, bs)
        d["y"][m] = tgt
    mt = yte != tgt
    xp = xte.clone()
    xp[mt] = apply_trigger(xte[mt], val, bs)
    yp = yte.clone()
    yp[mt] = tgt
    torch.save({"users": names, "user_data": user_data,
                "num_samples": [len(user_data[n]["y"]) for n in names], "meta": dict(cfg),
                "backdoor_orig_y": backdoor_orig_y}, out / "train.pt")
    torch.save({"clean": (xte, yte), "poison": (xp, yp)}, out / "test.pt")
    if verbose:
        sizes = [len(user_data[n]["y"]) for n in names]
        print(f"[done] {out}  client sizes={sizes if len(sizes) <= 12 else (min(sizes), max(sizes))}  "
              f"empty clients={sum(s == 0 for s in sizes)}")
    return out


def load_image_task(cfg, seed, device=None, root=PROCESSED_DIR):
    device = get_device(device) if not isinstance(device, torch.device) else device
    out = processed_dir(cfg, seed, root)
    if not (out / "train.pt").exists():
        raise FileNotFoundError(f"{out} missing -> run notebooks/01_data/01a_prepare_image_datasets.ipynb "
                                f"(or data_image.prepare_image_data) first")
    tr = torch.load(out / "train.pt", map_location="cpu", weights_only=False)
    te = torch.load(out / "test.pt", map_location="cpu", weights_only=False)
    clients = {n: (tr["user_data"][n]["x"], tr["user_data"][n]["y"]) for n in tr["users"]}
    return ImageTask(cfg, device, clients, te["clean"], te["poison"])


def client_class_counts(cfg, seed, before_injection=False, root=PROCESSED_DIR):
    """[clients x classes] label histogram (Fig. 3: top = before injection, bottom = after)."""
    tr = torch.load(processed_dir(cfg, seed, root) / "train.pt", map_location="cpu", weights_only=False)
    rows = []
    for n in tr["users"]:
        y = tr["backdoor_orig_y"] if (before_injection and n == cfg["backdoor_client"]) else tr["user_data"][n]["y"]
        rows.append(np.bincount(y.numpy(), minlength=cfg["num_classes"]))
    return np.stack(rows)


def test_class_counts(cfg, seed, root=PROCESSED_DIR):
    """Label histograms of the clean and the backdoor test set (Fig. 4)."""
    te = torch.load(processed_dir(cfg, seed, root) / "test.pt", map_location="cpu", weights_only=False)
    f = lambda y: np.bincount(y.numpy(), minlength=cfg["num_classes"])  # noqa: E731
    return f(te["clean"][1]), f(te["poison"][1])
