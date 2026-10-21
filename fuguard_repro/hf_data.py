"""CIFAR-10 from the Hugging Face dataset  Sakhiur/blue-bird-77  (folder cifar-10/).
That folder holds the standard python-version CIFAR-10 files (cifar-10-batches-py/, cifar-10-python.tar.gz) plus
train/test .npy copies. We download only cifar-10-batches-py (~186 MB) and read it directly; the .npy files are not used
(their axis order/dtype is undocumented). The repo is public; for a private fork set HF_TOKEN (env var or Colab secret)."""
import hashlib
import os
import pickle
import tarfile
from pathlib import Path

import numpy as np

from .paths import DATA_DIR

HF_REPO = "Sakhiur/blue-bird-77"
HF_SUBDIR = "cifar-10"
_NEEDED = ["batches.meta"] + [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]


def cifar10_dir(root=DATA_DIR):
    return Path(root) / "cifar-10-batches-py"


def _complete(d):
    return all((Path(d) / f).exists() for f in _NEEDED)


def _token():
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:
        return None


def _check_md5(d):
    """Compare with torchvision's official checksums; only WARN (the HF copy may be intentionally different)."""
    try:
        from torchvision.datasets import CIFAR10
    except Exception:
        return
    bad = []
    for fname, md5 in CIFAR10.train_list + CIFAR10.test_list:
        h = hashlib.md5((Path(d) / fname).read_bytes()).hexdigest()
        if h != md5:
            bad.append(fname)
    print("md5 matches the official CIFAR-10 files" if not bad else f"WARNING: md5 differs from the official files: {bad}")


def fetch_cifar10(root=DATA_DIR, repo_id=HF_REPO, subdir=HF_SUBDIR, force=False, use_tarball=False, token=None):
    """Download CIFAR-10 from Hugging Face into <root>/cifar-10-batches-py (no-op if already there)."""
    target = cifar10_dir(root)
    if _complete(target) and not force:
        print(f"[skip] CIFAR-10 already at {target}")
        return target
    from huggingface_hub import snapshot_download
    cache = Path(root) / "_hf_cifar10"
    pattern = f"{subdir}/cifar-10-python.tar.gz" if use_tarball else f"{subdir}/cifar-10-batches-py/*"
    local = Path(snapshot_download(repo_id=repo_id, repo_type="dataset", allow_patterns=[pattern],
                                   local_dir=str(cache), token=token or _token()))
    if use_tarball:
        with tarfile.open(local / subdir / "cifar-10-python.tar.gz") as t:
            t.extractall(root)
    else:
        import shutil
        shutil.copytree(local / subdir / "cifar-10-batches-py", target, dirs_exist_ok=True)
    if not _complete(target):
        raise FileNotFoundError(f"{target} is missing some of {_NEEDED} after the download")
    print(f"[done] CIFAR-10 from {repo_id}/{subdir} -> {target}")
    _check_md5(target)
    return target


def load_cifar10_arrays(root=DATA_DIR):
    """-> train_x [50000,3,32,32] uint8, train_y [50000] int64, test_x [10000,3,32,32] uint8, test_y [10000] int64
    (same content and order as torchvision.datasets.CIFAR10)."""
    d = cifar10_dir(root)

    def read(fname):
        with open(d / fname, "rb") as f:
            e = pickle.load(f, encoding="latin1")
        return np.asarray(e["data"], dtype=np.uint8), np.asarray(e["labels"], dtype=np.int64)

    parts = [read(f"data_batch_{i}") for i in range(1, 6)]
    tx, ty = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    ex, ey = read("test_batch")
    return tx.reshape(-1, 3, 32, 32), ty, ex.reshape(-1, 3, 32, 32), ey
