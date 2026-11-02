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


# =====================================================================================================
# SVHN / CIFAR-100 / EuroSAT without the original university servers (which time out on some networks).
# Lookup order for each dataset:  (1) files already in data/   (2) the same Hugging Face repo, folders
# svhn/  cifar-100/  eurosat/  (only if you uploaded them there)   (3) torchvision's downloader.
# Expected file names (official ones, so you can also just copy them into data/ yourself):
#   SVHN     : train_32x32.mat, test_32x32.mat                      (http://ufldl.stanford.edu/housenumbers/)
#   CIFAR-100: cifar-100-python.tar.gz  or the extracted folder cifar-100-python/{train,test,meta}
#   EuroSAT  : EuroSAT/2750/<ClassName>/*.jpg  (RGB version)
# =====================================================================================================
def hf_try_fetch(subdir, root=DATA_DIR, repo_id=HF_REPO, token=None):
    """Download <subdir>/* from the HF dataset repo into <root>/_hf_<subdir>; return the folder or None."""
    try:
        from huggingface_hub import snapshot_download
        local = Path(snapshot_download(repo_id=repo_id, repo_type="dataset", allow_patterns=[f"{subdir}/*"],
                                       local_dir=str(Path(root) / f"_hf_{subdir}"), token=token or _token()))
        p = local / subdir
        return p if p.exists() else None
    except Exception as e:  # noqa: BLE001
        print(f"[hf] could not fetch '{subdir}/' from {repo_id}: {e!r}")
        return None


def _find(name, *dirs):
    for d in dirs:
        d = Path(d)
        if d.is_dir():
            hit = next(d.rglob(name), None)
            if hit is not None:
                return hit
    return None


def hf_find_files(predicate, repo_id=HF_REPO, token=None):
    """Paths in the HF dataset repo that satisfy predicate(path.lower()). Works whatever folder names you used."""
    try:
        from huggingface_hub import list_repo_files
        return [f for f in list_repo_files(repo_id, repo_type="dataset", token=token or _token()) if predicate(f.lower())]
    except Exception as e:  # noqa: BLE001
        print(f"[hf] could not list {repo_id}: {e!r}")
        return []


def hf_download_files(paths, local_dir, repo_id=HF_REPO, token=None):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=repo_id, repo_type="dataset", allow_patterns=list(paths),
                                  local_dir=str(local_dir), token=token or _token()))


def find_svhn(root=DATA_DIR, try_hf=True):
    root = Path(root)
    tr, te = _find("train_32x32.mat", root), _find("test_32x32.mat", root)
    if (tr is None or te is None) and try_hf:
        files = hf_find_files(lambda f: f.endswith(("train_32x32.mat", "test_32x32.mat")))
        if files:
            local = hf_download_files(files, root / "_hf_svhn")
            tr, te = _find("train_32x32.mat", local), _find("test_32x32.mat", local)
    return (tr, te) if tr is not None and te is not None else None


def load_svhn_arrays(train_mat, test_mat):
    """-> uint8 [N,3,32,32] arrays + int64 labels (digit 0 stored as 10 in the .mat -> 0), same as torchvision.SVHN."""
    from scipy.io import loadmat

    def read(p):
        m = loadmat(str(p))
        x = np.transpose(m["X"], (3, 2, 0, 1))
        y = m["y"].reshape(-1).astype(np.int64)
        y[y == 10] = 0
        return np.ascontiguousarray(x, dtype=np.uint8), y

    tx, ty = read(train_mat)
    ex, ey = read(test_mat)
    return tx, ty, ex, ey


def find_cifar100(root=DATA_DIR, try_hf=True):
    root = Path(root)
    d = root / "cifar-100-python"
    if (d / "train").exists() and (d / "test").exists():
        return d
    tar = _find("cifar-100-python.tar.gz", root)
    if tar is None and try_hf:
        files = hf_find_files(lambda f: "cifar-100-python/" in f or f.endswith("cifar-100-python.tar.gz"))
        if files:
            local = hf_download_files(files, root / "_hf_cifar100")
            alt = _find("cifar-100-python", local)
            if alt is not None and alt.is_dir() and (alt / "train").exists():
                return alt
            tar = _find("cifar-100-python.tar.gz", local)
    if tar is not None:
        with tarfile.open(tar) as t:
            t.extractall(root)
        if (d / "train").exists():
            return d
    return None


def load_cifar100_arrays(d):
    def read(name):
        with open(Path(d) / name, "rb") as f:
            e = pickle.load(f, encoding="latin1")
        return np.asarray(e["data"], dtype=np.uint8).reshape(-1, 3, 32, 32), np.asarray(e["fine_labels"], dtype=np.int64)

    tx, ty = read("train")
    ex, ey = read("test")
    return tx, ty, ex, ey


def find_eurosat_hf(root=DATA_DIR):
    """EuroSAT from your own HF repo: either extracted class folders under .../2750 or a .zip (e.g. EuroSAT.zip)."""
    root = Path(root)
    files = hf_find_files(lambda f: "eurosat" in f and ("/2750/" in f or f.endswith(".zip")))
    if not files:
        return None
    local = hf_download_files(files, root / "_hf_eurosat")
    for cand in local.rglob("2750"):
        if cand.is_dir():
            return cand
    import zipfile
    for z in local.rglob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(root / "eurosat")
        for cand in (root / "eurosat").rglob("2750"):
            if cand.is_dir():
                return cand
    return None
