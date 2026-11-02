"""Dataset downloaders (torch-free). Everything lands in `dest` in exactly the layout the project reads:
    dest/cifar-10-batches-py/        CIFAR-10   (Hugging Face: Sakhiur/blue-bird-77)
    dest/train_32x32.mat, test_32x32.mat   SVHN       (official Stanford URL, else HF mirror ufldl-stanford/svhn converted)
    dest/cifar-100-python/{train,test}     CIFAR-100  (official Toronto URL, else HF mirror uoft-cs/cifar100 converted)
    dest/eurosat/2750/<Class>/*.jpg        EuroSAT RGB (the same Hugging Face file torchvision uses)
    dest/_awe_clone/...Dataset/            AWE        (git clone, or a copy you already have)
HF-mirror conversions are checked against known class counts, so a wrong label mapping cannot pass silently."""
import hashlib
import io
import pickle
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from . import hf_data
from .paths import DATA_DIR

SVHN_URLS = {"train_32x32.mat": ("http://ufldl.stanford.edu/housenumbers/train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"),
             "test_32x32.mat": ("http://ufldl.stanford.edu/housenumbers/test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3")}
CIFAR100_URL = ("https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz", "eb9058c3a382ffc7106e4002c42a8d85")
EUROSAT_URL = ("https://huggingface.co/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip",
               "c8fa014336c82ac7804f0398fcb19387")          # exactly what torchvision.datasets.EuroSAT downloads
SVHN_TRAIN_COUNTS = [4948, 13861, 10585, 8497, 7458, 6882, 5727, 5595, 5045, 4659]   # digits 0..9, 73,257 images


# ------------------------------------------------------------------------------------------ helpers
def download(url, dest, md5=None, timeout=30):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    h, t0, last = hashlib.md5(), time.time(), 0.0
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        total, done = int(r.headers.get("Content-Length") or 0), 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            h.update(b)
            done += len(b)
            if time.time() - last > 3:
                last = time.time()
                print(f"\r  {dest.name}: {done / 1e6:.0f}" + (f"/{total / 1e6:.0f}" if total else "") + " MB", end="", flush=True)
    print()
    if md5 and h.hexdigest() != md5:
        tmp.unlink()
        raise IOError(f"md5 mismatch for {dest.name}: got {h.hexdigest()}, expected {md5}")
    tmp.replace(dest)
    return dest


def _decode_images(col):
    from PIL import Image
    return np.stack([np.array(Image.open(io.BytesIO(c["bytes"])).convert("RGB")) for c in col])


def _read_parquet(files, label_candidates):
    import pyarrow.parquet as pq
    frames = [pq.read_table(str(f)).to_pandas() for f in sorted(files)]
    import pandas as pd
    df = pd.concat(frames, ignore_index=True)
    img_col = next(c for c in df.columns if isinstance(df[c].iloc[0], dict) and "bytes" in df[c].iloc[0])
    lab_col = next(c for c in label_candidates if c in df.columns)
    return _decode_images(df[img_col]), df[lab_col].to_numpy().astype(np.int64), df


# ------------------------------------------------------------------------------------------ CIFAR-10
def get_cifar10(dest):
    return hf_data.fetch_cifar10(dest)


# ------------------------------------------------------------------------------------------ SVHN
def _svhn_present(dest):
    return hf_data.find_svhn(dest, try_hf=False) is not None


def convert_svhn_parquet(train_files, test_files, dest):
    from scipy.io import savemat
    for name, files in (("train_32x32.mat", train_files), ("test_32x32.mat", test_files)):
        x, y, _ = _read_parquet(files, ("label", "labels"))
        if x.shape[1:] != (32, 32, 3):
            raise ValueError(f"unexpected SVHN image shape {x.shape}")
        if name.startswith("train"):
            counts = np.bincount(y, minlength=10).tolist()
            if len(x) != 73257 or counts != SVHN_TRAIN_COUNTS:
                raise ValueError(f"SVHN train check failed (n={len(x)}, class counts {counts}); expected 73257 images with digit counts "
                                 f"{SVHN_TRAIN_COUNTS}. The HF label mapping differs from what this script assumes - not saving.")
        y_mat = np.where(y == 0, 10, y).reshape(-1, 1).astype(np.uint8)       # official .mat stores digit 0 as 10
        savemat(str(Path(dest) / name), {"X": np.ascontiguousarray(x.transpose(1, 2, 3, 0)), "y": y_mat})   # X: [H,W,C,N]
        print(f"  wrote {name}: {len(x)} images")


def get_svhn(dest, source="auto", timeout=30):
    dest = Path(dest)
    if _svhn_present(dest):
        print("[skip] SVHN files already in", dest)
        return
    if source != "official" and hf_data.find_svhn(dest, try_hf=True) is not None:
        print("[done] SVHN from your Hugging Face repo")
        return
    if source in ("auto", "official"):
        try:
            for fname, (url, md5) in SVHN_URLS.items():
                if not (dest / fname).exists():
                    download(url, dest / fname, md5, timeout)
            print("[done] SVHN from the official site")
            return
        except Exception as e:  # noqa: BLE001
            print(f"[warn] official SVHN download failed ({e!r})")
            if source == "official":
                raise
    from huggingface_hub import snapshot_download
    local = Path(snapshot_download(repo_id="ufldl-stanford/svhn", repo_type="dataset", revision="refs/convert/parquet",
                                   allow_patterns=["cropped_digits/train/*", "cropped_digits/test/*"],
                                   local_dir=str(dest / "_hf_svhn_parquet"), token=hf_data._token()))
    tr, te = list((local / "cropped_digits" / "train").glob("*.parquet")), list((local / "cropped_digits" / "test").glob("*.parquet"))
    if not tr or not te:
        raise FileNotFoundError(f"no parquet files found under {local}/cropped_digits/{{train,test}}")
    convert_svhn_parquet(tr, te, dest)
    print("[done] SVHN from the Hugging Face mirror ufldl-stanford/svhn (converted to the official .mat format)")


# ------------------------------------------------------------------------------------------ CIFAR-100
def convert_cifar100_parquet(train_files, test_files, dest):
    out = Path(dest) / "cifar-100-python"
    out.mkdir(parents=True, exist_ok=True)
    for name, files, n in (("train", train_files, 50000), ("test", test_files, 10000)):
        x, y, df = _read_parquet(files, ("fine_label", "label"))
        counts = np.bincount(y, minlength=100)
        if len(x) != n or x.shape[1:] != (32, 32, 3) or not (counts == n // 100).all():
            raise ValueError(f"CIFAR-100 {name} check failed (n={len(x)}, shape {x.shape}); expected {n} balanced images - not saving.")
        coarse = df["coarse_label"].tolist() if "coarse_label" in df.columns else []
        data = np.ascontiguousarray(x.transpose(0, 3, 1, 2).reshape(len(x), 3072))     # official layout: R,G,B planes
        with open(out / name, "wb") as f:
            pickle.dump({"data": data, "fine_labels": y.tolist(), "coarse_labels": coarse}, f)
        print(f"  wrote cifar-100-python/{name}: {len(x)} images")


def get_cifar100(dest, source="auto", timeout=30):
    dest = Path(dest)
    if hf_data.find_cifar100(dest, try_hf=False) is not None:
        print("[skip] CIFAR-100 already in", dest)
        return
    if source != "official" and hf_data.find_cifar100(dest, try_hf=True) is not None:
        print("[done] CIFAR-100 from your Hugging Face repo")
        return
    if source in ("auto", "official"):
        try:
            tgz = download(CIFAR100_URL[0], dest / "cifar-100-python.tar.gz", CIFAR100_URL[1], timeout)
            with tarfile.open(tgz) as t:
                t.extractall(dest)
            print("[done] CIFAR-100 from the official site")
            return
        except Exception as e:  # noqa: BLE001
            print(f"[warn] official CIFAR-100 download failed ({e!r})")
            if source == "official":
                raise
    from huggingface_hub import hf_hub_download
    kw = dict(repo_id="uoft-cs/cifar100", repo_type="dataset", local_dir=str(dest / "_hf_cifar100_parquet"), token=hf_data._token())
    tr = hf_hub_download(filename="cifar100/train-00000-of-00001.parquet", **kw)
    te = hf_hub_download(filename="cifar100/test-00000-of-00001.parquet", **kw)
    convert_cifar100_parquet([tr], [te], dest)
    print("[done] CIFAR-100 from the Hugging Face mirror uoft-cs/cifar100 (converted to the official pickle format)")


# ------------------------------------------------------------------------------------------ EuroSAT
def eurosat_dir(dest):
    for c in (Path(dest) / "EuroSAT" / "2750", Path(dest) / "eurosat" / "2750"):
        if c.exists():
            return c
    return None


def get_eurosat(dest, timeout=60):
    dest = Path(dest)
    if eurosat_dir(dest) is not None:
        print("[skip] EuroSAT already in", eurosat_dir(dest))
        return
    if hf_data.find_eurosat_hf(dest) is not None:
        print("[done] EuroSAT from your Hugging Face repo")
        return
    base = dest / "eurosat"
    zp = download(EUROSAT_URL[0], base / "EuroSAT.zip", EUROSAT_URL[1], timeout)
    with zipfile.ZipFile(zp) as z:
        z.extractall(base)
    d = eurosat_dir(dest)
    if d is None:
        found = [p for p in base.rglob("2750") if p.is_dir()]
        if not found:
            raise FileNotFoundError(f"EuroSAT.zip extracted but no '2750' folder found under {base}")
        shutil.move(str(found[0]), str(base / "2750"))
        d = base / "2750"
    print(f"[done] EuroSAT -> {d} ({len([p for p in d.iterdir() if p.is_dir()])} classes)")


# ------------------------------------------------------------------------------------------ AWE
def get_awe(dest, source=None, repo="https://github.com/youngmanyd/NGIT-2022-AWE-Industrial-Dataset"):
    from .awe_prep import find_awe_root, list_dataset
    dest = Path(dest)
    clone = dest / "_awe_clone"
    try:
        root = find_awe_root(clone if clone.exists() else dest / "AWE")
        print("[skip] AWE already at", root)
        return root
    except FileNotFoundError:
        pass
    if source is None:
        subprocess.run(["git", "clone", "--depth", "1", repo, str(clone)], check=True)
    else:
        src = Path(source)
        if src.suffix == ".zip":
            with zipfile.ZipFile(src) as z:
                z.extractall(clone)
        elif src.is_dir():
            shutil.copytree(src, clone, dirs_exist_ok=True)
        else:
            raise FileNotFoundError(source)
    root = find_awe_root(clone)
    print("[done] AWE ->", root, {s: len(g) for s, g in list_dataset(root).items()}, "(expected 7,8,8,8,8,8)")
    return root


# ------------------------------------------------------------------------------------------ status
def status(dest=DATA_DIR):
    from .awe_prep import find_awe_root
    dest = Path(dest)
    try:
        awe = find_awe_root(dest / "_awe_clone" if (dest / "_awe_clone").exists() else dest / "AWE")
    except FileNotFoundError:
        awe = None
    return {"cifar10": hf_data._complete(hf_data.cifar10_dir(dest)), "svhn": _svhn_present(dest),
            "cifar100": hf_data.find_cifar100(dest, try_hf=False) is not None,
            "eurosat": eurosat_dir(dest) is not None, "awe": awe is not None}
