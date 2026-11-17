"""Dataset access, built so that a dataset is downloaded ONCE and then lives inside the repo (data/raw):

  1. cached arrays   data/raw/<name>/arrays.npz  (written after any successful load, so every later run is instant)
  2. bundled copy    a read-only data/raw next to the code, e.g. the Kaggle input dataset
                     /kaggle/input/datasets/<user>/<dataset>/FedUP_Reproduction/data/raw  -> linked into data/raw, no copy
  3. torchvision files already on disk (download=False)
  4. Hugging Face hub (parquet via `datasets`): much faster than the University of Toronto mirror torchvision uses
  5. the personal HF mirror (CIFAR-10 only), then torchvision's own download (slow, last resort)

AG News is tokenised once with the text model's tokenizer (fast -> slow -> bert-base-uncased fallbacks, because some
tokenizer repos cannot be converted to a fast tokenizer without sentencepiece) and cached as .npz.
Nothing here is needed for the synthetic smoke test."""
import os
import pickle
import shutil
from pathlib import Path

import numpy as np

from .paths import RAW_DATA_DIR, ROOT

HF_MIRROR = os.environ.get("FEDUP_HF_MIRROR", "Sakhiur/blue-bird-77")      # optional personal mirror (CIFAR-10 folder cifar-10/)
SVHN_TRAIN_COUNTS = [4948, 13861, 10585, 8497, 7458, 6882, 5727, 5595, 5045, 4659]     # digits 0..9, 73,257 training images
EUROSAT_URL = ("https://huggingface.co/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip",
               "c8fa014336c82ac7804f0398fcb19387")                 # exactly the file torchvision.datasets.EuroSAT downloads
HF_IDS = {"mnist": ("ylecun/mnist", "image", "label"),
          "cifar10": ("uoft-cs/cifar10", "img", "label"),
          "cifar100": ("uoft-cs/cifar100", "img", "fine_label")}
# svhn is loaded separately (config name "cropped_digits"); eurosat from the official class folders


def _token():
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------------------------------------ bundled raw data
def bundled_raw_dirs():
    """Read-only data/raw folders that ship with the code (FEDUP_BUNDLED_RAW, the repo's own data/raw, Kaggle inputs)."""
    out = []
    for p in os.environ.get("FEDUP_BUNDLED_RAW", "").split(os.pathsep):
        if p:
            out.append(Path(p))
    out.append(Path(ROOT) / "data" / "raw")
    kin = Path("/kaggle/input")
    if kin.exists():
        for dp, dn, _ in os.walk(kin):
            if len(Path(dp).relative_to(kin).parts) > 6:
                dn[:] = []
                continue
            if Path(dp).parts[-2:] == ("data", "raw"):
                out.append(Path(dp))
                dn[:] = []
    seen, res = set(), []
    for p in out:
        try:
            rp = p.resolve()
        except OSError:
            continue
        if p.is_dir() and rp != RAW_DATA_DIR.resolve() and rp not in seen:
            seen.add(rp)
            res.append(p)
    return res


def link_bundled_raw(verbose=True):
    """Symlink (or copy, if links are not allowed) every top-level item of the bundled raw folders into data/raw."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in bundled_raw_dirs():
        for e in sorted(src.iterdir()):
            if e.name.startswith("."):
                continue
            dst = RAW_DATA_DIR / e.name
            if dst.exists() or dst.is_symlink():
                continue
            try:
                os.symlink(e, dst, target_is_directory=e.is_dir())
            except OSError:
                shutil.copytree(e, dst) if e.is_dir() else shutil.copy2(e, dst)
            n += 1
    if n and verbose:
        print(f"[data] linked {n} bundled raw item(s) from {[str(p) for p in bundled_raw_dirs()]} into {RAW_DATA_DIR}")
    return n


def dataset_status():
    """{dataset: True/False} for what is already available locally (cached arrays, torchvision files or tokenised AG News)."""
    link_bundled_raw(verbose=False)
    r = RAW_DATA_DIR
    return {"mnist": (r / "mnist" / "arrays.npz").exists() or (r / "MNIST").exists(),
            "cifar10": (r / "cifar10" / "arrays.npz").exists() or (r / "cifar-10-batches-py").exists(),
            "cifar100": (r / "cifar100" / "arrays.npz").exists() or (r / "cifar-100-python").exists(),
            "svhn": (r / "svhn" / "arrays.npz").exists() or (r / "train_32x32.mat").exists(),
            "eurosat": (r / "eurosat" / "arrays.npz").exists() or (r / "eurosat" / "2750").exists() or (r / "EuroSAT" / "2750").exists(),
            "agnews": any((r / "agnews").glob("*.npz")) if (r / "agnews").exists() else False}


# ------------------------------------------------------------------------------------------------ arrays cache
def _cache_path(name):
    return RAW_DATA_DIR / name / "arrays.npz"


def _load_cached(name):
    p = _cache_path(name)
    if p.exists():
        z = np.load(p)
        return z["tx"], z["ty"], z["ex"], z["ey"]
    return None


def _save_cached(name, arrays):
    p = _cache_path(name)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, tx=arrays[0], ty=arrays[1], ex=arrays[2], ey=arrays[3])      # uncompressed: fastest to reload
    except OSError as e:  # noqa: BLE001  (read-only location)
        print(f"[data] could not cache {name}: {e!r}")
    return arrays


def _to_chw(x):
    x = np.asarray(x)
    x = x[:, None] if x.ndim == 3 else x.transpose(0, 3, 1, 2)
    return np.ascontiguousarray(x, dtype=np.uint8)


def _hf_arrays(name):
    from datasets import load_dataset
    repo, ic, lc = HF_IDS[name]
    ds = load_dataset(repo)

    def conv(split):
        d = ds[split]
        try:
            x = np.asarray(d.with_format("numpy")[ic])
            if x.dtype == object:
                raise ValueError("ragged")
        except Exception:  # noqa: BLE001
            x = np.stack([np.asarray(im) for im in d[ic]])
        return _to_chw(x), np.asarray(d[lc], dtype=np.int64)

    tx, ty = conv("train")
    ex, ey = conv("test")
    return tx, ty, ex, ey


def _cifar10_personal_mirror(root):
    from huggingface_hub import snapshot_download
    cache = Path(root) / "_hf_cifar10"
    local = Path(snapshot_download(repo_id=HF_MIRROR, repo_type="dataset", allow_patterns=["cifar-10/cifar-10-batches-py/*"],
                                   local_dir=str(cache), token=_token()))
    d = local / "cifar-10" / "cifar-10-batches-py"

    def read(fname):
        with open(d / fname, "rb") as f:
            e = pickle.load(f, encoding="latin1")
        return np.asarray(e["data"], dtype=np.uint8), np.asarray(e["labels"], dtype=np.int64)

    parts = [read(f"data_batch_{i}") for i in range(1, 6)]
    tx, ty = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    ex, ey = read("test_batch")
    return tx.reshape(-1, 3, 32, 32), ty, ex.reshape(-1, 3, 32, 32), ey


def _torchvision_arrays(name, root, download):
    import torchvision
    cls = {"mnist": torchvision.datasets.MNIST, "cifar10": torchvision.datasets.CIFAR10,
           "cifar100": torchvision.datasets.CIFAR100}[name]
    tr, te = cls(str(root), train=True, download=download), cls(str(root), train=False, download=download)
    return (_to_chw(tr.data), np.asarray(tr.targets, dtype=np.int64), _to_chw(te.data), np.asarray(te.targets, dtype=np.int64))


def _svhn_hf():
    """SVHN (cropped digits, 73,257 train / 26,032 test) from the Hugging Face hub; class counts are checked so a wrong label
    mapping cannot pass silently."""
    from datasets import load_dataset
    ds = load_dataset("ufldl-stanford/svhn", "cropped_digits")

    def conv(split):
        d = ds[split]
        try:
            x = np.asarray(d.with_format("numpy")["image"])
            if x.dtype == object:
                raise ValueError("ragged")
        except Exception:  # noqa: BLE001
            x = np.stack([np.asarray(im.convert("RGB")) for im in d["image"]])
        return _to_chw(x), np.asarray(d["label"], dtype=np.int64)

    tx, ty = conv("train")
    ex, ey = conv("test")
    counts = np.bincount(ty, minlength=10).tolist()
    if len(tx) != 73257 or counts != SVHN_TRAIN_COUNTS:
        raise ValueError(f"SVHN check failed (n={len(tx)}, class counts {counts}); expected {SVHN_TRAIN_COUNTS}: label mapping differs, not using it")
    return tx, ty, ex, ey


def _svhn_torchvision(root, download):
    import torchvision
    tr = torchvision.datasets.SVHN(str(root), split="train", download=download)
    te = torchvision.datasets.SVHN(str(root), split="test", download=download)
    return (np.ascontiguousarray(tr.data, dtype=np.uint8), np.asarray(tr.labels, dtype=np.int64),
            np.ascontiguousarray(te.data, dtype=np.uint8), np.asarray(te.labels, dtype=np.int64))


def _eurosat_dir(root):
    for c in (Path(root) / "EuroSAT" / "2750", Path(root) / "eurosat" / "2750"):
        if c.exists():
            return c
    return None


def _eurosat_download(root):
    import urllib.request
    import zipfile
    base = Path(root) / "eurosat"
    base.mkdir(parents=True, exist_ok=True)
    zp = base / "EuroSAT.zip"
    if not zp.exists():
        req = urllib.request.Request(EUROSAT_URL[0], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(zp, "wb") as f:
            shutil.copyfileobj(r, f)
    with zipfile.ZipFile(zp) as z:
        z.extractall(base)
    d = _eurosat_dir(root)
    if d is None:
        found = [p for p in base.rglob("2750") if p.is_dir()]
        if not found:
            raise FileNotFoundError(f"EuroSAT.zip extracted but no '2750' folder found under {base}")
        shutil.move(str(found[0]), str(base / "2750"))


def _eurosat_arrays(root, size=32, test_frac=0.2, seed=0):
    """EuroSAT RGB (27,000 images, 10 classes) resized to size x size, then a fixed random 80/20 train/test split (the EuroSAT
    release has no official split; FuGuard also splits 80/20 at random, with the run seed - here the split is fixed to seed 0)."""
    from PIL import Image
    d = _eurosat_dir(root)
    if d is None:
        _eurosat_download(root)
        d = _eurosat_dir(root)
    classes = sorted(p.name for p in d.iterdir() if p.is_dir())
    xs, ys = [], []
    for k, c in enumerate(classes):
        for f in sorted((d / c).iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif"):
                xs.append(np.asarray(Image.open(f).convert("RGB").resize((size, size), Image.BILINEAR)))
                ys.append(k)
    x, y = np.stack(xs), np.asarray(ys, dtype=np.int64)
    perm = np.random.RandomState(seed).permutation(len(y))
    n_test = int(round(len(y) * test_frac))
    te, tr = perm[:n_test], perm[n_test:]
    return _to_chw(x[tr]), y[tr], _to_chw(x[te]), y[te]


def load_image_arrays(name, root=None):
    """-> train_x [N,C,H,W] uint8, train_y int64, test_x, test_y   for mnist / cifar10 / cifar100 / svhn / eurosat (cached in data/raw)."""
    root = Path(root) if root else RAW_DATA_DIR
    root.mkdir(parents=True, exist_ok=True)
    link_bundled_raw()
    hit = _load_cached(name)
    if hit is not None:
        return hit
    if name == "svhn":
        steps = [("torchvision files on disk", lambda: _svhn_torchvision(root, False)), ("Hugging Face hub", _svhn_hf),
                 ("torchvision download", lambda: _svhn_torchvision(root, True))]
    elif name == "eurosat":
        steps = [("EuroSAT class folders (download if missing)", lambda: _eurosat_arrays(root))]
    else:
        steps = [("torchvision files on disk", lambda: _torchvision_arrays(name, root, False)),
                 ("Hugging Face hub", lambda: _hf_arrays(name))]
        if name == "cifar10":
            steps.append(("personal HF mirror", lambda: _cifar10_personal_mirror(root)))
        steps.append(("torchvision download (slow)", lambda: _torchvision_arrays(name, root, True)))
    errs = []
    for label, fn in steps:
        try:
            print(f"[data] {name}: trying {label} ...", flush=True)
            return _save_cached(name, fn())
        except Exception as e:  # noqa: BLE001
            errs.append(f"{label}: {repr(e)[:160]}")
            print(f"[data] {name}: {label} failed ({repr(e)[:120]})")
    raise RuntimeError(f"could not load {name}:\n  " + "\n  ".join(errs))


# ------------------------------------------------------------------------------------------------ AG News (text)
def load_tokenizer(text_model):
    """Fast tokenizer first; if the repo cannot be converted to a fast tokenizer (the sentencepiece / tiktoken error), fall back to
    the slow pure-python BERT WordPiece tokenizer, and finally to bert-base-uncased (same uncased vocabulary as BERT-tiny)."""
    from transformers import AutoTokenizer
    tries = [("AutoTokenizer", lambda: AutoTokenizer.from_pretrained(text_model, use_fast=True))]
    try:
        from transformers import BertTokenizer, BertTokenizerFast
        tries += [("BertTokenizerFast", lambda: BertTokenizerFast.from_pretrained(text_model)),
                  ("BertTokenizer (slow)", lambda: BertTokenizer.from_pretrained(text_model))]
    except Exception:  # noqa: BLE001
        pass
    tries.append(("bert-base-uncased", lambda: AutoTokenizer.from_pretrained("bert-base-uncased")))
    errs = []
    for label, fn in tries:
        try:
            tok = fn()
            if errs:
                print(f"[data] tokenizer: using {label} (earlier attempts failed: {errs})")
            return tok
        except Exception as e:  # noqa: BLE001
            errs.append(f"{label}: {repr(e)[:100]}")
    raise RuntimeError("no tokenizer could be loaded: " + " | ".join(errs))


def load_agnews_arrays(text_model, max_len, root=None):
    """-> train_x [N,2,L] int64 (ids; attention mask), train_y, test_x, test_y. Tokenised once and cached as .npz."""
    root = Path(root) if root else RAW_DATA_DIR
    link_bundled_raw()
    slug = text_model.replace("/", "_").replace("\\", "_")
    cache = root / "agnews" / f"{slug}_L{max_len}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["tx"], z["ty"], z["ex"], z["ey"]
    from datasets import load_dataset
    try:
        ds = load_dataset("fancyzhx/ag_news")
    except Exception:  # noqa: BLE001
        ds = load_dataset("ag_news")
    tok = load_tokenizer(text_model)

    def enc(split):
        texts = list(ds[split]["text"])
        out = tok(texts, truncation=True, padding="max_length", max_length=max_len, return_tensors="np")
        x = np.stack([out["input_ids"], out["attention_mask"]], axis=1).astype(np.int64)
        return x, np.asarray(ds[split]["label"], dtype=np.int64)

    tx, ty = enc("train")
    ex, ey = enc("test")
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, tx=tx, ty=ty, ex=ex, ey=ey)
    except OSError as e:  # noqa: BLE001
        print(f"[data] could not cache AG News: {e!r}")
    return tx, ty, ex, ey


def synthetic_arrays(num_classes=10, n_train=6000, n_test=1000, seed=0):
    """Class-conditional random images (3x32x32): every class has its own colour (+ a faint pattern), so a tiny CNN learns
    them in a few rounds. Only meant to exercise the pipeline (smoke test), never to produce results."""
    rng = np.random.RandomState(seed)
    colours = rng.rand(num_classes, 3, 1, 1) * 0.8 + 0.1
    pattern = rng.rand(num_classes, 3, 32, 32) * 0.1

    def make(n):
        y = rng.randint(0, num_classes, size=n).astype(np.int64)
        x = colours[y] + pattern[y] + rng.rand(n, 3, 32, 32) * 0.08
        return np.clip(x * 255, 0, 255).astype(np.uint8), y

    tx, ty = make(n_train)
    ex, ey = make(n_test)
    return tx, ty, ex, ey
