"""Google Colab helpers. Everything is stored on the Colab VM (/content/...). The VM is ephemeral:
download your results (zip_results) before the runtime is recycled, or use the optional Drive backup."""
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from .paths import ROOT, RESULTS_DIR, CKPT_DIR, PROCESSED_DIR


def in_colab():
    try:
        import importlib.util
        return importlib.util.find_spec("google.colab") is not None
    except Exception:
        return False


def _size_gb(path):
    total = 0
    for dp, _, fn in os.walk(path):
        for f in fn:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total / 1e9


def status():
    """GPU, RAM, disk and how much space data/results/checkpoints use."""
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE (Runtime > Change runtime type > GPU)"
        print(f"torch {torch.__version__} | GPU: {gpu}")
    except ImportError:
        print("torch not installed")
    du = shutil.disk_usage(ROOT)
    print(f"disk free: {du.free / 1e9:.1f} GB of {du.total / 1e9:.1f} GB")
    try:
        mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
        print(f"RAM available: {mem['MemAvailable'] / 1e6:.1f} GB of {mem['MemTotal'] / 1e6:.1f} GB")
    except Exception:
        pass
    for name, p in (("processed data", PROCESSED_DIR.parent), ("checkpoints", CKPT_DIR), ("results", RESULTS_DIR)):
        print(f"{name:>15}: {_size_gb(p):6.2f} GB  ({p})" if p.exists() else f"{name:>15}: (empty)")
    print("project root:", ROOT)


def zip_results(include_checkpoints=True, download=True):
    """Zip results/ (+ cached pre-unlearning checkpoints) and trigger a browser download in Colab."""
    out = ROOT.parent / f"fuguard_results_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        dirs = [RESULTS_DIR] + ([CKPT_DIR] if include_checkpoints and CKPT_DIR.exists() else [])
        for d in dirs:
            for dp, _, fn in os.walk(d):
                for f in fn:
                    p = Path(dp) / f
                    z.write(p, p.relative_to(ROOT))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if download and in_colab():
        from google.colab import files
        files.download(str(out))
    return out


def backup_to_drive(dest="MyDrive/FuGuard_backup", include_checkpoints=True):
    """OPTIONAL: copy results (and checkpoints) to Google Drive so they survive a recycled runtime."""
    from google.colab import drive
    drive.mount("/content/drive")
    base = Path("/content/drive") / dest
    for src in [RESULTS_DIR] + ([CKPT_DIR] if include_checkpoints else []):
        if src.exists():
            shutil.copytree(src, base / src.relative_to(ROOT), dirs_exist_ok=True)
    print("backed up to", base)


def restore_from_drive(src="MyDrive/FuGuard_backup"):
    """OPTIONAL: bring a previous backup back into a fresh runtime (finished runs are then skipped)."""
    from google.colab import drive
    drive.mount("/content/drive")
    base = Path("/content/drive") / src
    for sub in ("results", "artifacts/checkpoints"):
        if (base / sub).exists():
            shutil.copytree(base / sub, ROOT / sub, dirs_exist_ok=True)
    print("restored from", base)


def run_in_background(cli_args, log="logs/run.log"):
    """Start scripts/run_experiment.py in a separate process (keeps going if the browser tab disconnects,
    but NOT if Colab recycles the runtime; runs are resumable). Don't run two GPU jobs at once."""
    log_path = ROOT / log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "a")
    proc = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "run_experiment.py"), *cli_args],
                            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT), start_new_session=True)
    print(f"started pid {proc.pid}; follow with colab.tail_log('{log}')")
    return proc


def tail_log(log="logs/run.log", n=25):
    p = ROOT / log
    print("".join(p.read_text().splitlines(True)[-n:]) if p.exists() else f"{p} does not exist yet")


def fetch_awe_dataset(source=None, repo="https://github.com/youngmanyd/NGIT-2022-AWE-Industrial-Dataset"):
    """Make the AWE dataset available under data/_awe_clone and check its layout.
    source=None            -> git clone (about 2 GB, 50 JSON files)
    source='/path/x.zip'   -> unzip your copy (e.g. NGIT-2022-AWE-Industrial-Dataset-master.zip from Drive)
    source='/path/folder'  -> use an existing folder (e.g. on a mounted Drive)
    Expected layout: Dataset/normal, Dataset/1chip, Dataset/2chips ... Dataset/5chips (the nesting depth does not matter)."""
    from .awe_prep import find_awe_root, list_dataset
    dst = ROOT / "data" / "_awe_clone"
    if source is None:
        if not dst.exists():
            subprocess.run(["git", "clone", "--depth", "1", repo, str(dst)], check=True)
    else:
        src = Path(source)
        if src.suffix == ".zip":
            with zipfile.ZipFile(src) as z:
                z.extractall(dst)
        elif src.is_dir():
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(src, dst)
        else:
            raise FileNotFoundError(source)
    root = find_awe_root(dst)
    counts = {s: len(g) for s, g in list_dataset(root).items()}
    print("AWE root:", root)
    print("files found per state (0=normal, 1..5=chips):", counts, "- expected 7, 8, 8, 8, 8, 8")
    return root
