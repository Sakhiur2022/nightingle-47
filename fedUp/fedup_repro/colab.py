"""Helpers for Kaggle / Colab sessions."""
import os
import subprocess
import sys
from pathlib import Path


def platform():
    if os.path.exists("/kaggle/working"):
        return "kaggle"
    try:
        import google.colab  # noqa: F401
        return "colab"
    except Exception:  # noqa: BLE001
        return "local"


def setup_dirs(persist_root=None):
    """Data, artifacts and results live INSIDE the repo folder (data/raw, data/processed, artifacts/..., results/...) so that a
    downloaded dataset or a trained model is simply part of the project. Pass persist_root (e.g. a mounted Drive folder) to
    keep data/ and artifacts/ somewhere that survives the session. Any read-only data/raw that ships with the code (the Kaggle
    input dataset) is linked in, so datasets never have to be downloaded again."""
    p = platform()
    if persist_root:
        root = Path(persist_root)
        root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("FEDUP_DATA_DIR", str(root / "data"))
        os.environ.setdefault("FEDUP_ARTIFACTS_DIR", str(root / "artifacts"))
    from . import downloaders
    from .paths import ensure_dirs
    ensure_dirs()
    downloaders.link_bundled_raw()
    return p


def dataset_status():
    """Which datasets are already available locally (nothing is downloaded for those)."""
    from . import downloaders
    from .paths import RAW_DATA_DIR
    st = downloaders.dataset_status()
    print(f"raw datasets in {RAW_DATA_DIR}: " + ", ".join(f"{k}: {'present' if v else 'MISSING'}" for k, v in st.items()))
    return st


def pip_install(requirements="requirements-colab.txt"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", requirements], check=True)


def gpu_summary():
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "no GPU"


def status():
    from .paths import ARTIFACTS_DIR, CKPT_DIR, DATA_DIR, FILTER_DIR, RAW_DIR
    n = lambda d, pat: len(list(Path(d).rglob(pat))) if Path(d).exists() else 0  # noqa: E731
    print(f"platform {platform()} | data {DATA_DIR} | artifacts {ARTIFACTS_DIR}")
    print(f"checkpoints {n(CKPT_DIR, '*.pt')} | filters {n(FILTER_DIR, '*.pt')} | finished runs {n(RAW_DIR, 'seed*.pkl')}")


# ------------------------------------------------------------------------------------------------ notebook config
CONFIG_NAME = "notebook_config.json"


def config_path():
    from .paths import ARTIFACTS_DIR
    return ARTIFACTS_DIR / CONFIG_NAME


def sets_from_hp(hp):
    """['key=value', ...] for every hyper-parameter that differs from the library default (None = keep the default)."""
    from .config import BASE
    out = []
    for k, v in (hp or {}).items():
        if v is None:
            continue
        if k in BASE and BASE[k] == v:
            continue
        out.append(f"{k}={v!r}")
    return out


def save_config(seeds=(0, 1), preset=None, quick=False, gpus=None, hp=None):
    """Persist the notebook settings in artifacts/notebook_config.json. Every experiment cell reads them from there, so after a
    kernel restart only the Setup cell has to be re-run (and the file travels inside the checkpoint zip)."""
    import json
    cfg = dict(seeds=list(seeds), preset=preset, quick=bool(quick), gpus=gpus, hp=hp or {}, sets=sets_from_hp(hp))
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=1))
    print(f"saved {p}\n  seeds {cfg['seeds']} | preset {preset or 'repo'} | quick {quick} | gpus {gpus or 'all visible'}")
    print("  changed hyper-parameters:", cfg["sets"] or "none (library defaults)")
    return cfg


def load_config():
    import json
    p = config_path()
    if p.exists():
        return json.loads(p.read_text())
    return dict(seeds=[0, 1], preset=None, quick=False, gpus=None, hp={}, sets=[])


def show_config():
    cfg = load_config()
    print({k: v for k, v in cfg.items() if k != "hp"})
    return cfg


def run_exp(experiment, benchmarks=None, methods=None, seeds=None, sets=None, preset=None, quick=None, gpus=None):
    """One experiment slice as a subprocess (multi-GPU is automatic). Any argument left as None is taken from the saved notebook
    config (`save_config`). Finished runs are skipped, so any cell can be re-run."""
    from .paths import ROOT
    c = load_config()
    seeds = c["seeds"] if seeds is None else seeds
    sets = c["sets"] if sets is None else sets
    preset = c["preset"] if preset is None else preset
    quick = c["quick"] if quick is None else quick
    gpus = c["gpus"] if gpus is None else gpus
    cmd = [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), experiment, "--seeds", *map(str, seeds)]
    if benchmarks:
        cmd += ["--benchmarks", *benchmarks]
    if methods:
        cmd += ["--methods", *methods]
    for s in sets:
        cmd += ["--set", s]
    if preset:
        cmd += ["--preset", preset]
    if quick:
        cmd += ["--quick"]
    if gpus:
        cmd += ["--gpus", str(gpus)]
    print("$", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def progress(experiment="main", benchmarks=None, methods=None, seeds=None):
    """How much of an experiment has finished: done / planned runs per benchmark, scenario and method."""
    import pandas as pd
    from . import experiments as E
    from .io_results import list_runs
    seeds = load_config()["seeds"] if seeds is None else seeds
    planned = E.plan(experiment, benchmarks, tuple(seeds), None, False, methods)
    done = {(t, m, s) for t, m, s in list_runs(experiment)}
    rows = {}
    for j in planned:
        tag = f"{j['benchmark']}__{j['scenario']}" + (f"__{j['tag_suffix']}" if j["tag_suffix"] else "")
        key = (j["benchmark"], j["scenario"], j["method"])
        r = rows.setdefault(key, [0, 0])
        r[1] += 1
        r[0] += int((tag, j["method"], j["seed"]) in done)
    df = pd.DataFrame([dict(benchmark=b, scenario=s, method=m, done=d, planned=n) for (b, s, m), (d, n) in rows.items()])
    if len(df):
        df["status"] = df.apply(lambda r: "done" if r.done == r.planned else ("partial" if r.done else "-"), axis=1)
    return df


# ------------------------------------------------------------------------------------------------ checkpoint zip
_STORED = {".pt", ".npz", ".png", ".zip", ".gz", ".jpg", ".parquet"}      # already compressed / incompressible: do not deflate


def _groups(raw_data, processed_data, checkpoints, filters, features, results):
    from . import paths
    g = []
    if raw_data:
        g.append(("data/raw", paths.RAW_DATA_DIR))
    if processed_data:
        g.append(("data/processed", paths.PROCESSED_DIR))
    if checkpoints:
        g.append(("artifacts/checkpoints", paths.CKPT_DIR))
    if filters:
        g.append(("artifacts/filters", paths.FILTER_DIR))
    if features:
        g.append(("artifacts/features", paths.FEATURE_DIR))
    if results:
        g.append(("results", paths.RESULTS_DIR))
    return g


def _default_out_dir():
    if os.path.exists("/kaggle/working"):
        return Path("/kaggle/working")
    if os.path.exists("/content"):
        return Path("/content")
    return Path.cwd()


def make_checkpoint_zip(out_dir=None, name=None, raw_data=True, processed_data=True, checkpoints=True, filters=True,
                        features=True, results=True, config=True):
    """Zip the current data/ and artifacts/ folders (plus results/ and the notebook config) into ONE file whose inside layout
    is always data/..., artifacts/..., results/... wherever the folders really live. Returns the zip path.
    Switch a group off to shrink the file (raw datasets are re-downloadable: raw_data=False)."""
    import time
    import zipfile
    from .paths import ARTIFACTS_DIR
    out_dir = Path(out_dir) if out_dir else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (name or f"fedup_checkpoint_{time.strftime('%Y%m%d_%H%M%S')}.zip")
    n_files, total = 0, 0
    with zipfile.ZipFile(path, "w", allowZip64=True) as z:
        def add(src, arc):
            comp = zipfile.ZIP_STORED if src.suffix.lower() in _STORED else zipfile.ZIP_DEFLATED
            z.write(src, arc, compress_type=comp, compresslevel=(1 if comp == zipfile.ZIP_DEFLATED else None))
        for arc_prefix, d in _groups(raw_data, processed_data, checkpoints, filters, features, results):
            n, size = 0, 0
            for dp, dn, fn in os.walk(d, followlinks=True):          # follow links: datasets linked from a bundled input folder count
                dn.sort()
                for name in sorted(fn):
                    f = Path(dp) / name
                    if name != ".gitkeep" and f.is_file():
                        add(f, f"{arc_prefix}/{f.relative_to(d).as_posix()}")
                        n += 1
                        size += f.stat().st_size
            print(f"  {arc_prefix:24s} {n:5d} files  {size / 2**20:9.1f} MB")
            n_files += n
            total += size
        cp = ARTIFACTS_DIR / CONFIG_NAME
        if config and cp.exists():
            add(cp, f"artifacts/{CONFIG_NAME}")
            n_files += 1
    print(f"zip: {path}  ({n_files} files, {total / 2**20:.1f} MB raw -> {path.stat().st_size / 2**20:.1f} MB)")
    return path


def restore_checkpoint_zip(path, overwrite=False):
    """Unpack a zip made by make_checkpoint_zip into the CURRENT data/artifacts/results folders (e.g. at the start of a new
    session). Existing files are kept unless overwrite=True. Finished runs and cached FedAvg models are then picked up again."""
    import zipfile
    from . import paths
    dest = {"data/raw": paths.RAW_DATA_DIR, "data/processed": paths.PROCESSED_DIR, "artifacts/checkpoints": paths.CKPT_DIR,
            "artifacts/filters": paths.FILTER_DIR, "artifacts/features": paths.FEATURE_DIR, "results": paths.RESULTS_DIR}
    n = skipped = 0
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            if info.filename == f"artifacts/{CONFIG_NAME}":
                target = paths.ARTIFACTS_DIR / CONFIG_NAME
            else:
                prefix = next((k for k in sorted(dest, key=len, reverse=True) if info.filename.startswith(k + "/")), None)
                if prefix is None:
                    continue
                target = Path(dest[prefix]) / info.filename[len(prefix) + 1:]
            if target.exists() and not overwrite:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            n += 1
    print(f"restored {n} files, kept {skipped} existing (overwrite={overwrite})")


def kaggle_link(path):
    """Clickable download link for a file under /kaggle/working (Kaggle interactive session)."""
    from IPython.display import FileLink
    return FileLink(os.path.relpath(str(path), "/kaggle/working"))


def run_script(name, *args):
    """Run one of scripts/*.py as a subprocess (so a crash never takes the notebook kernel down)."""
    from .paths import ROOT
    cmd = [sys.executable, str(ROOT / "scripts" / name), *map(str, args)]
    print("$", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))
