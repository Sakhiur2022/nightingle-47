"""results/raw/<experiment>/<tag>/<method>/seed<k>.pkl  (+ a small .json next to it)."""
import json
import pickle

from .paths import RAW_DIR


def tag_of(benchmark, scenario):
    return f"{benchmark}__{scenario}"


def run_dir(experiment, tag, method):
    d = RAW_DIR / experiment / tag / method
    d.mkdir(parents=True, exist_ok=True)
    return d


def result_exists(experiment, tag, method, seed):
    return (RAW_DIR / experiment / tag / method / f"seed{seed}.pkl").exists()


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if hasattr(o, "item") and getattr(o, "ndim", 1) == 0:
        return o.item()
    return o if isinstance(o, (int, float, str, bool)) or o is None else str(o)


def save_result(experiment, tag, method, seed, result):
    d = run_dir(experiment, tag, method)
    with open(d / f"seed{seed}.pkl", "wb") as f:
        pickle.dump(result, f)
    summary = {k: v for k, v in result.items() if k not in ("cfg", "base_history")}
    with open(d / f"seed{seed}.json", "w") as f:
        json.dump(_plain(summary), f, indent=1)


def load_runs(experiment, tag, method):
    d = RAW_DIR / experiment / tag / method
    if not d.exists():
        raise FileNotFoundError(str(d))
    out = []
    for p in sorted(d.glob("seed*.pkl")):
        with open(p, "rb") as f:
            out.append(pickle.load(f))
    return out


def list_runs(experiment):
    """-> [(tag, method, seed)] present on disk."""
    base = RAW_DIR / experiment
    res = []
    if not base.exists():
        return res
    for tag_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for m_dir in sorted(p for p in tag_dir.iterdir() if p.is_dir()):
            for s in sorted(m_dir.glob("seed*.pkl")):
                res.append((tag_dir.name, m_dir.name, int(s.stem[4:])))
    return res
