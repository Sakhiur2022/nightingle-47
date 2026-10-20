"""Results store:  results/raw/<experiment>/<tag>/<variant>/seed<k>.pkl  (+ a human-readable .json summary)."""
import json
import pickle
from pathlib import Path

from .paths import RAW_DIR


def result_path(experiment, tag, variant, seed, root=RAW_DIR) -> Path:
    return Path(root) / experiment / tag / variant / f"seed{seed}.pkl"


def result_exists(experiment, tag, variant, seed, root=RAW_DIR):
    return result_path(experiment, tag, variant, seed, root).exists()


def save_result(res, experiment, tag, variant, seed, root=RAW_DIR):
    p = result_path(experiment, tag, variant, seed, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(res, f)
    ur, R = res["unlearn_round"], res["recovery_rounds"]
    summary = {"experiment": experiment, "tag": tag, "variant": variant, "seed": seed, "method": res["method"],
               "acc_before_unlearn": res["acc_before_unlearn"], "asr_before_unlearn": res["asr_before_unlearn"],
               "acc_post_unlearn": res["acc_post_unlearn"], "asr_post_unlearn": res["asr_post_unlearn"],
               "acc_end": res["test_acc_per_round"][ur + R - 1], "asr_end": res["test_asr_per_round"][ur + R - 1],
               "recovery_round_95": res["recovery_round_95"], "recovery_round_full": res["recovery_round_full"],
               "unlearn_time_s": res["unlearn_time"]}
    p.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return p


def load_result(experiment, tag, variant, seed, root=RAW_DIR):
    with open(result_path(experiment, tag, variant, seed, root), "rb") as f:
        return pickle.load(f)


def load_runs(experiment, tag, variant, root=RAW_DIR):
    """All seeds available for (experiment, tag, variant), sorted by seed."""
    d = Path(root) / experiment / tag / variant
    out = []
    for p in sorted(d.glob("seed*.pkl"), key=lambda q: int(q.stem[4:])):
        with open(p, "rb") as f:
            out.append(pickle.load(f))
    return out
