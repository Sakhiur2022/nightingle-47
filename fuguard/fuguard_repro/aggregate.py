"""Turn raw runs into the paper's tables (II, III, IV) and curve arrays (Fig. 6/9/10)."""
import numpy as np
import pandas as pd

from .config import METHODS, METHOD_LABELS, IMAGE_DATASETS, ot_variant
from .io_results import load_runs
from .paper_reference import PAPER_TABLE2, PAPER_TABLE3, PAPER_TABLE4

STD_DDOF = 0   # paper does not say; numpy default. Change to 1 for the sample std.


def mean_std(vals, ddof=None):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=STD_DDOF if ddof is None else ddof)) if len(a) > 1 else 0.0


def recovery_curve(res, metric):
    """Values for recovery rounds 1..R (round unlearn_round+1 ... unlearn_round+R)."""
    ur, R = res["unlearn_round"], res["recovery_rounds"]
    if metric == "acc":
        return np.array(res["test_acc_per_round"][ur:ur + R])
    if metric == "asr":
        return np.array(res["test_asr_per_round"][ur:ur + R])
    if metric == "mia":
        return np.array([m["attack_acc"] for m in res["mia_per_round_after"]][:R])
    raise ValueError(metric)


def end_value(res, metric, n_rec=None):
    """ACC/ASR at the end of the recovery phase (default: after all recovery rounds of the run)."""
    c = recovery_curve(res, metric)
    return float(c[(n_rec or len(c)) - 1])


def curves(experiment, tag, variant, metric):
    """[n_seeds, R] array (empty array if no runs)."""
    runs = load_runs(experiment, tag, variant)
    return np.stack([recovery_curve(r, metric) for r in runs]) if runs else np.empty((0, 0))


# ----------------------------------------------------------------------------- Table II
def table2_numeric(experiment="table2", datasets=IMAGE_DATASETS, methods=METHODS, n_rec=5):
    """{(dataset, 'fedavg'|method): {'acc':(m,s),'asr':(m,s),'n':k}} in %, values taken after n_rec recovery rounds."""
    out = {}
    for ds in datasets:
        per_method = {m: load_runs(experiment, ds, m) for m in methods}
        fed = {}
        for runs in per_method.values():          # pre-unlearning phase is identical across methods
            for r in runs:
                fed.setdefault(r["seed"], []).append((r["acc_before_unlearn"], r["asr_before_unlearn"]))
        if fed:
            accs = [np.mean([a for a, _ in v]) * 100 for v in fed.values()]
            asrs = [np.mean([b for _, b in v]) * 100 for v in fed.values()]
            out[(ds, "fedavg")] = {"acc": mean_std(accs), "asr": mean_std(asrs), "n": len(fed)}
        for m, runs in per_method.items():
            if runs:
                out[(ds, m)] = {"acc": mean_std([end_value(r, "acc", n_rec) * 100 for r in runs]),
                                "asr": mean_std([end_value(r, "asr", n_rec) * 100 for r in runs]), "n": len(runs)}
    return out


def table2_df(num, datasets=IMAGE_DATASETS, methods=METHODS):
    rows = []
    for key in ["fedavg"] + list(methods):
        row = {"Method": "FedAvg (training)" if key == "fedavg" else METHOD_LABELS[key]}
        for ds in datasets:
            v = num.get((ds, key))
            row[f"{ds} ACC"] = f"{v['acc'][0]:.2f}±{v['acc'][1]:.2f}" if v else "—"
            row[f"{ds} ASR"] = f"{v['asr'][0]:.2f}±{v['asr'][1]:.2f}" if v else "—"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Method")


def table2_vs_paper(num, datasets=IMAGE_DATASETS, methods=METHODS):
    """Long-format comparison: ours vs paper (means) with absolute difference in percentage points."""
    rows = []
    for ds in datasets:
        for key in ["fedavg"] + list(methods):
            v, p = num.get((ds, key)), PAPER_TABLE2[ds][key]
            if not v:
                continue
            rows.append({"dataset": ds, "method": METHOD_LABELS.get(key, "FedAvg"),
                         "ACC ours": round(v["acc"][0], 2), "ACC paper": p[0], "ΔACC": round(v["acc"][0] - p[0], 2),
                         "ASR ours": round(v["asr"][0], 2), "ASR paper": p[2], "ΔASR": round(v["asr"][0] - p[2], 2),
                         "seeds": v["n"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Table III
def table3_numeric(experiment="table3", datasets=IMAGE_DATASETS, methods=METHODS, seed=0):
    """{(dataset, method): (RR95, RR100, cap, reached95, reached100)}  RR = recovery rounds after the unlearning round."""
    out = {}
    for ds in datasets:
        for m in methods:
            runs = [r for r in load_runs(experiment, ds, m) if r["seed"] == seed]
            if not runs:
                continue
            r, ur, cap = runs[0], runs[0]["unlearn_round"], runs[0]["recovery_rounds"]
            f = lambda a: (a - ur) if a is not None else cap  # noqa: E731
            out[(ds, m)] = (f(r["recovery_round_95"]), f(r["recovery_round_full"]), cap,
                            r["recovery_round_95"] is not None, r["recovery_round_full"] is not None)
    return out


def table3_df(num, datasets=IMAGE_DATASETS, methods=METHODS):
    rows = []
    for m in methods:
        row = {"Method": METHOD_LABELS[m]}
        for ds in datasets:
            v = num.get((ds, m))
            row[f"{ds} RR(0.95)"] = ("—" if not v else f"{v[0]}" + ("" if v[3] else "*"))
            row[f"{ds} RR(1)"] = ("—" if not v else f"{v[1]}" + ("" if v[4] else "*"))
        rows.append(row)
    return pd.DataFrame(rows).set_index("Method")   # '*' = not reached within the recovery budget (value = cap)


def table3_vs_paper(num, datasets=IMAGE_DATASETS, methods=METHODS):
    rows = []
    for ds in datasets:
        for m in methods:
            v = num.get((ds, m))
            if v:
                p = PAPER_TABLE3[ds][m]
                rows.append({"dataset": ds, "method": METHOD_LABELS[m], "RR95 ours": v[0], "RR95 paper": p[0],
                             "RR1 ours": v[1], "RR1 paper": p[1]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Table IV (ablation, SVHN)
def table4_df(tag="SVHN", n_rec=5):
    rows = []
    for lam in (0.0, 0.1):
        runs = (load_runs("ablation", tag, ot_variant(0.0)) if lam == 0.0 else load_runs("table2", tag, "fuguard"))
        if not runs:
            continue
        for mode in ("only unlearn", "with recovery"):
            if mode == "only unlearn":
                acc = [r["acc_post_unlearn"] * 100 for r in runs]; asr = [r["asr_post_unlearn"] * 100 for r in runs]
            else:
                acc = [end_value(r, "acc", n_rec) * 100 for r in runs]; asr = [end_value(r, "asr", n_rec) * 100 for r in runs]
            p = PAPER_TABLE4[(lam, mode)]
            rows.append({"λ_OT": lam, "mode": mode, "ACC ours": round(np.mean(acc), 2), "ACC paper": p[0],
                         "ASR ours": round(np.mean(asr), 2), "ASR paper": p[1], "seeds": len(runs)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Fig. 10 (sensitivity)
SENS_LAMBDAS = (0.08, 0.09, 0.10, 0.11, 0.12)


def sensitivity_numeric(tag, n_rec=5, lambdas=SENS_LAMBDAS):
    """{lambda: (ACC mean, ASR mean)} at the end of recovery; lambda=0.10 is reused from the Table II run."""
    out = {}
    for lam in lambdas:
        runs = load_runs("table2", tag, "fuguard") if abs(lam - 0.1) < 1e-12 else load_runs("sensitivity", tag, ot_variant(lam))
        if runs:
            out[lam] = (float(np.mean([end_value(r, "acc", n_rec) for r in runs])),
                        float(np.mean([end_value(r, "asr", n_rec) for r in runs])))
    return out
