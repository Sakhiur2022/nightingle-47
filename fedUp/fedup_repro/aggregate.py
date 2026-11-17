"""Turn results/raw into tables (pandas): ours vs the paper, cost, ablations, cross-protocol (FuGuard-style)."""
import pandas as pd

from . import paper_reference as P
from .io_results import list_runs, load_runs
from .utils import mean_std

SC_KEYS = {"client": ["R_A", "F_A", "MIA"], "class": ["R_A", "F_A", "MIA"], "sample": ["0A", "PS", "MIA"],
           "client_backdoor": ["ACC", "ASR", "ASR_incl_target", "MIA"]}


def _flat(r):
    d = dict(benchmark=r["benchmark"], scenario=r["scenario"], method=r["method"], seed=r["seed"])
    for k, v in r["post"].items():
        d[k] = v
    for k, v in r["pre"].items():
        d["pre_" + k] = v
    d.update(MIA=r["MIA"], time_s=r["time_s"], time_filter_s=r.get("time_filter_s"), time_client_s=r.get("time_client_s"),
             comm_mb=r["comm_mb"], fl_time_s=r.get("fl_time_s"))
    for k, v in r["cfg"].items():
        d["cfg_" + k] = v
    return d


def runs_df(experiment):
    rows = []
    for tag, method, seed in list_runs(experiment):
        for r in load_runs(experiment, tag, method):
            if r["seed"] == seed:
                rows.append(_flat(r))
    return pd.DataFrame(rows)


def summarize(df, group=("benchmark", "scenario", "method"), keys=None):
    """mean and std over seeds for every numeric metric."""
    if df.empty:
        return df
    keys = keys or [c for c in df.columns if c not in group and c != "seed" and not c.startswith("cfg_")
                    and pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for g, sub in df.groupby(list(group)):
        row = dict(zip(group, g if isinstance(g, tuple) else (g,)))
        row["n_seeds"] = sub["seed"].nunique()
        for k in keys:
            m, s = mean_std(sub[k].tolist())
            row[k], row[k + "_std"] = m, s
        rows.append(row)
    return pd.DataFrame(rows)


def table_vs_paper(experiment="main"):
    """One row per (benchmark, scenario, method, metric): ours (mean) vs paper, with the gap."""
    s = summarize(runs_df(experiment))
    rows = []
    for _, r in s.iterrows():
        ref = P.PAPER_MAIN.get(r["benchmark"], {}).get(r["scenario"], {}).get(r["method"], {})
        for k in SC_KEYS.get(r["scenario"], []):
            if k in r and pd.notna(r[k]):
                rows.append(dict(benchmark=r["benchmark"], scenario=r["scenario"], method=r["method"], metric=k,
                                 ours=r[k], ours_std=r[k + "_std"], paper=ref.get(k), n_seeds=r["n_seeds"]))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["delta"] = df["ours"] - df["paper"]
    return df


def table_main(experiment="main", style="md"):
    """Paper-style summary: for every scenario a table benchmark x method with 'mean +- std' (R-A/F-A/MIA or 0A/PS/MIA)."""
    s = summarize(runs_df(experiment))
    out = {}
    for sc, keys in SC_KEYS.items():
        sub = s[s["scenario"] == sc] if not s.empty else s
        if sub.empty:
            continue
        rows = []
        for (b, m), g in sub.groupby(["benchmark", "method"]):
            r = g.iloc[0]
            rows.append(dict(benchmark=b, method=m, **{k: f"{r[k]:.3f}±{r[k + '_std']:.3f}" for k in keys if k in r}))
        out[sc] = pd.DataFrame(rows)
    return out


def table_cost(experiment="main", scenario="client"):
    s = summarize(runs_df(experiment))
    if s.empty:
        return s
    s = s[s["scenario"] == scenario]
    rows = []
    for _, r in s.iterrows():
        ref = P.PAPER_COST.get(r["benchmark"], {}).get(r["method"], (None, None))
        rows.append(dict(benchmark=r["benchmark"], method=r["method"], time_s=r["time_s"], time_filter_s=r.get("time_filter_s"),
                         comm_mb=r["comm_mb"], paper_time_s=ref[0], paper_comm_mb=ref[1]))
    return pd.DataFrame(rows)


def table_non_target_loss(experiment="main"):
    """Utility kept: retained accuracy before vs after unlearning (the paper's 'non-target knowledge loss')."""
    s = summarize(runs_df(experiment))
    rows = []
    for _, r in s.iterrows():
        if r["scenario"] in ("client", "class") and "pre_R_A" in r:
            rows.append(dict(benchmark=r["benchmark"], scenario=r["scenario"], method=r["method"], R_A_before=r["pre_R_A"],
                             R_A_after=r["R_A"], loss=r["pre_R_A"] - r["R_A"]))
    return pd.DataFrame(rows)


def table_sweep(experiment, param, metrics=("R_A", "F_A", "MIA")):
    """Ablation experiments store the swept value in cfg_<param>."""
    df = runs_df(experiment)
    if df.empty:
        return df
    col = "cfg_" + param
    s = summarize(df, group=("benchmark", "scenario", "method", col))
    keep = ["benchmark", "scenario", "method", col, "n_seeds"] + [c for m in metrics for c in (m, m + "_std") if c in s.columns]
    return s[keep].sort_values(["benchmark", col])


def table_cross_protocol(experiment="cross"):
    """FuGuard-style backdoored-client protocol: ACC, ASR (strict and FuGuard's definition), MIA, time, comm."""
    s = summarize(runs_df(experiment))
    if s.empty:
        return s
    keep = ["benchmark", "method", "n_seeds"] + [c for k in ("ACC", "ASR", "ASR_incl_target", "MIA", "time_s", "comm_mb", "pre_ACC", "pre_ASR")
                                                for c in (k, k + "_std") if c in s.columns]
    return s[keep]
