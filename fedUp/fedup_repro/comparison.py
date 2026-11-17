"""Our results vs the papers.

 fedup_vs_paper()     our FedUP reproduction against the FedUP paper: every table (main, cost, four ablations) with gaps,
                      a per-scenario summary and an automatic claim scoreboard.
 fuguard_vs_fedup()   FuGuard (numbers from the FuGuard paper, and from our own FuGuard reproduction) against FedUP run by us
                      under FuGuard's protocol (the `cross` experiment): ACC and ASR on the four FuGuard datasets.
All our fractions are shown in the paper's units (fractions for FedUP's tables, percent for FuGuard's)."""
import numpy as np
import pandas as pd

from . import aggregate as A
from . import paper_reference as P

TOL = (0.02, 0.05)                                 # |ours - paper| thresholds for fractions (R-A, F-A, MIA, 0A, PS)


# ------------------------------------------------------------------------------------------------ FedUP paper
def main_vs_paper(experiment="main"):
    df = A.table_vs_paper(experiment)
    if df.empty:
        return df
    df = df[df["paper"].notna()].copy()
    df["abs_delta"] = df["delta"].abs()
    for t in TOL:
        df[f"within_{t}"] = df["abs_delta"] <= t
    return df.sort_values(["scenario", "benchmark", "method", "metric"]).reset_index(drop=True)


def summary(main_df):
    """Per scenario and metric: how many cells, mean / median / max |gap| and the share within the tolerances."""
    if main_df is None or main_df.empty:
        return pd.DataFrame()
    g = main_df.groupby(["scenario", "metric"])["abs_delta"]
    s = pd.DataFrame({"cells": g.size(), "mean_abs_gap": g.mean(), "median_abs_gap": g.median(), "max_abs_gap": g.max()})
    for t in TOL:
        s[f"share_within_{t}"] = main_df.groupby(["scenario", "metric"])[f"within_{t}"].mean()
    tot = pd.DataFrame([{"scenario": "ALL", "metric": "all", "cells": len(main_df), "mean_abs_gap": main_df["abs_delta"].mean(),
                         "median_abs_gap": main_df["abs_delta"].median(), "max_abs_gap": main_df["abs_delta"].max(),
                         **{f"share_within_{t}": main_df[f"within_{t}"].mean() for t in TOL}}])
    return pd.concat([s.reset_index(), tot], ignore_index=True)


def cost_vs_paper(experiment="main"):
    """Time and communication, ours vs the paper (CIFAR-10 ResNet-18 and AG News client unlearning), with the FedUP / Retrain ratios."""
    c = A.table_cost(experiment)
    if c.empty:
        return c
    c = c[c["paper_time_s"].notna()].copy()
    rows = []
    for b, g in c.groupby("benchmark"):
        d = g.set_index("method")
        for m, r in d.iterrows():
            row = dict(benchmark=b, method=m, ours_time_s=r["time_s"], paper_time_s=r["paper_time_s"], ours_comm_mb=r["comm_mb"],
                       paper_comm_mb=r["paper_comm_mb"])
            if "retrain" in d.index:
                row["ours_time_vs_retrain"] = r["time_s"] / d.loc["retrain", "time_s"]
                row["paper_time_vs_retrain"] = r["paper_time_s"] / d.loc["retrain", "paper_time_s"]
                row["ours_comm_vs_retrain"] = r["comm_mb"] / d.loc["retrain", "comm_mb"]
                row["paper_comm_vs_retrain"] = r["paper_comm_mb"] / d.loc["retrain", "paper_comm_mb"]
            rows.append(row)
    return pd.DataFrame(rows)


def dim_vs_paper(experiment="ablation_dim"):
    s = A.table_sweep(experiment, "out_dim")
    if s.empty:
        return s
    rows = []
    for _, r in s.iterrows():
        d = int(r["cfg_out_dim"])
        pr = P.PAPER_DIM["R_A"].get(r["benchmark"])
        paper = pr[P.PAPER_DIM["dims"].index(d)] if (pr and d in P.PAPER_DIM["dims"]) else np.nan
        mem = P.PAPER_DIM["memory_kb"].get(r["benchmark"], [None] * 5)[P.PAPER_DIM["dims"].index(d)] if d in P.PAPER_DIM["dims"] else None
        rows.append(dict(benchmark=r["benchmark"], out_dim=d, ours_R_A=r["R_A"], paper_R_A=paper, delta=r["R_A"] - paper, paper_memory_kb=mem))
    return pd.DataFrame(rows)


def alpha_vs_paper(experiment="ablation_alpha"):
    s = A.table_sweep(experiment, "loss_alpha")
    if s.empty:
        return s
    rows = []
    for _, r in s.iterrows():
        a = float(r["cfg_loss_alpha"])
        pa = P.PAPER_ALPHA.get(r["benchmark"])
        pr, pf = pa[P.PAPER_ALPHA["alphas"].index(a)] if (pa and a in P.PAPER_ALPHA["alphas"]) else (np.nan, np.nan)
        rows.append(dict(benchmark=r["benchmark"], alpha=a, ours_R_A=r["R_A"], paper_R_A=pr, delta_R_A=r["R_A"] - pr,
                         ours_F_A=r["F_A"], paper_F_A=pf))
    return pd.DataFrame(rows)


def dp_vs_paper(experiment="ablation_dp"):
    """'w/o DP' = sigma 0; 'with DP' = the paper's sigma for the benchmark."""
    s = A.table_sweep(experiment, "dp_sigma")
    if s.empty:
        return s
    rows = []
    for b, g in s.groupby("benchmark"):
        sig = P_SIGMA(b)
        for label, val in (("with DP", sig), ("w/o DP", 0.0)):
            m = g[np.isclose(g["cfg_dp_sigma"].astype(float), val)]
            if m.empty or b not in P.PAPER_DP:
                continue
            k = 0 if label == "with DP" else 1
            r = m.iloc[0]
            for met in ("R_A", "F_A", "MIA"):
                rows.append(dict(benchmark=b, setting=label, sigma=val, metric=met, ours=r[met], paper=P.PAPER_DP[b][met][k],
                                 delta=r[met] - P.PAPER_DP[b][met][k]))
    return pd.DataFrame(rows)


def P_SIGMA(benchmark):
    from .config import PAPER_DP_SIGMA
    return PAPER_DP_SIGMA.get(benchmark, 0.001)


def centroid_vs_paper(experiment="ablation_centroid"):
    s = A.table_sweep(experiment, "centroid_method")
    if s.empty:
        return s
    rows = []
    for _, r in s.iterrows():
        ref = P.PAPER_CENTROID.get(r["benchmark"])
        k = 0 if r["cfg_centroid_method"] == "none" else 1
        for met in ("R_A", "F_A"):
            rows.append(dict(benchmark=r["benchmark"], input="raw features" if k == 0 else "KMeans centroids", metric=met,
                             ours=r[met], paper=(ref[met][k] if ref else np.nan), delta=r[met] - (ref[met][k] if ref else np.nan)))
    return pd.DataFrame(rows)


def claims_scoreboard(main="main", dim="ablation_dim", dp="ablation_dp"):
    """Claims of the FedUP paper checked automatically against our runs. verdict: yes / no / n/a (not enough runs)."""
    rows = []
    s = A.summarize(A.runs_df(main))

    def get(b, sc, m, col):
        if s.empty:
            return None
        r = s[(s["benchmark"] == b) & (s["scenario"] == sc) & (s["method"] == m)]
        return None if r.empty or col not in r else float(r.iloc[0][col])

    def add(claim, paper, ours, ok):
        rows.append(dict(claim=claim, paper_says=paper, ours=ours, verdict=("n/a" if ok is None else ("yes" if ok else "no"))))

    # 1 unlearning time << retraining
    rs = [(b, get(b, "client", "fedup", "time_s"), get(b, "client", "retrain", "time_s")) for b in sorted(set(s["benchmark"]))] if not s.empty else []
    rs = [(b, f, r) for b, f, r in rs if f and r]
    add("FedUP unlearning time is below 1 % of Retrain (client)", "0.68 % (CIFAR-10), 0.75 % (AG News)",
        "; ".join(f"{b}: {100 * f / r:.2f} %" for b, f, r in rs) or "no runs", (all(f / r < 0.01 for _, f, r in rs) if rs else None))
    # 2 communication << retraining
    rc = [(b, get(b, "client", "fedup", "comm_mb"), get(b, "client", "retrain", "comm_mb")) for b in sorted(set(s["benchmark"]))] if not s.empty else []
    rc = [(b, f, r) for b, f, r in rc if f and r]
    add("FedUP communication is below 5 % of Retrain (client)", "0.7 % (CIFAR-10), 1.3 % (AG News)",
        "; ".join(f"{b}: {100 * f / r:.2f} %" for b, f, r in rc) or "no runs", (all(f / r < 0.05 for _, f, r in rc) if rc else None))
    # 3 class unlearning forgets completely
    fa = [(b, get(b, "class", "fedup", "F_A")) for b in sorted(set(s["benchmark"]))] if not s.empty else []
    fa = [(b, v) for b, v in fa if v is not None]
    add("Class unlearning: F-A is ~0 on every benchmark", "0.00 everywhere", "; ".join(f"{b}: {v:.3f}" for b, v in fa) or "no runs",
        (all(v < 0.02 for _, v in fa) if fa else None))
    # 4 client R-A close to Retrain
    gaps = []
    for b in sorted(set(s["benchmark"])) if not s.empty else []:
        a, r = get(b, "client", "fedup", "R_A"), get(b, "client", "retrain", "R_A")
        if a is not None and r is not None:
            gaps.append((b, a - r))
    add("Client unlearning: FedUP R-A is within 0.03 of Retrain", "gap <= 0.02 on all benchmarks",
        "; ".join(f"{b}: {g:+.3f}" for b, g in gaps) or "no runs", (all(abs(g) <= 0.03 for _, g in gaps) if gaps else None))
    # 5 filter dimension 32 is (near) best
    d = A.table_sweep(dim, "out_dim")
    if not d.empty:
        ok = []
        for b, g in d.groupby("benchmark"):
            r32 = g[g["cfg_out_dim"].astype(int) == 32]["R_A"]
            if len(r32):
                ok.append((b, float(r32.iloc[0]) >= float(g["R_A"].max()) - 0.02))
        add("Bottleneck 32 is within 0.02 R-A of the best dimension", "32 is best or tied", "; ".join(f"{b}: {'yes' if o else 'no'}" for b, o in ok),
            all(o for _, o in ok) if ok else None)
    else:
        add("Bottleneck 32 is within 0.02 R-A of the best dimension", "32 is best or tied", "no runs", None)
    # 6 DP lowers MIA
    dd = A.table_sweep(dp, "dp_sigma")
    if not dd.empty:
        out = []
        for b, g in dd.groupby("benchmark"):
            w, wo = g[np.isclose(g["cfg_dp_sigma"].astype(float), P_SIGMA(b))], g[np.isclose(g["cfg_dp_sigma"].astype(float), 0.0)]
            if len(w) and len(wo) and "MIA" in g:
                out.append((b, float(w.iloc[0]["MIA"]) < float(wo.iloc[0]["MIA"])))
        add("DP noise lowers the membership-inference accuracy", "CIFAR-10 0.64 -> 0.52, AG News 0.62 -> 0.57",
            "; ".join(f"{b}: {'lower' if o else 'not lower'}" for b, o in out), all(o for _, o in out) if out else None)
    else:
        add("DP noise lowers the membership-inference accuracy", "CIFAR-10 0.64 -> 0.52, AG News 0.62 -> 0.57", "no runs", None)
    return pd.DataFrame(rows)


def fedup_vs_paper(main="main", dim="ablation_dim", alpha="ablation_alpha", dp="ablation_dp", centroid="ablation_centroid"):
    m = main_vs_paper(main)
    return dict(main=m, summary=summary(m), cost=cost_vs_paper(main), dim=dim_vs_paper(dim), alpha=alpha_vs_paper(alpha),
                dp=dp_vs_paper(dp), centroid=centroid_vs_paper(centroid), claims=claims_scoreboard(main, dim, dp))


# ------------------------------------------------------------------------------------------------ FuGuard vs FedUP
def fuguard_vs_fedup(cross="cross"):
    """One row per FuGuard dataset (percent). Columns: FedAvg and FuGuard from the FuGuard paper, FuGuard from our reproduction,
    and FedUP from our `cross` runs (ACC, ASR in FuGuard's definition, and the strict ASR on non-target inputs only)."""
    df = A.summarize(A.runs_df(cross))
    rows = []
    for bench, ds in P.FUGUARD_DATASETS.items():
        r = dict(dataset=ds, benchmark=bench)
        fp, fr = P.FUGUARD_PAPER[ds], P.FUGUARD_REPRO[ds]
        r.update(fedavg_paper_ACC=fp["fedavg"][0], fedavg_paper_ASR=fp["fedavg"][2],
                 fuguard_paper_ACC=fp["fuguard"][0], fuguard_paper_ACC_std=fp["fuguard"][1],
                 fuguard_paper_ASR=fp["fuguard"][2], fuguard_paper_ASR_std=fp["fuguard"][3],
                 fuguard_repro_ACC=fr["fuguard"][0], fuguard_repro_ACC_std=fr["fuguard"][1],
                 fuguard_repro_ASR=fr["fuguard"][2], fuguard_repro_ASR_std=fr["fuguard"][3])
        if not df.empty:
            sub = df[(df["benchmark"] == bench) & (df["method"] == "fedup")]
            if len(sub):
                x = sub.iloc[0]
                r.update(fedavg_ours_ACC=100 * x.get("pre_ACC", np.nan), fedavg_ours_ASR=100 * x.get("pre_ASR_incl_target", np.nan),
                         fedup_ACC=100 * x["ACC"], fedup_ACC_std=100 * x["ACC_std"],
                         fedup_ASR=100 * x["ASR_incl_target"], fedup_ASR_std=100 * x["ASR_incl_target_std"],
                         fedup_ASR_strict=100 * x["ASR"], n_seeds=int(x["n_seeds"]))
        rows.append(r)
    out = pd.DataFrame(rows)
    if "fedup_ACC" in out:
        out["dACC_fedup_minus_fuguard_paper"] = out["fedup_ACC"] - out["fuguard_paper_ACC"]
        out["dASR_fedup_minus_fuguard_paper"] = out["fedup_ASR"] - out["fuguard_paper_ASR"]
        out["lower_ASR"] = np.where(out["fedup_ASR"].isna(), "n/a", np.where(out["fedup_ASR"] < out["fuguard_paper_ASR"], "FedUP", "FuGuard"))
        out["higher_ACC"] = np.where(out["fedup_ACC"].isna(), "n/a", np.where(out["fedup_ACC"] > out["fuguard_paper_ACC"], "FedUP", "FuGuard"))
    return out


def fuguard_vs_fedup_compact(cross="cross"):
    """The same comparison as a short 'mean ± std' table for slides / the report."""
    t = fuguard_vs_fedup(cross)
    f = lambda m, s: "n/a" if pd.isna(m) else f"{m:.2f}±{s:.2f}"  # noqa: E731
    return pd.DataFrame([{
        "dataset": r.dataset,
        "FedAvg ACC / ASR (paper)": f"{r.fedavg_paper_ACC:.2f} / {r.fedavg_paper_ASR:.2f}",
        "FuGuard ACC (paper)": f(r.fuguard_paper_ACC, r.fuguard_paper_ACC_std), "FuGuard ASR (paper)": f(r.fuguard_paper_ASR, r.fuguard_paper_ASR_std),
        "FuGuard ACC (our repro)": f(r.fuguard_repro_ACC, r.fuguard_repro_ACC_std), "FuGuard ASR (our repro)": f(r.fuguard_repro_ASR, r.fuguard_repro_ASR_std),
        "FedUP ACC (ours)": f(getattr(r, "fedup_ACC", np.nan), getattr(r, "fedup_ACC_std", np.nan)),
        "FedUP ASR (ours)": f(getattr(r, "fedup_ASR", np.nan), getattr(r, "fedup_ASR_std", np.nan)),
        "FedUP ASR strict": ("n/a" if pd.isna(getattr(r, "fedup_ASR_strict", np.nan)) else f"{r.fedup_ASR_strict:.2f}")}
        for r in t.itertuples()])
