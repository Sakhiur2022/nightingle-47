"""Matplotlib figures from the aggregated tables (saved to results/figures)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import paper_reference as P  # noqa: E402
from .paths import FIGURES_DIR  # noqa: E402

COLORS = {"fedup": "#d62728", "retrain": "#1f77b4", "eraseclient": "#2ca02c", "federaser": "#9467bd", "exactfun": "#8c564b",
          "fused": "#ff7f0e"}


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    p = FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_cost(cost_df, name="cost_time_comm"):
    """Time and communication per method (log scale), ours as bars and the paper as black ticks."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (col, ref, title) in zip(axes, [("time_s", "paper_time_s", "Unlearning time (s)"),
                                            ("comm_mb", "paper_comm_mb", "Communication (MB)")]):
        d = cost_df.copy()
        d["label"] = d["benchmark"].str.split("-").str[0] + "\n" + d["method"]
        ax.bar(range(len(d)), d[col], color=[COLORS.get(m, "gray") for m in d["method"]])
        ax.scatter(range(len(d)), d[ref], marker="_", s=350, color="black", zorder=3, label="paper")
        ax.set_yscale("log")
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels(d["label"], rotation=70, fontsize=7)
        ax.set_title(title)
        ax.legend(fontsize=7)
    return _save(fig, name)


def plot_non_target_loss(df, name="non_target_loss"):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    d = df.copy()
    d["label"] = d["benchmark"].str.split("-").str[0] + "/" + d["scenario"] + "/" + d["method"]
    ax.bar(range(len(d)), d["loss"], color=[COLORS.get(m, "gray") for m in d["method"]])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d["label"], rotation=75, fontsize=6)
    ax.set_ylabel("R-A before - after (lower = less knowledge lost)")
    return _save(fig, name)


def plot_dim_sweep(sweep_df, name="ablation_filter_dim"):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for b, g in sweep_df.groupby("benchmark"):
        g = g.sort_values("cfg_out_dim")
        ax.plot(g["cfg_out_dim"], g["R_A"], marker="o", label=f"{b} (ours)")
        if b in P.PAPER_DIM["R_A"]:
            ax.plot(P.PAPER_DIM["dims"], P.PAPER_DIM["R_A"][b], ls="--", color="gray", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("filter bottleneck dimension")
    ax.set_ylabel("R-A")
    ax.legend(fontsize=6)
    return _save(fig, name)


def plot_alpha_sweep(sweep_df, name="ablation_loss_alpha"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for b, g in sweep_df.groupby("benchmark"):
        g = g.sort_values("cfg_loss_alpha")
        axes[0].plot(g["cfg_loss_alpha"], g["R_A"], marker="o", label=b)
        axes[1].plot(g["cfg_loss_alpha"], g["F_A"], marker="o", label=b)
    for ax, t in zip(axes, ["R-A", "F-A"]):
        ax.set_xlabel("alpha (weight of the cross-entropy)")
        ax.set_ylabel(t)
    axes[0].legend(fontsize=6)
    return _save(fig, name)


def plot_cross_protocol(df, name="cross_protocol_fuguard"):
    """ACC and ASR (strict and FuGuard-style) of every method on the backdoored-client protocol."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (col, t) in zip(axes, [("ACC", "clean ACC"), ("ASR", "ASR (non-target inputs)"),
                                   ("ASR_incl_target", "ASR (FuGuard definition)")]):
        if col not in df.columns:
            continue
        ax.bar(range(len(df)), df[col], yerr=df.get(col + "_std"), color=[COLORS.get(m, "gray") for m in df["method"]])
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["benchmark"].str.split("-").str[0] + "\n" + df["method"], rotation=60, fontsize=7)
        ax.set_title(t)
    return _save(fig, name)


def plot_fedup_parity(main_df, name="cmp_fedup_parity"):
    """Every (benchmark, scenario, method, metric) cell: our value against the FedUP paper's value (dashed = exact agreement)."""
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    markers = {"client": "o", "class": "s", "sample": "^"}
    for sc, g in main_df.groupby("scenario"):
        ax.scatter(g["paper"], g["ours"], s=22, alpha=0.8, marker=markers.get(sc, "o"), label=sc)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
    ax.set_xlabel("FedUP paper")
    ax.set_ylabel("our reproduction")
    ax.set_title(f"all cells (mean |gap| = {main_df['abs_delta'].mean():.3f})", fontsize=9)
    ax.legend(fontsize=7)
    return _save(fig, name)


def plot_fuguard_vs_fedup(df, name="cmp_fuguard_vs_fedup"):
    """ACC and ASR on the four FuGuard datasets: FuGuard (paper), FuGuard (our reproduction), FedUP (ours, FuGuard's protocol)."""
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(df))
    w = 0.26
    series = [("FuGuard (paper)", "fuguard_paper_%s", "#d62728"), ("FuGuard (our repro)", "fuguard_repro_%s", "#ff9896"),
              ("FedUP (ours)", "fedup_%s", "#1f77b4")]
    for ax, met, title in zip(axes, ("ACC", "ASR"), ("clean ACC (%)", "backdoor ASR (%), FuGuard's definition")):
        for k, (lab, pat, col) in enumerate(series):
            c = pat % met
            if c not in df.columns:
                continue
            err = df[c + "_std"] if (c + "_std") in df.columns else None
            ax.bar(x + (k - 1) * w, df[c], w, yerr=err, label=lab, color=col, capsize=2)
        if met == "ASR" and "fedup_ASR_strict" in df.columns:
            ax.scatter(x + w, df["fedup_ASR_strict"], marker="_", s=300, color="black", zorder=3, label="FedUP strict ASR")
        ax.plot(x, df[f"fedavg_paper_{met}"], "k*", ms=8, label="FedAvg before unlearning (paper)")
        ax.set_xticks(x)
        ax.set_xticklabels(df["dataset"])
        ax.set_title(title)
    axes[0].legend(fontsize=7)
    return _save(fig, name)
