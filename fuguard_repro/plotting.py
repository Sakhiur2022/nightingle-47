"""Figure scripts. Styles copy the images shipped in the repo's results/ folder:
 - recovery curves (Fig. 6 / Fig. 9): one panel per dataset x metric, rounds 1..5 on the x-axis, mean line + shaded
   mean±std band over seeds; ACC solid, ASR dashed, MIA dash-dot; FuGuard in red and thicker.
 - lambda_OT sensitivity (Fig. 10): ACC (blue circles) and ASR (orange squares) vs lambda_OT with value labels.
Fig. 3/4/5/7/8 are drawn from the paper's descriptions (no reference image was available)."""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .aggregate import curves, sensitivity_numeric, STD_DDOF, SENS_LAMBDAS  # noqa: E402
from .config import METHODS, METHOD_LABELS, IMAGE_DATASETS, lr_variant  # noqa: E402
from .io_results import load_runs  # noqa: E402
from .paths import FIGURES_DIR  # noqa: E402

COLORS = {"retrain": "tab:blue", "fedsga": "orange", "fu": "tab:green", "not": "tab:purple",
          "fastfedul": "tab:gray", "fuguard": "tab:red"}
LINESTYLE = {"acc": "-", "asr": "--", "mia": "-."}
YLABEL = {"acc": "ACC", "asr": "ASR", "mia": "MIA"}
TITLE = {"SVHN": "SVHN", "EuroSAT": "EuroSAT", "CIFAR10": "CIFAR10", "CIFAR100": "CIFAR100", "SVHN-100": "SVHN"}
FIGSIZE = (4.9, 3.8)
DPI = 150


def _save(fig, name, out_dir):
    out_dir = out_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def draw_recovery_panel(ax, experiment, tag, metric, methods=METHODS, title=None, legend=True):
    """Mean ± std (over the available seeds) of one recovery metric, rounds 1..R."""
    any_data, ymax = False, 0.0
    for m in methods:
        c = curves(experiment, tag, m, metric)
        if c.size == 0:
            continue
        any_data = True
        x = np.arange(1, c.shape[1] + 1)
        mu = c.mean(0)
        sd = c.std(0, ddof=STD_DDOF) if len(c) > 1 else np.zeros_like(mu)
        lw = 3.0 if m == "fuguard" else 2.0
        ax.plot(x, mu, LINESTYLE[metric], color=COLORS[m], lw=lw, label=METHOD_LABELS[m])
        ax.fill_between(x, mu - sd, mu + sd, color=COLORS[m], alpha=0.18, linewidth=0)
        ymax = max(ymax, float((mu + sd).max()))
        ax.set_xticks(x)
    ax.set_title(title or TITLE.get(tag, tag), fontsize=17)
    ax.set_xlabel("Round", fontsize=15)
    ax.set_ylabel(YLABEL[metric], fontsize=15)
    ax.tick_params(labelsize=13)
    if metric == "acc":
        ax.set_ylim(0, 1)
    elif metric == "mia":
        ax.set_ylim(0, max(0.5, ymax * 1.05))
    if legend and any_data:
        ax.legend(ncol=2, fontsize=10.5, loc="upper right" if metric == "mia" else "lower right", framealpha=0.9)
    return any_data


def save_recovery_figures(experiment, tag, out_dir=None, name_prefix=None, methods=METHODS):
    """Writes <prefix>-acc.png / -asr.png / -mia.png (e.g. cifar10-acc.png, svhn-100-acc.png)."""
    prefix = name_prefix or tag.lower()
    paths = []
    for metric in ("acc", "asr", "mia"):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ok = draw_recovery_panel(ax, experiment, tag, metric, methods)
        if not ok:
            plt.close(fig)
            continue
        paths.append(_save(fig, f"{prefix}-{metric}.png", out_dir))
    return paths


def fig6_grid(experiment="table2", datasets=IMAGE_DATASETS, out_dir=None, name="fig6_recovery_grid.png"):
    """Fig. 6: rows = datasets, columns = ACC / ASR / MIA."""
    fig, axes = plt.subplots(len(datasets), 3, figsize=(15, 3.9 * len(datasets)), squeeze=False)
    for i, ds in enumerate(datasets):
        for j, metric in enumerate(("acc", "asr", "mia")):
            draw_recovery_panel(axes[i][j], experiment, ds, metric, legend=(i == 0))
    fig.tight_layout()
    return _save(fig, name, out_dir)


def plot_ot_sensitivity(tag, n_rec=5, out_dir=None, name=None):
    """Fig. 10 panel (<dataset>-ot.png)."""
    num = sensitivity_numeric(tag, n_rec)
    if not num:
        return None
    lams = sorted(num)
    acc = [num[l][0] for l in lams]
    asr = [num[l][1] for l in lams]
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.plot(lams, acc, "-o", color="tab:blue", lw=2, label="ACC")
    ax.plot(lams, asr, "-s", color="tab:orange", lw=2, label="ASR")
    for l, a, s in zip(lams, acc, asr):
        ax.annotate(f"{a:.2f}", (l, a), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=10)
        ax.annotate(f"{s:.2f}", (l, s), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=10)
    ax.set_xticks(list(SENS_LAMBDAS))
    ax.set_xticklabels([f"{l:.2f}" for l in SENS_LAMBDAS])
    ax.set_xlabel(r"$ot$", fontsize=15)
    ax.set_ylabel("Recovery ACC/ASR", fontsize=14)
    ax.set_title(TITLE.get(tag, tag), fontsize=16)
    ax.grid(True)
    ax.legend(loc="center right")
    return _save(fig, name or f"{tag.lower()}-ot.png", out_dir)


# ---------------------------------------------------------------------------------- data figures
def plot_label_distribution(before, after, title, out_dir=None, name="fig3_label_distribution_svhn.png"):
    """Fig. 3: stacked label histogram per client before (top) and after (bottom) backdoor injection."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    cmap = plt.get_cmap("tab10") if before.shape[1] <= 10 else plt.get_cmap("tab20")
    for ax, counts, sub in zip(axes, (before, after), ("before injection", "after injection")):
        bottoms = np.zeros(len(counts))
        for c in range(counts.shape[1]):
            ax.bar(np.arange(len(counts)), counts[:, c], bottom=bottoms, color=cmap(c % cmap.N), label=str(c))
            bottoms += counts[:, c]
        ax.set_ylabel("samples")
        ax.set_title(f"{title} - {sub}")
    axes[1].set_xticks(np.arange(len(after)))
    axes[1].set_xticklabels([f"C{i}" for i in range(len(after))])
    axes[0].legend(title="class", ncol=5, fontsize=8)
    fig.tight_layout()
    return _save(fig, name, out_dir)


def plot_test_distribution(clean, poison, title, out_dir=None, name="fig4_test_distribution_svhn.png"):
    """Fig. 4: label counts of the clean vs the backdoor test set."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    k = np.arange(len(clean))
    ax.bar(k - 0.2, clean, 0.4, label="clean test", color="lightgreen")
    ax.bar(k + 0.2, poison, 0.4, label="backdoor test", color="salmon")
    ax.set_xlabel("class")
    ax.set_ylabel("samples")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", ls="--", alpha=0.6)
    return _save(fig, name, out_dir)


def plot_awe_trigger(clean_window, trig_window, out_dir=None, name="fig5_awe_trigger.png"):
    """Fig. 5: same AWE sample before / after trigger injection."""
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(clean_window, color="blue", label="clean")
    ax.plot(trig_window, color="red", ls="--", label="triggered")
    ax.set_xlabel("time step")
    ax.set_ylabel("amplitude")
    ax.legend()
    ax.grid(True)
    return _save(fig, name, out_dir)


# ---------------------------------------------------------------------------------- AWE (Fig. 7 / 8)
def plot_awe_training(tag="AWE", experiment="awe", variant="retrain", out_dir=None, name="fig7_awe_training.png"):
    """Fig. 7: clean ACC and backdoor ASR during the training phase (rounds 1..unlearn_round-1)."""
    runs = load_runs(experiment, tag, variant)
    if not runs:
        return None
    r = runs[0]
    n = r["unlearn_round"] - 1
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(range(1, n + 1), r["test_acc_per_round"][:n], color="tab:blue", label="ACC (clean)")
    ax.plot(range(1, n + 1), r["test_asr_per_round"][:n], color="tab:red", ls="--", label="ASR (backdoor)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ACC / ASR")
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend()
    return _save(fig, name, out_dir)


def plot_awe_recovery(lrs=(0.1, 0.05, 0.01), tag="AWE", experiment="awe", out_dir=None, name="fig8_awe_recovery.png"):
    """Fig. 8: recovery ACC (top) and ASR (bottom) for Retrain, FedSGA, FuGuard at three unlearning learning rates."""
    fig, axes = plt.subplots(2, len(lrs), figsize=(4.6 * len(lrs), 6.4), squeeze=False)
    any_data = False
    for j, lr in enumerate(lrs):
        for i, metric in enumerate(("acc", "asr")):
            ax = axes[i][j]
            for m, var in (("retrain", "retrain"), ("fedsga", lr_variant("fedsga", lr)), ("fuguard", lr_variant("fuguard", lr))):
                c = curves(experiment, tag, var, metric)
                if c.size == 0:
                    continue
                any_data = True
                x = np.arange(1, c.shape[1] + 1)
                ax.plot(x, c.mean(0), color=COLORS[m], lw=3 if m == "fuguard" else 2, label=METHOD_LABELS[m])
            ax.set_title(f"unlearning lr = {lr:g}")
            ax.set_xlabel("Recovery epoch")
            ax.set_ylabel(YLABEL[metric])
            ax.set_ylim(0, 1)
            ax.grid(True)
            if i == 0 and j == 0 and any_data:
                ax.legend()
    if not any_data:
        plt.close(fig)
        return None
    fig.tight_layout()
    return _save(fig, name, out_dir)
