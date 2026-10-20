"""One-call builders for every table and figure (used by notebooks/03_analysis and scripts/)."""
import pandas as pd

from .aggregate import (table2_numeric, table2_df, table2_vs_paper, table3_numeric, table3_df, table3_vs_paper,
                        table4_df, SENS_LAMBDAS, sensitivity_numeric)
from .config import IMAGE_DATASETS, ot_variant
from .paths import TABLES_DIR, FIGURES_DIR
from . import plotting


def df_to_markdown(df, index=True):
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def _save_df(df, name, index=True, out_dir=None):
    out_dir = out_dir or TABLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=index)
    (out_dir / f"{name}.md").write_text(df_to_markdown(df, index) + "\n")


def build_tables(n_rec=5, seed_table3=0, out_dir=None):
    """Writes results/tables/*.csv|md and returns the DataFrames."""
    out = {}
    n2 = table2_numeric(n_rec=n_rec)
    if n2:
        out["table2"], out["table2_vs_paper"] = table2_df(n2), table2_vs_paper(n2)
        _save_df(out["table2"], "table2_acc_asr", out_dir=out_dir)
        _save_df(out["table2_vs_paper"], "table2_vs_paper", index=False, out_dir=out_dir)
    n3 = table3_numeric(seed=seed_table3)
    if n3:
        out["table3"], out["table3_vs_paper"] = table3_df(n3), table3_vs_paper(n3)
        _save_df(out["table3"], "table3_recovery_rounds", out_dir=out_dir)
        _save_df(out["table3_vs_paper"], "table3_vs_paper", index=False, out_dir=out_dir)
    t4 = table4_df(n_rec=n_rec)
    if len(t4):
        out["table4"] = t4
        _save_df(t4, "table4_ablation_svhn", index=False, out_dir=out_dir)
    rows = []
    for ds in IMAGE_DATASETS:
        for lam, (a, s) in sorted(sensitivity_numeric(ds, n_rec).items()):
            rows.append({"dataset": ds, "lambda_OT": lam, "ACC": round(a, 4), "ASR": round(s, 4)})
    if rows:
        out["sensitivity"] = pd.DataFrame(rows)
        _save_df(out["sensitivity"], "fig10_sensitivity_values", index=False, out_dir=out_dir)
    return out


def build_figures(out_dir=None, n_rec=5):
    """Writes results/figures/*.png (names mirror the repo's results/ folder) and returns the paths."""
    paths = []
    for ds in IMAGE_DATASETS:
        paths += plotting.save_recovery_figures("table2", ds, out_dir)
        p = plotting.plot_ot_sensitivity(ds, n_rec, out_dir)
        if p:
            paths.append(p)
    paths += plotting.save_recovery_figures("scalability", "SVHN-100", out_dir, name_prefix="svhn-100")
    paths.append(plotting.fig6_grid("table2", IMAGE_DATASETS, out_dir))
    for fn in (plotting.plot_awe_training, plotting.plot_awe_recovery):
        p = fn(out_dir=out_dir)
        if p:
            paths.append(p)
    return paths
