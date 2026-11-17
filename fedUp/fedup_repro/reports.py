"""Write every table (markdown + csv) and figure from results/raw."""
from . import aggregate as A
from . import comparison as C
from . import plotting as PL
from .paths import TABLES_DIR


def _write(df, name):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    if df is None or len(df) == 0:
        return None
    df.to_csv(TABLES_DIR / f"{name}.csv", index=False)
    try:
        (TABLES_DIR / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=".4f"))
    except ImportError:                                   # tabulate not installed
        (TABLES_DIR / f"{name}.md").write_text(df.to_string(index=False))
    return df


def make_all(main="main", cross="cross", dim_exp="ablation_dim", alpha_exp="ablation_alpha", dp_exp="ablation_dp",
             centroid_exp="ablation_centroid"):
    out = {}
    for sc, df in A.table_main(main).items():
        out[f"main_{sc}"] = _write(df, f"table_main_{sc}")
    out["vs_paper"] = _write(A.table_vs_paper(main), "table_vs_paper")
    cost = A.table_cost(main)
    out["cost"] = _write(cost, "table_cost")
    ntl = A.table_non_target_loss(main)
    out["non_target_loss"] = _write(ntl, "table_non_target_loss")
    if cost is not None and len(cost):
        PL.plot_cost(cost.dropna(subset=["paper_time_s"]) if cost["paper_time_s"].notna().any() else cost)
    if ntl is not None and len(ntl):
        PL.plot_non_target_loss(ntl)
    for exp, param, fn in [(dim_exp, "out_dim", PL.plot_dim_sweep), (alpha_exp, "loss_alpha", PL.plot_alpha_sweep)]:
        sw = A.table_sweep(exp, param)
        if sw is not None and len(sw):
            _write(sw, f"table_{exp}")
            fn(sw)
    for exp, param in [(dp_exp, "dp_sigma"), (centroid_exp, "centroid_method")]:
        sw = A.table_sweep(exp, param)
        if sw is not None and len(sw):
            _write(sw, f"table_{exp}")
    cmp = C.fedup_vs_paper(main, dim_exp, alpha_exp, dp_exp, centroid_exp)
    for k, df in cmp.items():
        out[f"cmp_fedup_{k}"] = _write(df, f"table_cmp_fedup_{k}")
    if len(cmp["main"]):
        PL.plot_fedup_parity(cmp["main"])
    fv = C.fuguard_vs_fedup(cross)
    out["cmp_fuguard_vs_fedup"] = _write(fv, "table_cmp_fuguard_vs_fedup")
    _write(C.fuguard_vs_fedup_compact(cross), "table_cmp_fuguard_vs_fedup_compact")
    if "fedup_ACC" in fv.columns and fv["fedup_ACC"].notna().any():
        PL.plot_fuguard_vs_fedup(fv.dropna(subset=["fedup_ACC"]))
    cp = A.table_cross_protocol(cross)
    if cp is not None and len(cp):
        _write(cp, "table_cross_protocol")
        PL.plot_cross_protocol(cp)
    return out
