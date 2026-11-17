# results/

* `raw/<experiment>/<benchmark>__<scenario>[__<sweep value>]/<method>/seed<k>.pkl` (+ `.json` summary): one file per run, written the moment the run finishes (a crashed session loses at most the run in progress).
* `tables/`: markdown + csv built by `python scripts/make_reports.py` (ours vs paper, cost, non-target loss, ablations, cross-protocol).
* `figures/`: png figures built by the same script.

Experiments: `main`, `ablation_dim`, `ablation_alpha`, `ablation_dp`, `ablation_centroid`, `cross`, `smoke`.
