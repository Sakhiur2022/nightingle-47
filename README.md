# FuGuard reproduction (fixed, notebook-driven) - Colab edition

## Colab quick start
1. Upload `COLAB_START_HERE.ipynb` to Colab (or open it from Drive), set **Runtime → GPU**.
2. Run it top to bottom: it asks for `FuGuard_Reproduction_Colab.zip`, unzips to `/content/FuGuard_Reproduction`, installs the two missing packages
   (`requirements-colab.txt`; torch is already on Colab - do not reinstall it) and runs a synthetic smoke test.
3. Open the numbered notebooks from Colab's file browser. Each one finds the project at `/content` and re-installs the two packages
   if a runtime restart wiped them.
4. All data (`data/`), cached checkpoints (`artifacts/`) and results (`results/`) live on the Colab VM. It is wiped when the runtime
   is recycled: call `colab.zip_results()` (end of `COLAB_START_HERE` and `03b_figures`) to download them; `colab.backup_to_drive()` is an optional alternative.
   Finished runs are saved immediately, so after a disconnect you re-run the cell and it resumes.
Not tested on Colab itself (see `docs/FIXES_AND_DEVIATIONS.md`, verification status).

Goal: reproduce the paper's tables and figures from the official repo's code, with the repo's bugs fixed
(see `docs/FIXES_AND_DEVIATIONS.md` - **read the verification status there first**).

## Run order
| Step | Notebook | Produces |
|---|---|---|
| 0 | `00_setup/00_environment_and_paths` | folders, smoke test of all methods, VAE download check |
| 1 | `01_data/01a_prepare_image_datasets` | SVHN, EuroSAT, CIFAR-10/100 splits (+ 100-client SVHN) |
|   | `01_data/01b_prepare_awe_dataset` | AWE data (download separately) |
|   | `01_data/01c_data_figures_fig3_fig4_fig5` | Fig. 3, 4, 5 |
| 2 | `02_experiments/02a_table2_protocol_5_recovery_rounds` | raw data for **Table II, Fig. 6** |
|   | `02_experiments/02b_table3_protocol_50_recovery_rounds` | raw data for **Table III** |
|   | `02_experiments/02c_ablation_table4` | **Table IV** (lambda_OT = 0; lambda 0.1 comes from 02a) |
|   | `02_experiments/02d_sensitivity_fig10` | **Fig. 10** |
|   | `02_experiments/02e_scalability_100_clients_fig9` | **Fig. 9** |
|   | `02_experiments/02f_awe_real_dataset_fig7_fig8` | **Fig. 7, 8** |
| 3 | `03_analysis/03a_tables`, `03b_figures` | tables (ours vs paper) and figures in `results/` |

Start small: CIFAR-10, seed 0, FuGuard + FedSGA, then scale up. Everything is resumable.
CLI equivalents: `python scripts/run_experiment.py table2 --datasets CIFAR10 --seeds 0`, `python scripts/make_reports.py`.

## Layout
`fuguard_repro/` shared code (config, data, models, methods, FL engine, aggregation, plotting) · `notebooks/` entry points ·
`results/` raw runs, tables, figures · `data/` downloads and processed splits · `artifacts/checkpoints/` cached pre-unlearning models.
Colab: upload the folder, `pip install -r requirements.txt`, open notebooks from inside it.
