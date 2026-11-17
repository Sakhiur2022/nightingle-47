# FedUP reproduction

One-shot federated unlearning with centroid-guided plug-in filters: clients upload DP class centroids once, the server trains a tiny
bottleneck **filter** between the frozen feature extractor and the classifier. Same structure and conventions as the FuGuard reproduction,
plus explicit `data/` and `artifacts/` folders.

```
FedUP_Reproduction/
├── README.md  run_all.py  requirements*.txt  COLAB_START_HERE.ipynb
├── docs/FIXES_AND_DEVIATIONS.md      paper text vs released code vs this harness, verification status  <- read first
├── data/        raw/ (datasets)  processed/ (federated splits per seed)
├── artifacts/   checkpoints/ (FedAvg + Retrain models)  filters/ (trained FedUP filters)  features/
├── results/     raw/ (one pkl per run)  tables/  figures/
├── logs/
├── fedup_repro/
│   config.py        benchmarks, scenarios, presets (repo / paper / fuguard), every deviation as one key
│   data.py          Dirichlet split, proxy hold-out, attacks (label flip, class relabel, backdoor), triggers, batches
│   models.py        LeNet-5, ResNet-18/34, ViT, BERT-tiny, ConvNet + the Filter
│   fl.py            FedAvg engine + cached pre-unlearning checkpoint
│   features.py      client side: features, KMeans centroids, DP noise
│   filter.py        server side: filter training (alpha*CE + (1-alpha)*MSE)
│   unlearn.py       fedup / retrain / eraseclient (+ stubs for federaser, exactfun, fused)
│   metrics.py mia.py runner.py experiments.py multigpu.py
│   aggregate.py paper_reference.py plotting.py reports.py smoke.py colab.py
├── scripts/     download_datasets.py prepare_data.py run_experiment.py make_reports.py
└── notebooks/   00_setup  01_data  02_experiments  03_analysis
```

## Quick start
```bash
pip install -r requirements.txt          # Kaggle / Colab: requirements-colab.txt
python -m fedup_repro.smoke              # 1 minute, CPU, synthetic data: checks the whole chain
python scripts/download_datasets.py      # MNIST, CIFAR-10/100, AG News -> data/raw
python scripts/run_experiment.py main --benchmarks MNIST-LeNet5 --seeds 0   # one real benchmark first
python scripts/run_experiment.py main --seeds 0 1                           # the full main grid (all GPUs)
python scripts/make_reports.py                                              # tables + figures
```
Kaggle / Colab: upload the folder and open `COLAB_START_HERE.ipynb`. **Every section is atomic**: after a restart run only *0. Setup*, then any single cell
(one cell per dataset, two per benchmark in the main grid: *FedUP + EraseClient*, then *Retrain*; one per ablation); finished runs are skipped, so any cell can be re-run.

Knobs (Setup cell, or `--set key=value` on any command, repeatable): `rounds_per_epoch`, `local_epochs`, `filter_epochs`, `num_clients`, ...
```bash
python scripts/run_experiment.py main --benchmarks CIFAR10-ResNet18 --set rounds_per_epoch=1 --set local_epochs=3
```
**Rounds.** The released code runs two FedAvg rounds per "global epoch", i.e. **100 rounds** for the README's 50. The library default (`repo` preset) keeps that;
the start notebook sets `rounds_per_epoch=1` (50 rounds, half the cost) in its Setup cell. Changing it retrains the cached FedAvg models.

**Progress line.** `[fl] round r/R held-out acc ...` is only a progress number: accuracy of the global model on held-out data (client scenario: the retained clients'
own test splits; other scenarios: 2000 images of the official test set). The reported metrics are computed after unlearning.

## Datasets live in the repo (no re-download)
Raw datasets are stored in `data/raw` **inside the repo** as arrays (`data/raw/<name>/arrays.npz`; AG News as `data/raw/agnews/<model>_L128.npz`). Lookup order: cached arrays,
the read-only `data/raw` that ships with the code (on Kaggle: `/kaggle/input/datasets/<user>/<dataset>/FedUP_Reproduction/data/raw`, linked in, not copied), torchvision files already on disk,
the Hugging Face hub (much faster than torchvision's Toronto mirror), a personal HF mirror, and only then torchvision's own download.
Run the download cells once, then notebook section 8d zips `data/`; put its content into your Kaggle dataset and every later session starts with the data already there.
AG News is tokenised once with the text model's tokenizer; if the repo's tokenizer cannot be converted to a fast tokenizer (the *sentencepiece / tiktoken* error) the loader falls back to the slow BERT WordPiece tokenizer and finally to `bert-base-uncased` (same uncased vocabulary as BERT-tiny).

## Backdoor explorer
`from fedup_repro import viz; viz.backdoor_explorer()` (notebook section 2b, or `notebooks/01_data/01c_backdoor_explorer.ipynb`): choose a dataset (MNIST, CIFAR-10, CIFAR-100, AG News) and a class, press *random image* and see the sample before and after the trigger
(FedUP's single pixel as released or as a real white pixel; FuGuard's k x k patch), with a corner zoom and a difference map. The after-image is produced by the training code path itself.

## Comparison with the papers
Two comparisons, both built by `python scripts/make_reports.py` (tables in `results/tables/table_cmp_*`, figures `cmp_*.png`) and shown in notebook section **7b** / `notebooks/03_analysis/03c_compare_with_papers.ipynb`:
* **FedUP: ours vs the FedUP paper** (`fedup_repro.comparison.fedup_vs_paper()`): every main-table cell with its gap and tolerance flags, a per-scenario summary, time and communication with the FedUP/Retrain ratios, the four ablations, and an automatic claim scoreboard.
* **FuGuard vs FedUP on the FuGuard paper's protocol** (`fuguard_vs_fedup()`): FuGuard's numbers from the FuGuard paper (and from our own FuGuard reproduction) next to FedUP run by us under that protocol, on SVHN, EuroSAT, CIFAR-10 and CIFAR-100 (`cross` experiment, `fuguard` preset: client 3 backdoored with FuGuard's trigger constants, ConvNet, Adam, 49 rounds, all data). ASR is reported in FuGuard's definition and strictly; FedUP has no recovery rounds and MIA is not compared (see the notes in the notebook).

## What each experiment produces
| experiment | scenario / methods | paper item | output |
|---|---|---|---|
| `main` | client, class, sample x fedup, retrain, eraseclient x 5 benchmarks | main results table, cost table, non-target knowledge loss | `table_main_*`, `table_vs_paper`, `table_cost`, `table_non_target_loss` |
| `ablation_dim` | client, fedup, dim 4..64 | filter-dimension table | `table_ablation_dim`, figure |
| `ablation_alpha` | class, fedup, alpha 0.1..0.9 | loss-weight table | `table_ablation_alpha`, figure |
| `ablation_dp` | client, fedup, sigma 0..0.1 | DP table | `table_ablation_dp` |
| `ablation_centroid` | client, fedup, features vs centroids | centroid table | `table_ablation_centroid` |
| `cross` | client_backdoor (FuGuard protocol) on SVHN / EuroSAT / CIFAR-10 / CIFAR-100 (ConvNet) x fedup (+ retrain, eraseclient) | **not in the paper**: FedUP under FuGuard's backdoored-client protocol | `table_cross_protocol`, `table_cmp_fuguard_vs_fedup*`, figures |

Presets: `--preset repo` (default, the released code), `--preset paper` (the paper's text), `--preset fuguard` (used by `cross`). `--quick` shrinks everything for a pipeline check.

## Multi-GPU (Kaggle T4 x2)
`run_experiment.py` starts one worker process per visible GPU (`--gpus N` to override). Jobs are grouped by (benchmark, seed) so every FedAvg checkpoint is written by exactly one worker; logs go to `logs/<experiment>_gpu<k>.log`.

## Honest scope
Four of the paper's five baselines are not in the released code. Of the six methods in its tables, FedUP and Retrain are implemented, EraseClient is a port, and three are documented stubs (see `docs/FIXES_AND_DEVIATIONS.md`). The synthetic smoke test passes, but real-data runs and agreement with the paper have **not** been checked.
