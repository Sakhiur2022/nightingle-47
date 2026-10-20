# Fixes, deviations and open questions

## Verification status (read this first)
* **Executed:** syntax/pyflakes on all modules; all 12 notebooks validate (nbformat) and every code cell parses;
  `aggregate.py`, `plotting.py`, `reports.py`, `io_results.py` were run end-to-end on *fabricated* result files (layout/style check only).
* **NOT executed:** anything that needs PyTorch (data prep, FL engine, the six methods, smoke test) - torch could not be installed in the
  authoring sandbox (disk). The torch code is a careful port but **unrun**. Run `notebooks/00_setup/00_environment_and_paths.ipynb` first:
  its synthetic smoke test exercises all methods (+ AWE) in about a minute and will surface any slip.
* The AWE pipeline was never run on the real dataset; `colab.fetch_awe_dataset()` (git clone + link folders) is a guess at the repo layout.
* Colab edition additions: project found at `/content/FuGuard_Reproduction`, `requirements-colab.txt` (no torch reinstall),
  `torch.load(..., weights_only=False)` (newer torch defaults to True and would reject our own files), `colab.py` helpers,
  `scripts/prepare_data.py`. None of it has been run on a real Colab runtime.

## Bugs in the official repo that are fixed here
| # | Where | Problem | Fix |
|---|---|---|---|
| B1 | main_image/FuGuard.py | imports `save_results_1`, calls undefined `save_results` -> NameError after the whole run | engine saves via `io_results` |
| B2 | main_image/utils.save_results_1 | path `ot{lambda}/fuguard.pkl` has no seed -> seed 1 overwrites seed 0 | path = `<experiment>/<tag>/<variant>/seed<k>.pkl` |
| B3 | main_awe/utils.py, read_data_trigger.py | use `CONFIG["SAVE_FOLDER"]`, `["CLIENTS"]`, `["SEED"]` that config.py never defines -> KeyError | keys defined in `config.awe_config` |
| B4 | main_awe/*.py | `CONFIG.get("seed")` is None -> unseeded runs | `seed` is an explicit argument |
| B5 | main_awe/FedSGA.py | gradient ascent loops over `train_dataset` (another client's data) instead of `target_dataset` | ascent on the forgotten client's data (note: paper's AWE FedSGA numbers may have been produced with this bug) |
| B6 | main_awe/FuGuard.py | encodes `some_samples[:100]` but pairs with all labels -> size mismatch unless subset == 100 | all proxy samples are used |
| B7 | requirements.txt | torch 2.5.1 + torchvision 0.16.1, scikit-learn 1.1.3 + numpy 2.2.6 are incompatible pairs | unpinned compatible set |
| B8 | config / paths | `../FuGuard/...` relative paths depend on cwd and folder name; AWE config says 10 clients (data code: 6) | project-relative paths; 6 clients |

## Protocol decisions (your answers)
* **Two separate protocols.** `table2`: 50 + 5 recovery rounds (Table II, Fig. 6/9/10, Table IV). `table3`: 50 + 50 (Table III).
  The paper's text says 5 recovery rounds but Table III has values up to 50, so one run cannot serve both. The repo's `ROUNDS=60` matches neither.
* **Adam kept** for local training (the paper text says SGD). Switch with `optimizer="sgd"` in the config if you ever want the paper text.
* Unlearning fires in round 50 (repo `EXTERNAL_ALERTS`): FedAvg runs rounds 1-49, round 50 = unlearning (no aggregation), recovery = rounds 51..
  The "FedAvg" row of Table II is the accuracy after round 49. "Only unlearn" (Table IV) = metrics right after the unlearning step.

## Deliberate engine changes (behaviour-preserving, except RNG stream)
* Rounds 1-49 are identical for all methods, so they are trained once per (dataset, seed) and cached (`artifacts/checkpoints`).
  Clients no longer do a training pass in the unlearning round that the repo discarded.
  Consequence: after round 49 the random-number stream differs from the repo's scripts, so numbers match statistically, not bit-for-bit.
* Data kept on the GPU, mini-batching by `randperm`; datasets stored as tensors (the repo pickled dataset objects tied to module paths).

## Kept exactly as in the repo (flagging, not changing)
* Unlearning batch size 32 (paper: "same as training"); SD-VAE gets mean/std-normalised images and decoded proxies are clamped to [0,1].
* ASR counts all test images (target-class images are untriggered but count as "success" if predicted 9) -> ASR has a non-zero floor.
* MIA = Yeom loss-threshold attack, plotted value = `attack_acc` (median-member-loss threshold; members = forgotten client's data, non-members = clean test set).

## Unspecified in the paper (assumed - change in config.py)
* std over seeds: population std (ddof=0). 100-client SVHN: alpha=0.1, attacker client_3. AWE: seed 0, one run.
* Fig. 10's level (CIFAR-10 ACC~0.69/ASR~0.14 at lambda=0.1) differs from Table II (63.8/7.67): its setting is not described.
* Paper's Table II lists identical EuroSAT numbers for FedSGA and FU (74.61+-0.87 / 15.74+-4.02): possibly a typo, so do not expect both to match.
* Fig. 3/4/5/7/8 had no reference image; those plots follow the paper's descriptions.
