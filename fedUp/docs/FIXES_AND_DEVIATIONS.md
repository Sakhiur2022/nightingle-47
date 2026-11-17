# FedUP reproduction: fixes, deviations and verification status

Sources: the paper's LaTeX source (main text only; the supplement was **not** available) and the authors' released code
(`FedUP-code-main`). The harness follows the same rule as the FuGuard reproduction: **the default preset (`repo`) does what the
released code does** (that is what produced the published numbers); the preset `paper` switches every disagreement to the
paper's wording; each row below is one key in `fedup_repro/config.py`, so it can also be flipped alone through `overrides`.

## 0. Verification status (read this first)

| what | status |
|---|---|
| all modules compile; the pipeline runs end to end on **synthetic** data for all four scenarios and the three implemented methods (`python -m fedup_repro.smoke`, CPU, about one minute) | **run** |
| Dirichlet partition (covers every sample once, disjoint clients), proxy / train / test split, triggers, label flipping, class relabelling, model output shapes (LeNet, ResNet-18/34, ViT, ConvNet), filter size (128 KB at 512-d, dim 32), tables, plots, two-worker launch | **run** |
| the text path (token tensors, text trigger, AdamW) with a *fake* encoder | **run** |
| real datasets, ImageNet / BERT-tiny weight downloads, GPU / fp16, the full-size grid, the multi-GPU launch on real GPUs | **not run** (no GPU, no network access to the data hosts in the authoring environment) |
| agreement with the paper's numbers | **unknown until you run it** |

Run the smoke test first; then one real benchmark (`MNIST-LeNet5`) before the heavy ones.

## 1. Paper text vs released code vs this harness

| # | item | paper | released code | harness (`repo` / `paper`) |
|---|---|---|---|---|
| 1 | Dirichlet alpha | 0.5 | argparse default 1.0 (README quick start does not set it) | 0.5 / 0.5 |
| 2 | number of clients | not in the main text | argparse default `num_user = 1` | **assumed 10** (flagged) |
| 3 | FedAvg rounds | W rounds, value not stated | quick start `global_epoch 50`, but `train_normal` calls `global_train_once` **twice per epoch** = 100 rounds | `rounds_per_epoch` 2 / 1 |
| 4 | aggregation | weighted by data size | plain (uniform) mean of client models | `uniform` / `size` |
| 5 | ResNet initialisation | not mentioned | torchvision ResNets with `pretrained=True` (ImageNet, 32x32 inputs) | `pretrained` True / False |
| 6 | CIFAR augmentation | not mentioned | random crop + flip applied **once** to every image when the dataset is materialised, **train and test** | `augment` static / none |
| 7 | train / test splits | not stated | client scenario: train+test merged, 70/30 per client; class / sample: 99/1 per client | same (class / sample are evaluated on the official test set; the repo's 1 % splits are too small to be meaningful) |
| 8 | differential privacy | Gaussian noise, sigma 0.001 (0.01 for CIFAR-100) for epsilon about 10 | off by default (`diff_privacy_scale = perturbation = 0`) | `dp_sigma` 0 / paper values |
| 9 | text model | "TinyBERT" | loads a local `bert-tiny-local` (hard-coded `/root/autodl-tmp/...` path) | `prajjwal1/bert-tiny` (see 20) |
| 10 | text optimiser | not stated | AdamW with lr 0.005 (very high for a BERT; kept) | `adamw`, 0.005 |
| 11 | sample trigger | "fixed trigger", target class 0 | writes **255 into the normalised tensor** at pixel (-1,-1); text: token id 1996 appended | `trigger_mode` repo / white |
| 12 | what a filter batch is | not stated | batches are formed once (shuffled once) and replayed in the same order every epoch; batch 10; 50 epochs; plain SGD with the dataset lr | `filter_reshuffle` False |
| 13 | filter output | "ReLU" | the adapter ends with a ReLU (clips negative features of the ViT / BERT extractors) | `filter_out_relu` True |
| 14 | time metric | "unlearning time" | sums the filter-training epochs only | we report `time_filter_s` (= repo), `time_client_s` (max over clients, parallel) and `time_s` (their sum) |
| 15 | communication | in MB | not computed in the release | float32 centroid upload (Retrain: 2 x model x clients x rounds) |
| 16 | class unlearning | forgotten class remapped to other classes | features of the forgotten class are relabelled to a random other class, one draw per (client, class) | same; forgotten class **assumed 0** (repo default: empty) |
| 17 | feature extraction | clients extract features | from the (already augmented) training arrays | eval mode on the same arrays |
| 18 | R-A / F-A | accuracies | mean over (client, class) cells of per-class accuracies | `r_a_style` class_mean / `pooled` available |
| 19 | MIA | "attack on forgotten vs retained" | shadow model + MLP on logits; **the function returns nothing**, so the exact reported number cannot be recovered | our version (balanced 1:1, shadow = original model fine-tuned on retained proxy data); compare methods with each other, not with the paper's absolute MIA |

### Data access (this harness)
* The released code downloads through torchvision (slow Toronto mirror) and loads a local `bert-tiny-local`; here datasets are cached as arrays in `data/raw`, bundled copies are linked in, and the Hugging Face hub is tried before torchvision's download.
* Tokenizer: AG News falls back from a fast tokenizer to the slow `BertTokenizer` to `bert-base-uncased` when the repo's tokenizer cannot be converted (the *sentencepiece / tiktoken* error). The trigger token id 1996 (`the`) is the same in all of them (uncased BERT vocabulary).
* **Not run in the authoring environment**: the Hugging Face and torchvision downloads (no network access to the data hosts); the bundled-data linking, the tokenizer fallback chain (with a fake `transformers`), the Kaggle folder layout and the explorer widgets were tested.

### The progress line and the round count
* `[fl] round r/R held-out acc x` is a monitoring number printed during FedAvg. It is computed on held-out data only: in the **client** scenario the train and test sets are
  merged and partitioned, so the official test set is part of the training pool; the monitor therefore uses the retained clients' own test splits (an earlier build
  monitored on the official test set, i.e. on training data, and was corrected). It never enters a table.
* R = `global_epochs * rounds_per_epoch`. Released code: 50 x 2 = **100** (row 3). The start notebook overrides this to 50 rounds.

### Evidence worth knowing
* **Feature dimensions (20).** The paper's filter-memory column implies the feature sizes: 128 KB at bottleneck 32 means a 512-d feature
  (LeNet `fc2`, ResNet avg-pool, ViT dim 512), and **32 KB for AG News means a 128-d feature**: BERT-tiny (hidden 128), not TinyBERT-312 (which would give 78 KB).
* **Main table probably without DP.** CIFAR-10 client MIA is 0.63 in the main table and 0.6373 "w/o DP" in the DP ablation.
* **0A / PS (sample scenario).** My reading of `test_backdoor_forget`: the whole test set is triggered but keeps its original labels;
  0A = recall of class 0, PS = precision of class 0. Under attack every triggered input goes to class 0, so **0A is maximal when the backdoor works**
  and cannot show removal; **PS drops when the backdoor works** and is the informative one. We also report `ASR0` (non-class-0 triggered inputs predicted as class 0).
* **ablation scenarios.** The alpha ablation values match the class scenario (F-A = 0); the dimension, DP and centroid ablations match the client scenario.

## 2. Baselines
The released code contains FedUP (class `LoraFU`, a historical name: it is a bottleneck adapter) and Retrain. **EraseClient, FedEraser, Exact-Fun and FUSED
are not released**, although the paper compares against them.
* `retrain`: implemented.
* `eraseclient`: **ported from the FuGuard reproduction's FU baseline** (reference model + projected gradient ascent, then `ec_recovery_rounds` FedAvg rounds), client scenarios only. It is not Halimi et al.'s code, so its numbers need not match the paper's E-C column.
* `federaser`, `exactfun`, `fused`: **stubs** raising `NotImplementedError` with instructions. Implement them against their original papers and register them in `fedup_repro/unlearn.py::REGISTRY`.

## 3. Extensions beyond the paper
* **`client_backdoor` scenario + `fuguard` preset + the ConvNet benchmarks** (`SVHN-ConvNet`, `EuroSAT-ConvNet`, `CIFAR10-ConvNet`, `CIFAR100-ConvNet`): FuGuard's protocol (Dirichlet 0.1, CIFAR-100: 1.0; 3-block ConvNet; Adam 1e-3; batch 256; 5 local epochs; 49 rounds; client 3 backdoored with a 4x4 patch to class 9 on every non-target sample; all data used, so no MIA proxy split and no MIA) run with FedUP (and optionally Retrain / EraseClient).
  Parity details taken from the FuGuard release: per-dataset normalisation statistics and its **trigger constants in normalised space (not exactly white; gray for EuroSAT)** (`bd_trigger="repo"`, `"white"` gives a true white patch). Known differences: EuroSAT uses a fixed 80/20 split (FuGuard splits with the run seed); SVHN and EuroSAT come from the Hugging Face hub / the official EuroSAT zip, which have **not** been run in the authoring environment.
  Reports ACC, ASR (strict: non-target inputs only) and `ASR_incl_target` (FuGuard's definition, which includes untouched target-class images and therefore has a floor). This answers whether a filter trained on clean centroids over a frozen extractor removes a backdoor.
  Keep in mind when comparing with FuGuard's tables: MIA is defined differently (not compared), FedUP has no recovery rounds (FuGuard's recovery-round metric has no FedUP counterpart), FedUP needs every retained client to upload centroids once (FuGuard needs the target client plus recovery rounds).
* `SYNTHETIC-Tiny`: smoke-test benchmark.

## 3b. The comparison tables
`fedup_repro/comparison.py` compares our runs with `paper_reference.py` (FedUP paper tables parsed from its LaTeX; FuGuard paper Table II/III; our FuGuard reproduction from Weekly Update 2). The scoreboard checks the FedUP paper's claims automatically (time and communication far below Retrain, class F-A about 0, client R-A within 0.03 of Retrain, bottleneck 32 near-best, DP lowers MIA).
The comparison code was tested on **fabricated** results shaped like the paper's numbers (to exercise every table and plot), never on real runs: do not read any value in the tests as a result.

## 4. Known limitations
* The supplement (client count, DP calibration details, any extra hyper-parameters) was not available.
* Exact-Fun, FedEraser, FUSED are not implemented (section 2).
* Single runs per seed; default seeds 0 and 1, population std (`FEDUP_STD_DDOF=1` for the sample std).
* ImageNet weights are downloaded by torchvision on first use; offline machines fall back to random initialisation with a warning.
