"""Configuration: benchmarks, scenarios, defaults and presets.

Presets
  "repo"    (default) what the authors' released code does (this is what produced the published numbers);
  "paper"   what the paper's text says, wherever the text and the code disagree (see docs/FIXES_AND_DEVIATIONS.md);
  "fuguard" the FuGuard protocol (Dirichlet 0.1, ConvNet, Adam, backdoored client) for the cross-evaluation experiment.
Every disagreement is a single config key, so each one can also be switched individually through `overrides`.
"""
import copy
import hashlib
import json

BENCHMARKS = {
    # name: dataset, model, classes, lr (from the official utils.model_init), modality
    "MNIST-LeNet5":        dict(dataset="mnist",    model="lenet5",   num_classes=10,  lr=0.005, modality="image", in_ch=1),
    "CIFAR10-ResNet18":    dict(dataset="cifar10",  model="resnet18", num_classes=10,  lr=0.005, modality="image", in_ch=3),
    "CIFAR10-Transformer": dict(dataset="cifar10",  model="vit",      num_classes=10,  lr=0.005, modality="image", in_ch=3),
    "AGNews-TinyBERT":     dict(dataset="agnews",   model="tinybert", num_classes=4,   lr=0.005, modality="text"),
    "CIFAR100-ResNet34":   dict(dataset="cifar100", model="resnet34", num_classes=100, lr=0.01,  modality="image", in_ch=3),
    # extension benchmarks (not in the FedUP paper)
    # FuGuard-protocol benchmarks (3-block ConvNet; per-dataset statistics and trigger constants as in the FuGuard release)
    "CIFAR10-ConvNet":     dict(dataset="cifar10",  model="convnet",  num_classes=10,  lr=0.001, modality="image", in_ch=3, norm_key="CIFAR10"),
    "CIFAR100-ConvNet":    dict(dataset="cifar100", model="convnet",  num_classes=100, lr=0.001, modality="image", in_ch=3, norm_key="CIFAR100", fuguard_alpha=1.0),
    "SVHN-ConvNet":        dict(dataset="svhn",     model="convnet",  num_classes=10,  lr=0.001, modality="image", in_ch=3, norm_key="SVHN"),
    "EuroSAT-ConvNet":     dict(dataset="eurosat",  model="convnet",  num_classes=10,  lr=0.001, modality="image", in_ch=3, norm_key="EuroSAT"),
    "SYNTHETIC-Tiny":      dict(dataset="synthetic", model="tiny",    num_classes=10,  lr=0.01,  modality="image", in_ch=3),   # smoke test
}
PAPER_BENCHMARKS = ["MNIST-LeNet5", "CIFAR10-ResNet18", "CIFAR10-Transformer", "AGNews-TinyBERT", "CIFAR100-ResNet34"]
FUGUARD_BENCHMARKS = ["SVHN-ConvNet", "EuroSAT-ConvNet", "CIFAR10-ConvNet", "CIFAR100-ConvNet"]     # the four datasets of the FuGuard paper

SCENARIOS = ("client", "class", "sample", "client_backdoor")
# centroid sampling ratio rho per scenario  [paper] 0.8 / 0.1 / 0.5 ; client_backdoor reuses the client value
RHO = {"client": 0.8, "class": 0.1, "sample": 0.5, "client_backdoor": 0.8}
# noise scale sigma chosen in the paper (epsilon about 10): MNIST/CIFAR-10/AG News 0.001, CIFAR-100 0.01
PAPER_DP_SIGMA = {"MNIST-LeNet5": 0.001, "CIFAR10-ResNet18": 0.001, "CIFAR10-Transformer": 0.001,
                  "AGNews-TinyBERT": 0.001, "CIFAR100-ResNet34": 0.01}
METHODS = ("fedup", "retrain", "eraseclient", "federaser", "exactfun", "fused")
IMPLEMENTED_METHODS = ("fedup", "retrain", "eraseclient")        # the others are documented stubs (no reference code)

BASE = dict(
    # ---- federated setting ------------------------------------------------------------------------------------
    num_clients=10,              # ASSUMPTION: not in the main text (supplement not available); the repo default is 1
    alpha=0.5,                   # [paper] Dirichlet 0.5   (repo argparse default 1.0)
    min_client_size=100,         # [repo] least_samples
    proxy_frac=0.2,              # [repo] fraction of each class of each client held out as proxy data (for the MIA shadow model)
    train_ratio_client=0.7,      # [repo] client scenario: train+test merged, then 70/30 per client
    train_ratio_other=0.99,      # [repo] class / sample scenarios: 99/1 per client; evaluation uses the official test set
    data_fraction=1.0,           # < 1 subsamples the raw data (quick runs)
    # ---- federated training -----------------------------------------------------------------------------------
    local_epochs=5,              # [repo README]
    local_batch=64,              # [repo]
    global_epochs=50,            # [repo README]
    rounds_per_epoch=2,          # [repo BUG] each global epoch runs two FedAvg rounds  -> 100 rounds ("paper": 1)
    aggregation="uniform",       # [repo] plain mean of client models ("paper": weighted by data size)
    optimizer="sgd_momentum",    # [repo] SGD(momentum 0.9, wd 5e-4); AdamW for text
    momentum=0.9, weight_decay=5e-4,
    pretrained=True,             # [repo] torchvision ResNets start from ImageNet weights (not mentioned in the paper)
    augment="static",            # [repo] one-time random crop+flip baked into every CIFAR image, train AND test ("none" = paper-neutral)
    amp=True,                    # fp16 autocast on GPU
    text_model="prajjwal1/bert-tiny",   # paper says "TinyBERT"; the repo loads a local 'bert-tiny-local'. The paper's filter-memory column (32 KB at dim 32) implies a 128-d feature = BERT-tiny, not TinyBERT-312
    max_len=128,
    # ---- unlearning scenarios ---------------------------------------------------------------------------------
    forget_clients=[0],          # [repo default] client 0 is the Byzantine / forgotten client
    forget_classes=[0],          # ASSUMPTION (repo default: empty list)
    backdoor_frac=0.2,           # [repo] fraction of each client's samples carrying the trigger (sample scenario)
    sample_target=0,             # [paper] triggered samples are mapped to class 0
    trigger_mode="repo",         # sample scenario: "repo" writes 255 into the normalised pixel; "white" = a real white pixel
    text_trigger_id=1996,        # [repo] token id appended at the end of the sequence
    bd_patch=4, bd_target=9, bd_ratio=1.0,   # client_backdoor scenario = FuGuard protocol (4x4 white patch -> class 9, all samples)
    bd_trigger="white",          # "white": a real white patch | "repo": FuGuard's released constants (slightly off white; GRAY for EuroSAT)
    # ---- FedUP (filter) ---------------------------------------------------------------------------------------
    out_dim=32,                  # [paper] bottleneck dimension 32
    filter_epochs=50,            # [repo] lora_trained_epoch
    filter_batch=10,             # [repo] centroid mini-batch
    filter_reshuffle=False,      # [repo] batches are formed once and replayed in the same order every epoch
    filter_out_relu=True,        # [repo] the adapter ends with a ReLU
    loss_alpha=0.5,              # [paper] alpha = 0.5 (weight of the cross-entropy; 1-alpha for reconstruction)
    centroid_method="cluster",   # "cluster" (KMeans) | "none" (raw features, ablation)
    dp_sigma=None,               # None -> preset ("repo": 0 = no noise, "paper": PAPER_DP_SIGMA)
    # ---- evaluation -------------------------------------------------------------------------------------------
    r_a_style="class_mean",      # [repo] accuracies are averaged over (client, class) cells; "pooled" = plain accuracy
    eval_batch=256,
    mia=True, mia_shadow_epochs=1, mia_attack_epochs=20, mia_max=2000,
    # ---- EraseClient baseline (ported from FuGuard's FU baseline) ---------------------------------------------
    ec_epochs=5, ec_lr=0.001, ec_distance_threshold=2.2, ec_recovery_rounds=3,
    eval_every=5,
)

PRESETS = {
    "repo": dict(),
    "paper": dict(rounds_per_epoch=1, aggregation="size", pretrained=False, augment="none", trigger_mode="white",
                  dp_sigma="paper"),
    # FuGuard protocol: backdoored client, Dirichlet 0.1, Adam 1e-3, 5 local epochs, batch 256, 49 rounds
    # FuGuard's protocol uses ALL of the data, so the MIA proxy hold-out and the MIA are off here (--set proxy_frac=0.2 --set mia=True
    # brings them back). The backdoored client is client_3 and the trigger constants are FuGuard's released ones.
    "fuguard": dict(alpha=0.1, num_clients=10, local_epochs=5, local_batch=256, global_epochs=49, rounds_per_epoch=1,
                    aggregation="size", optimizer="adam", pretrained=False, augment="none", train_ratio_client=1.0,
                    train_ratio_other=1.0, proxy_frac=0.0, min_client_size=10, forget_clients=[3], bd_trigger="repo",
                    mia=False),
}

QUICK = dict(global_epochs=3, rounds_per_epoch=1, local_epochs=1, filter_epochs=10, data_fraction=0.2,
             mia_attack_epochs=5, mia_max=500, ec_epochs=1, ec_recovery_rounds=1, eval_every=1)

# keys that change federated training / the data, i.e. that define a cached FedAvg checkpoint
_TRAIN_KEYS = ["benchmark", "dataset", "model", "num_clients", "alpha", "min_client_size", "proxy_frac", "train_ratio_client",
               "train_ratio_other", "data_fraction", "local_epochs", "local_batch", "global_epochs", "rounds_per_epoch",
               "aggregation", "optimizer", "momentum", "weight_decay", "pretrained", "augment", "lr", "text_model", "max_len",
               "forget_clients", "backdoor_frac", "sample_target", "trigger_mode", "text_trigger_id",
               "bd_patch", "bd_target", "bd_ratio", "bd_trigger", "norm_key"]


def make_config(benchmark, scenario, preset="repo", quick=False, **overrides):
    if benchmark not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {benchmark!r}; choose from {list(BENCHMARKS)}")
    if scenario not in SCENARIOS:
        raise KeyError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
    cfg = copy.deepcopy(BASE)
    cfg.update(copy.deepcopy(BENCHMARKS[benchmark]))
    cfg.update(copy.deepcopy(PRESETS[preset]))
    if preset == "fuguard" and "fuguard_alpha" in cfg:
        cfg["alpha"] = cfg["fuguard_alpha"]                  # FuGuard: Dirichlet 1.0 for CIFAR-100, 0.1 otherwise
    if quick:
        cfg.update(QUICK)
    cfg.update(copy.deepcopy(overrides))
    cfg["benchmark"], cfg["scenario"], cfg["preset"], cfg["quick"] = benchmark, scenario, preset, quick
    cfg.setdefault("rho", None)
    if cfg["rho"] is None:
        cfg["rho"] = RHO[scenario]
    if cfg["dp_sigma"] in (None, "repo"):
        cfg["dp_sigma"] = 0.0
    elif cfg["dp_sigma"] == "paper":
        cfg["dp_sigma"] = PAPER_DP_SIGMA.get(benchmark, 0.001)
    if cfg["modality"] == "text" and cfg["optimizer"] == "sgd_momentum":
        cfg["optimizer"] = "adamw"                      # [repo] AdamW for text
    return cfg


def n_rounds(cfg):
    return int(cfg["global_epochs"]) * int(cfg["rounds_per_epoch"])


def train_view(cfg):
    """Which poisoned/clean version of the data a FedAvg checkpoint was trained on."""
    s = cfg["scenario"]
    return "clean" if s == "class" else s


def train_key(cfg, seed):
    d = {k: cfg.get(k) for k in _TRAIN_KEYS}
    d["view"] = train_view(cfg)
    d["seed"] = int(seed)
    h = hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return f"{cfg['benchmark']}_{d['view']}_s{seed}_{h}"
