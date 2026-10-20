"""Configuration.

Values marked [paper] come from the paper's Section V; [repo] from the official code;
[assumed] are not specified anywhere and are exposed so you can change them.
"""
import copy

METHODS = ["retrain", "fedsga", "fu", "not", "fastfedul", "fuguard"]
METHOD_LABELS = {"retrain": "Retrain", "fedsga": "FedSGA", "fu": "FU", "not": "NoT",
                 "fastfedul": "FastFedUL", "fuguard": "FuGuard"}
IMAGE_DATASETS = ["SVHN", "EuroSAT", "CIFAR10", "CIFAR100"]

# dataset -> (num_classes, dirichlet alpha)   [paper] alpha=0.1 except CIFAR-100 alpha=1
_DATASET_SPECS = {
    "SVHN": (10, 0.1), "EuroSAT": (10, 0.1), "CIFAR10": (10, 0.1), "CIFAR100": (100, 1.0),
}

# Two evaluation protocols, run separately (see docs/FIXES_AND_DEVIATIONS.md, item P1):
#   table2 : 50 training rounds, then 5 recovery rounds  (Table II, Fig. 6, Fig. 9, Fig. 10, Table IV)
#   table3 : 50 training rounds, then 50 recovery rounds (Table III: rounds to reach 95%/100% accuracy)
PROTOCOLS = {
    "table2": {"recovery_rounds": 5, "seeds": (0, 1)},   # [paper] seeds 0 and 1
    "table3": {"recovery_rounds": 50, "seeds": (0,)},    # [paper] "under a fixed random seed"
}

BASE_IMAGE_CONFIG = dict(
    dataset_name="SVHN", tag="SVHN", num_classes=10, image_size=(32, 32),
    num_clients=10,                      # [paper]
    dirichlet_alpha=0.1,                 # [paper]
    inject_ratio=1.0,                    # [repo] all samples of the attacker are poisoned
    backdoor_client="client_3",          # [repo]
    target_label=9,                      # [paper] target class "9"
    trigger_size=4,                      # [paper] 4x4 white patch
    # --- schedule ---
    unlearn_round=50,                    # [repo] unlearning request fires in round 50 (FedAvg ran rounds 1..49)
    recovery_rounds=5,                   # protocol-dependent, see PROTOCOLS
    # --- local training ---
    local_epochs=5, batch_size=256, lr=1e-3,   # [paper]
    optimizer="adam",                    # [repo] kept on purpose (paper text says SGD)
    # --- FuGuard ---
    samples_scale=0.1,                   # [paper] proxy = 10% of the client's data
    gen_bs=64, unlearn_bs=32,            # [repo]
    ot_lambda=0.1,                       # [paper]
    forgetting_epoch=5,                  # [paper]
    forgetting_lr=1e-3,                  # [paper] same lr as training
    pca_components=5, pca_dir_idx=1, pca_alpha=1.0,   # [paper] K=5, r=2 (0-indexed 1), alpha=1.0
    sinkhorn_blur=0.05,                  # [repo]
    vae_id="stabilityai/sd-vae-ft-mse",  # [repo] (paper: "a pretrained VAE")
    # --- baselines ---
    distance_threshold=2.2,              # [repo] FU
    fastfedul_scale=1.0,                 # [repo]
    compute_mia=True,
)


def image_config(dataset, **overrides):
    """Config for one image dataset. Extra keyword args override any field."""
    if dataset not in _DATASET_SPECS:
        raise ValueError(f"unknown dataset {dataset}; choose from {list(_DATASET_SPECS)}")
    cfg = copy.deepcopy(BASE_IMAGE_CONFIG)
    n_cls, alpha = _DATASET_SPECS[dataset]
    cfg.update(dataset_name=dataset, tag=dataset, num_classes=n_cls, dirichlet_alpha=alpha)
    cfg.update(overrides)
    cfg["rounds"] = cfg["unlearn_round"] + cfg["recovery_rounds"]
    return cfg


BASE_AWE_CONFIG = dict(
    dataset_name="AWE", tag="AWE", kind="awe",
    # --- preprocessing [repo config.py] ---
    fs_orig=4000, fs_target=500, window_size=30, stride=10, lowpass_cutoff=250, filter_order=4,
    input_dim=1000, num_classes=6,
    num_clients=6,                       # [paper] six clients (repo config said 10 -> fixed)
    backdoor_client="client_2", target_label=5,        # [repo]
    trigger_amplitude=20.0,              # [paper] +20 (the repo config's 5.0 was never used)
    trigger_extra_channels=(1, 2), trigger_extra_times=(5, 6, 7), trigger_noise_std=0.5,
    data_seed=42,                        # [repo] the stray key "F": 42
    # --- schedule: [paper] 100 training epochs, 50 recovery epochs ---
    unlearn_round=100, recovery_rounds=50,
    local_epochs=2, batch_size=32, lr=5e-5, optimizer="adam",
    # --- FuGuard ---
    samples_scale=0.1, gen_bs=64, unlearn_bs=64, ot_lambda=0.1,
    forgetting_epoch=5, forgetting_lr=0.05,
    pca_components=5, pca_dir_idx=1, pca_alpha=1.0, sinkhorn_blur=0.05,
    vae_hidden=256, vae_latent=32, vae_epochs=1000, vae_lr=1e-3,
    distance_threshold=2.2, fastfedul_scale=1.0,
    compute_mia=False,                   # the AWE scripts never computed MIA
)


def awe_config(**overrides):
    cfg = copy.deepcopy(BASE_AWE_CONFIG)
    cfg.update(overrides)
    cfg["rounds"] = cfg["unlearn_round"] + cfg["recovery_rounds"]
    return cfg


def ot_variant(lam):
    """Variant name used for FuGuard runs with a non-default lambda_OT, e.g. 'fuguard_ot0.08'."""
    return f"fuguard_ot{lam:g}"


def lr_variant(method, lr):
    """Variant name for the AWE unlearning-learning-rate sweep, e.g. 'fedsga_lr0.05'."""
    return f"{method}_lr{lr:g}"
