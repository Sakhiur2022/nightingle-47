"""Numbers published in the FedUP paper (parsed from the LaTeX tables). Methods: E-C = eraseclient, Federaser = federaser,
E-F = exactfun, FUSED = fused. Values are fractions (R-A, F-A, MIA, 0A, PS), seconds (Time) and MB (Comm)."""

CLIENT_METHODS = ["eraseclient", "federaser", "exactfun", "fused", "retrain", "fedup"]
CLASS_METHODS = ["fused", "retrain", "fedup"]
SAMPLE_METHODS = ["fused", "retrain", "fedup"]

_RAW = {
 "MNIST-LeNet5": {"R-A": [0.99, 0.97, 0.97, 0.99, 0.99, 0.99], "R-A_cls": [0.97, 0.99, 1.0], "0A_smp": [0.95, 1.0, 0.95], "F-A": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "F-A_cls": [0.0, 0.0, 0.0], "PS_smp": [1.0, 1.0, 1.0], "MIA": [0.8, 0.6, 0.76, 0.69, 0.67, 0.7], "MIA_cls": [0.99, 0.97, 0.99], "MIA_smp": [0.96, 0.68, 0.97]},
 "CIFAR10-ResNet18": {"R-A": [0.57, 0.46, 0.56, 0.56, 0.58, 0.57], "R-A_cls": [0.71, 0.72, 0.71], "0A_smp": [0.6, 0.73, 0.7], "F-A": [0.04, 0.08, 0.04, 0.05, 0.03, 0.03], "F-A_cls": [0.0, 0.0, 0.0], "PS_smp": [0.52, 0.6, 0.53], "MIA": [0.67, 0.7, 0.62, 0.68, 0.57, 0.63], "MIA_cls": [0.81, 0.63, 0.78], "MIA_smp": [0.95, 0.98, 0.96]},
 "CIFAR10-Transformer": {"R-A": [0.48, 0.51, 0.5, 0.49, 0.52, 0.52], "R-A_cls": [0.59, 0.6, 0.59], "0A_smp": [0.35, 0.64, 0.57], "F-A": [0.05, 0.05, 0.05, 0.05, 0.04, 0.04], "F-A_cls": [0.0, 0.0, 0.0], "PS_smp": [0.5, 0.63, 0.52], "MIA": [0.54, 0.35, 0.59, 0.56, 0.64, 0.52], "MIA_cls": [0.81, 0.78, 0.9], "MIA_smp": [0.96, 0.91, 0.76]},
 "AGNews-TinyBERT": {"R-A": [0.89, 0.88, 0.88, 0.89, 0.89, 0.89], "R-A_cls": [0.89, 0.93, 0.93], "0A_smp": [0.92, 0.93, 0.93], "F-A": [0.03, 0.05, 0.04, 0.04, 0.03, 0.05], "F-A_cls": [0.0, 0.0, 0.0], "PS_smp": [0.87, 0.89, 0.88], "MIA": [0.63, 0.63, 0.79, 0.68, 0.68, 0.77], "MIA_cls": [0.44, 0.63, 0.5], "MIA_smp": [0.77, 0.54, 0.7]},
 "CIFAR100-ResNet34": {"R-A": [0.13, 0.15, 0.27, 0.26, 0.27, 0.26], "R-A_cls": [0.36, 0.37, 0.37], "0A_smp": [0.5, 0.57, 0.55], "F-A": [0.01, 0.01, 0.01, 0.01, 0.01, 0.02], "F-A_cls": [0.0, 0.0, 0.0], "PS_smp": [0.55, 0.56, 0.54], "MIA": [0.32, 0.49, 0.43, 0.4, 0.26, 0.69], "MIA_cls": [0.91, 0.38, 0.67], "MIA_smp": [0.98, 0.98, 0.99]},
}


def _zip(names, vals):
    return dict(zip(names, vals))


PAPER_MAIN = {}
for b, d in _RAW.items():
    PAPER_MAIN[b] = {
        "client": {m: dict(R_A=d["R-A"][i], F_A=d["F-A"][i], MIA=d["MIA"][i]) for i, m in enumerate(CLIENT_METHODS)},
        "class": {m: dict(R_A=d["R-A_cls"][i], F_A=d["F-A_cls"][i], MIA=d["MIA_cls"][i]) for i, m in enumerate(CLASS_METHODS)},
        "sample": {m: {"0A": d["0A_smp"][i], "PS": d["PS_smp"][i], "MIA": d["MIA_smp"][i]} for i, m in enumerate(SAMPLE_METHODS)},
    }

# Table "Comparing time and communication costs": CIFAR-10 (ResNet-18) and AG News, client unlearning
PAPER_COST = {
    "CIFAR10-ResNet18": {"eraseclient": (1627.76, 1110.98), "federaser": (1586.59, 2820.18), "exactfun": (898.70, 1452.82),
                         "fused": (151.21, 31.36), "retrain": (935.54, 1538.25), "fedup": (6.33, 11.50)},
    "AGNews-TinyBERT": {"eraseclient": (567.56, 268.64), "federaser": (634.67, 738.76), "exactfun": (629.83, 369.38),
                        "fused": (88.31, 0.55), "retrain": (859.28, 537.28), "fedup": (6.45, 6.73)},
}                                                       # (Time s, Comm MB)

# Ablations (client scenario unless noted). Bottleneck dimension -> (R-A, filter memory KB)
PAPER_DIM = {
    "dims": [4, 8, 16, 32, 64],
    "R_A": {"MNIST-LeNet5": [0.98, 0.99, 0.99, 0.99, 0.99], "CIFAR10-ResNet18": [0.55, 0.58, 0.58, 0.60, 0.59],
            "CIFAR10-Transformer": [0.43, 0.47, 0.46, 0.51, 0.46], "AGNews-TinyBERT": [0.89, 0.90, 0.90, 0.91, 0.90],
            "CIFAR100-ResNet34": [0.06, 0.18, 0.24, 0.25, 0.23]},
    "memory_kb": {"MNIST-LeNet5": [16, 32, 64, 128, 256], "CIFAR10-ResNet18": [16, 32, 64, 128, 256],
                  "CIFAR10-Transformer": [16, 32, 64, 128, 256], "AGNews-TinyBERT": [4, 8, 16, 32, 64],
                  "CIFAR100-ResNet34": [16, 32, 64, 128, 256]},
}
# Loss weight alpha_CE (class unlearning): (R-A, F-A)
PAPER_ALPHA = {
    "alphas": [0.1, 0.3, 0.5, 0.7, 0.9],
    "MNIST-LeNet5": [(0.99, 0.67), (1.00, 0.00), (0.99, 0.00), (0.99, 0.00), (0.99, 0.00)],
    "CIFAR10-ResNet18": [(0.70, 0.00), (0.68, 0.00), (0.71, 0.00), (0.68, 0.00), (0.64, 0.00)],
    "CIFAR10-Transformer": [(0.51, 0.22), (0.56, 0.00), (0.59, 0.00), (0.51, 0.00), (0.45, 0.00)],
    "AGNews-TinyBERT": [(0.92, 0.00), (0.93, 0.00), (0.93, 0.00), (0.92, 0.00), (0.87, 0.00)],
    "CIFAR100-ResNet34": [(0.35, 0.11), (0.37, 0.00), (0.37, 0.00), (0.31, 0.00), (0.30, 0.00)],
}
# Differential privacy (client scenario): (with DP, without DP)
PAPER_DP = {"CIFAR10-ResNet18": {"R_A": (0.5814, 0.5833), "F_A": (0.0438, 0.0460), "MIA": (0.5239, 0.6373)},
            "AGNews-TinyBERT": {"R_A": (0.9132, 0.9114), "F_A": (0.0490, 0.0303), "MIA": (0.5654, 0.6196)}}
# Raw features vs class centroids (client scenario): (features, centroids)
PAPER_CENTROID = {"MNIST-LeNet5": {"R_A": (0.9900, 0.9921), "F_A": (0.0001, 0.0005)},
                  "CIFAR10-ResNet18": {"R_A": (0.6011, 0.6020), "F_A": (0.0498, 0.0331)},
                  "CIFAR10-Transformer": {"R_A": (0.5264, 0.5671), "F_A": (0.0632, 0.0490)},
                  "AGNews-TinyBERT": {"R_A": (0.9153, 0.9118), "F_A": (0.0331, 0.0482)},
                  "CIFAR100-ResNet34": {"R_A": (0.2537, 0.2508), "F_A": (0.0076, 0.0172)}}


# ======================================================================================================================
# FuGuard (the other paper) -- numbers for the "FuGuard vs FedUP" comparison. Percent, (mean, std).
# Protocol: 10 clients (Dirichlet 0.1; CIFAR-100: 1.0), one client backdoored with a 4x4 white patch -> class 9, 3-block ConvNet,
# FedAvg 49 rounds, then unlearning + 5 recovery rounds. ASR = fraction of the triggered test set predicted as class 9
# (FuGuard's definition: untouched target-class images stay in the set, so ASR has a floor).
FUGUARD_DATASETS = {"SVHN-ConvNet": "SVHN", "EuroSAT-ConvNet": "EuroSAT", "CIFAR10-ConvNet": "CIFAR-10", "CIFAR100-ConvNet": "CIFAR-100"}

# Table II of the FuGuard paper: (ACC mean, ACC std, ASR mean, ASR std)
FUGUARD_PAPER = {
    "SVHN":      {"fedavg": (83.92, 1.25, 26.32, 11.06), "fuguard": (82.86, 0.63, 4.09, 1.74)},
    "EuroSAT":   {"fedavg": (73.84, 0.18, 21.47, 0.49),  "fuguard": (75.79, 0.51, 15.13, 4.00)},
    "CIFAR-10":  {"fedavg": (64.86, 1.56, 16.59, 2.19),  "fuguard": (63.80, 2.16, 7.67, 2.12)},
    "CIFAR-100": {"fedavg": (45.08, 0.08, 15.47, 3.02),  "fuguard": (48.30, 0.10, 0.55, 0.03)},
}
# Table III of the FuGuard paper: recovery rounds to reach 95 % / 100 % of the FedAvg accuracy (FuGuard row)
FUGUARD_RR_PAPER = {"SVHN": (1, 1), "EuroSAT": (2, 6), "CIFAR-10": (1, 2), "CIFAR-100": (2, 2)}

# Our own FuGuard reproduction (Weekly Update 2, results.zip): same layout
FUGUARD_REPRO = {
    "SVHN":      {"fedavg": (84.04, 0.66, 26.92, 7.85), "fuguard": (83.29, 0.59, 4.22, 1.79)},
    "EuroSAT":   {"fedavg": (72.32, 2.38, 22.30, 1.24), "fuguard": (74.46, 0.61, 16.54, 3.13)},
    "CIFAR-10":  {"fedavg": (64.81, 1.25, 15.50, 0.86), "fuguard": (63.66, 2.05, 7.77, 1.93)},
    "CIFAR-100": {"fedavg": (44.52, 0.39, 15.53, 3.33), "fuguard": (48.53, 0.03, 0.68, 0.04)},
}
FUGUARD_RR_REPRO = {"SVHN": (3, 17), "EuroSAT": (2, 3), "CIFAR-10": (3, 13), "CIFAR-100": (2, 2)}
