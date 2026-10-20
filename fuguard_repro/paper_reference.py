"""Numbers printed in the paper (transcribed from the OCR of Tables II-IV), used for side-by-side comparison.
Table II entries are (ACC mean, ACC std, ASR mean, ASR std) in %.
NOTE: in the paper, FedSGA and FU have IDENTICAL EuroSAT entries (74.61+-0.87 / 15.74+-4.02) -> possible typo."""

PAPER_TABLE2 = {
    "SVHN": {"fedavg": (83.92, 1.25, 26.32, 11.06), "retrain": (70.91, 3.96, 6.32, 2.58),
             "fedsga": (82.86, 0.41, 4.24, 1.32), "fu": (83.85, 0.76, 11.90, 2.21),
             "not": (71.81, 0.20, 5.73, 1.17), "fastfedul": (84.32, 1.14, 6.26, 0.16),
             "fuguard": (82.86, 0.63, 4.09, 1.74)},
    "EuroSAT": {"fedavg": (73.84, 0.18, 21.47, 0.49), "retrain": (53.66, 3.99, 10.71, 4.84),
                "fedsga": (74.61, 0.87, 15.74, 4.02), "fu": (74.61, 0.87, 15.74, 4.02),
                "not": (54.58, 5.16, 9.81, 0.03), "fastfedul": (75.05, 1.25, 18.35, 1.39),
                "fuguard": (75.79, 0.51, 15.13, 4.00)},
    "CIFAR10": {"fedavg": (64.86, 1.56, 16.59, 2.19), "retrain": (47.69, 3.78, 5.18, 2.78),
                "fedsga": (62.77, 3.06, 7.68, 1.46), "fu": (64.68, 1.88, 9.72, 0.82),
                "not": (45.79, 3.52, 7.67, 2.14), "fastfedul": (65.40, 1.22, 8.91, 0.62),
                "fuguard": (63.80, 2.16, 7.67, 2.12)},
    "CIFAR100": {"fedavg": (45.08, 0.08, 15.47, 3.02), "retrain": (35.33, 0.28, 0.47, 0.15),
                 "fedsga": (45.70, 0.94, 0.62, 0.12), "fu": (48.85, 0.04, 0.82, 0.10),
                 "not": (36.44, 0.20, 0.52, 0.04), "fastfedul": (48.86, 0.15, 0.78, 0.07),
                 "fuguard": (48.30, 0.10, 0.55, 0.03)},
}

# Table III: dataset -> method -> (RR(0.95), RR(1))
PAPER_TABLE3 = {
    "SVHN": {"retrain": (12, 27), "fedsga": (1, 1), "fu": (1, 1), "not": (18, 50), "fastfedul": (1, 1), "fuguard": (1, 1)},
    "EuroSAT": {"retrain": (34, 46), "fedsga": (2, 3), "fu": (2, 3), "not": (20, 35), "fastfedul": (2, 3), "fuguard": (2, 6)},
    "CIFAR10": {"retrain": (20, 33), "fedsga": (1, 2), "fu": (1, 2), "not": (24, 50), "fastfedul": (2, 2), "fuguard": (1, 2)},
    "CIFAR100": {"retrain": (9, 12), "fedsga": (4, 5), "fu": (1, 1), "not": (11, 14), "fastfedul": (1, 1), "fuguard": (2, 2)},
}

# Table IV (SVHN): (lambda_OT, mode) -> (ACC %, ASR %)
PAPER_TABLE4 = {
    (0.0, "only unlearn"): (19.58, 0.0), (0.0, "with recovery"): (75.72, 4.44),
    (0.1, "only unlearn"): (17.32, 0.0), (0.1, "with recovery"): (82.85, 4.09),
}
