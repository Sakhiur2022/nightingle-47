"""Pure-numpy data partitioning (no torch import, easy to test)."""
import numpy as np


def dirichlet_partition(y, num_clients, num_classes, alpha, rng, min_size=100, max_tries=500):
    """Label-skew Dirichlet partition as in the official data_utils.separate_data (PFLlib style):
    for each class draw client proportions ~ Dir(alpha), stop assigning to clients that already hold their fair share,
    and retry until every client has at least `min_size` samples. Returns a list of index arrays."""
    y = np.asarray(y)
    n = len(y)
    if n < num_clients * min_size:
        min_size = max(1, n // (2 * num_clients))        # tiny / subsampled datasets: relax the minimum
    for _ in range(max_tries):
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(y == k)[0]
            if len(idx_k) == 0:
                continue
            rng.shuffle(idx_k)
            p = rng.dirichlet(np.repeat(alpha, num_clients))
            p = np.array([pj * (len(idx_j) < n / num_clients) for pj, idx_j in zip(p, idx_batch)])
            if p.sum() <= 0:
                p = np.ones(num_clients)
            p = p / p.sum()
            cuts = (np.cumsum(p) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + part.tolist() for idx_j, part in zip(idx_batch, np.split(idx_k, cuts))]
        if min(len(i) for i in idx_batch) >= min_size:
            return [np.array(i, dtype=np.int64) for i in idx_batch]
    raise RuntimeError(f"could not give every client >= {min_size} samples in {max_tries} tries (alpha={alpha}); "
                       "increase alpha / data_fraction or lower min_client_size")


def split_proxy_main(idx, y, proxy_frac, rng):
    """Hold out `proxy_frac` of every class of one client as proxy data. Returns (main_idx, proxy_idx)."""
    idx = np.asarray(idx)
    if proxy_frac <= 0 or len(idx) == 0:
        return idx, idx[:0]
    main, proxy = [], []
    yy = y[idx]
    for c in np.unique(yy):
        ic = idx[yy == c].copy()
        rng.shuffle(ic)
        k = int(round(len(ic) * proxy_frac))
        if len(ic) - k < 1:                  # keep at least one sample in the main part
            k = 0
        proxy.append(ic[:k])
        main.append(ic[k:])
    return np.concatenate(main), (np.concatenate(proxy) if proxy else idx[:0])


def train_test_split_idx(idx, train_ratio, rng):
    idx = np.asarray(idx).copy()
    rng.shuffle(idx)
    k = int(round(len(idx) * train_ratio))
    if train_ratio < 1.0:
        k = min(max(k, 1), len(idx) - 1) if len(idx) > 1 else len(idx)
    return idx[:k], idx[k:]
