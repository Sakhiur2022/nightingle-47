"""Client side of FedUP: feature extraction with the frozen extractor, class centroids (KMeans) and DP noise."""
import math

import torch

from .data import iterate


@torch.no_grad()
def extract_by_class(model, x, y, cfg, device, poison=None, trigger=None, keep=None):
    """-> {class: [n_c, d] float32 CPU tensor}. `keep` is an optional boolean mask of samples to use."""
    model.eval()
    if keep is not None:
        x, y = x[keep], y[keep]
        poison = poison[keep] if poison is not None else None
    feats = []
    for xb, _ in iterate(x, y, cfg["eval_batch"], cfg, device, poison=poison, trigger=trigger):
        feats.append(model.features(xb).float().cpu())
    if not feats:
        return {}
    f = torch.cat(feats)
    return {int(c): f[y == c] for c in torch.unique(y)}


def _sqdist(a, b):
    return torch.clamp((a * a).sum(1, keepdim=True) + (b * b).sum(1)[None] - 2 * a @ b.t(), min=0.0)


def kmeans_centroids(x, k, device, n_init=5, max_iter=100, tol=1e-4, seed=0):
    """Plain KMeans on [n, d] features (best of n_init runs by inertia), as in the official `cluster_sample`."""
    n = x.shape[0]
    if n == 0:
        return x.float()
    if n == 1 or k <= 1:
        return x.float().mean(0, keepdim=True)
    k = min(k, n)
    if k == n:
        return x.float()
    x = x.to(device).float()
    g = torch.Generator(device=device).manual_seed(seed)
    best, best_in = None, None
    for _ in range(n_init):
        centers = x[torch.randperm(n, generator=g, device=device)[:k]].clone()
        for _ in range(max_iter):
            old = centers.clone()
            labels = _sqdist(x, centers).argmin(1)
            sums = torch.zeros_like(centers).index_add_(0, labels, x)
            counts = torch.bincount(labels, minlength=k).float()
            centers = sums / counts.clamp(min=1)[:, None]
            empty = counts == 0
            if bool(empty.any()):                       # re-seed empty clusters with random samples
                centers[empty] = x[torch.randint(0, n, (int(empty.sum()),), generator=g, device=device)]
            if float((centers - old).norm(dim=1).max()) <= tol:
                break
        inertia = float(_sqdist(x, centers).min(1).values.sum())
        if best_in is None or inertia < best_in:
            best, best_in = centers.clone(), inertia
    return best.cpu()


def make_centroids(feats_by_class, rho, method, sigma, device, seed=0):
    """Per class: KMeans with K = ceil(rho * n) centroids, then N(0, sigma^2 I) noise (differential privacy).
    method 'none' keeps the raw features (ablation). Returns {class: [m, d]}."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for c, f in feats_by_class.items():
        if len(f) == 0:
            continue
        cen = f.float() if method == "none" else kmeans_centroids(f, math.ceil(rho * len(f)), device, seed=seed)
        if sigma and sigma > 0:
            cen = cen + sigma * torch.randn(cen.shape, generator=g)
        out[c] = cen.cpu()
    return out


def upload_bytes(centroids_by_client):
    """Bytes a client set uploads: float32 centroids only (labels are implied by the class grouping)."""
    return sum(t.numel() * 4 for cl in centroids_by_client.values() for t in cl.values())
