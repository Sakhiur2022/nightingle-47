"""FedUP server side: train the plug-in filter on the aggregated class centroids (classifier and extractor frozen)."""
import time

import torch
import torch.nn.functional as F

from .models import Filter, FilteredModel
from .paths import FILTER_DIR
from .utils import sync


def centroid_batches(centroids_by_client, batch_size, generator=None):
    """Concatenate every client's centroids with their class labels, shuffle once and cut into full batches
    (the official code drops the incomplete last batch and replays the same batches every epoch)."""
    xs, ys = [], []
    for cl in centroids_by_client.values():
        for c, t in cl.items():
            xs.append(t.float())
            ys.append(torch.full((len(t),), int(c), dtype=torch.long))
    if not xs:
        raise ValueError("no centroids to train the filter on")
    x, y = torch.cat(xs), torch.cat(ys)
    perm = torch.randperm(len(x), generator=generator)
    x, y = x[perm], y[perm]
    nb = len(x) // batch_size
    if nb == 0:                               # fewer centroids than one batch: use them all
        return [(x, y)]
    return [(x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]) for i in range(nb)]


def train_filter(model, centroids_by_client, cfg, device, seed=0, tag=None, save=True):
    """Train M_Fi with  L = alpha * CE(M_cl(M_Fi(mu)), y) + (1 - alpha) * MSE(M_Fi(mu), mu).
    Returns dict(filtered=FilteredModel, filter=Filter, time_s, history, n_centroids, filter_bytes)."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    filt = Filter(model.feat_dim, cfg["out_dim"], cfg["filter_out_relu"]).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    opt = torch.optim.SGD(filt.parameters(), lr=cfg["lr"])        # [repo] plain SGD with the dataset's learning rate
    batches = centroid_batches(centroids_by_client, cfg["filter_batch"], g)
    batches = [(a.to(device), b.to(device)) for a, b in batches]
    alpha, history, total = float(cfg["loss_alpha"]), [], 0.0
    for ep in range(int(cfg["filter_epochs"])):
        sync(device)
        t0 = time.perf_counter()
        if cfg["filter_reshuffle"] and ep > 0:
            batches = [(a.to(device), b.to(device)) for a, b in centroid_batches(centroids_by_client, cfg["filter_batch"], g)]
        filt.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for xb, yb in batches:
            opt.zero_grad(set_to_none=True)
            z = filt(xb)
            logits = model.classifier(z)
            loss = alpha * F.cross_entropy(logits, yb) + (1 - alpha) * F.mse_loss(z, xb)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            correct += int((logits.argmax(1) == yb).sum())
            seen += len(yb)
        sync(device)
        total += time.perf_counter() - t0                         # [repo] the reported time sums the filter-training epochs
        history.append((ep + 1, loss_sum / len(batches), correct / max(seen, 1)))
    filt.eval()
    if save and tag:
        FILTER_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(filt.state_dict(), FILTER_DIR / f"{tag}.pt")
    return dict(filtered=FilteredModel(model, filt).eval(), filter=filt, time_s=total, history=history,
                n_centroids=sum(len(t) for cl in centroids_by_client.values() for t in cl.values()),
                filter_bytes=filt.nbytes())
