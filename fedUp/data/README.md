# data/

Everything the experiments read. Nothing here is committed; it is created by the scripts.

| folder | content | created by |
|---|---|---|
| `raw/` | MNIST, CIFAR-10, CIFAR-100 (torchvision files), `agnews/<text-model>_L128.npz` (tokenised AG News: input ids + attention mask) | `python scripts/download_datasets.py` |
| `processed/<benchmark>/fed_s<seed>_<hash>.pt` | the federated split of one seed: Dirichlet partition over clients, each client's 20 % per-class proxy hold-out, train / test split, plus the official test set | `python scripts/prepare_data.py` (or automatically on first use) |

Split rules (see `fedup_repro/data.py`)
* **client scenario**: train and test sets are merged, partitioned (Dirichlet 0.5, at least 100 samples per client), then every client keeps 70 % for training and 30 % as its own test split.
* **class / sample / client_backdoor scenarios**: only the training set is partitioned (99 % train / 1 % hold-out per client); evaluation uses the official test set.
* the cache key contains every setting that changes the split (clients, alpha, augmentation, text model, ...), so changing a setting never silently reuses a stale split.

Point the folder elsewhere with `FEDUP_DATA_DIR` (Kaggle: `/kaggle/working/...`, Colab: a Drive folder).
