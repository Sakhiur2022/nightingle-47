# artifacts/

Models produced along the way. They are what makes a restart cheap, and what makes FedUP's unlearning *reversible*.

| folder | content | notes |
|---|---|---|
| `checkpoints/<benchmark>_<view>_s<seed>_<hash>.pt` | the FedAvg model **before** unlearning (state dict, per-round test accuracy, FL time, communication, and the forgotten client's last local model for EraseClient) | shared by every unlearning method of the same seed; the hash covers every training setting |
| `checkpoints/..._retrain*.pt` | the Retrain model (FedAvg from scratch on the retained data) | the expensive baseline, cached |
| `filters/<key>_<scenario>_fedup.pt` | the trained FedUP plug-in filter (a 2-layer bottleneck, 128 KB for a 512-d feature at dim 32) | load it between extractor and classifier to unlearn, delete it to restore the original model |
| `features/` | optional per-client feature caches | not written by default |

`view` is `clean` (class unlearning trains on clean data), `client`, `sample` or `client_backdoor` (the attacked versions).
Override the location with `FEDUP_ARTIFACTS_DIR`.
