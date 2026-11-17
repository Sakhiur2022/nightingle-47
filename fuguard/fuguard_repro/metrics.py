import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def predict(model, x, bs=1024):
    model.eval()
    return torch.cat([model(x[i:i + bs]).argmax(1) for i in range(0, len(x), bs)])


def evaluate_acc(model, data, bs=1024):
    x, y = data
    return (predict(model, x, bs) == y).float().mean().item()


def evaluate_asr(model, data, target_label, bs=1024):
    """Fraction of the poisoned test set predicted as the attacker's target class.
    NOTE: for the image tasks the poisoned test set contains *all* test images (target-class
    images are left untriggered), exactly as in the official code, so ASR has a non-zero floor."""
    x, _ = data
    return (predict(model, x, bs) == target_label).float().mean().item()


@torch.no_grad()
def _losses(model, x, y, bs=1024):
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="none")
    return torch.cat([crit(model(x[i:i + bs]), y[i:i + bs]) for i in range(0, len(x), bs)]).cpu().numpy()


def mia_loss_attack(model, member, nonmember, bs=1024):
    """Yeom et al. loss-threshold MIA, ported 1:1 from the official utils.py.
    members = forgotten client's local data, non-members = clean global test set.
    Threshold = median member loss; 'attack_acc' is what the paper plots as MIA."""
    ml = _losses(model, member[0], member[1], bs)
    nl = _losses(model, nonmember[0], nonmember[1], bs)
    thr = np.median(ml)
    member_pred = ml < thr
    nonmember_pred = nl < thr
    acc = (member_pred.sum() + (~nonmember_pred).sum()) / (len(member_pred) + len(nonmember_pred))
    return {"attack_acc": float(acc), "member_loss_mean": float(ml.mean()),
            "nonmember_loss_mean": float(nl.mean()), "threshold": float(thr)}
