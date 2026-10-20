"""Generative proxy construction (FuGuard step 1): encode -> PCA-guided latent shift -> decode."""
import random
import numpy as np
import torch
from sklearn.decomposition import PCA


class _Out:  # tiny stand-in for diffusers' output objects
    def __init__(self, **kw):
        self.__dict__.update(kw)


class StubVAE(torch.nn.Module):
    """Offline stand-in for the Stable-Diffusion VAE (same interface). ONLY for smoke tests."""

    def __init__(self):
        super().__init__()
        self.enc = torch.nn.Conv2d(3, 4, 8, stride=8)
        self.dec = torch.nn.ConvTranspose2d(4, 3, 8, stride=8)

    def encode(self, x):
        return _Out(latent_dist=_Out(mean=self.enc(x)))

    def decode(self, z):
        return _Out(sample=self.dec(z))


def load_vae(vae_id, device):
    if vae_id == "stub":
        return StubVAE().to(device).eval()
    from diffusers import AutoencoderKL  # needs internet on first call (Hugging Face)
    return AutoencoderKL.from_pretrained(vae_id).to(device).eval()


def _pca_shift(z_flat, cfg):
    """z_flat: [N, D] tensor. Returns the PCA direction p_r (shape [D]) as a tensor."""
    k = min(cfg["pca_components"], z_flat.shape[0], z_flat.shape[1])
    pca = PCA(n_components=k).fit(z_flat.detach().cpu().numpy())
    comps = torch.tensor(pca.components_, device=z_flat.device, dtype=z_flat.dtype)
    r = min(cfg["pca_dir_idx"], k - 1)
    return comps[r]


def image_proxy(vae, x, y, cfg, device):
    """Official main_image/FuGuard.py logic. NOTE (kept as in the repo): inputs are the
    mean/std-normalised images fed straight to the SD-VAE and the decoded proxies are clamped to [0, 1]."""
    n = int(cfg["samples_scale"] * len(x))
    if n < 2:
        raise ValueError(f"target client has too little data ({len(x)}) for a proxy of {n} samples")
    idx = random.sample(range(len(x)), n)
    xs, ys = x[idx], y[idx]
    with torch.no_grad():
        z = torch.cat([vae.encode(xs[i:i + cfg["gen_bs"]]).latent_dist.mean
                       for i in range(0, n, cfg["gen_bs"])])
        direction = _pca_shift(z.view(n, -1), cfg).view_as(z[0])
        z_new = z + cfg["pca_alpha"] * direction
        out = torch.cat([vae.decode(z_new[i:i + cfg["gen_bs"]]).sample
                         for i in range(0, n, cfg["gen_bs"])])
    return out.clamp(0, 1).detach(), ys


def awe_proxy(x, y, cfg, device):
    """Official main_awe/FuGuard.py logic with one fix: the repo encoded only the first 100
    samples but kept all labels (shape mismatch whenever the 10% subset != 100 samples)."""
    from .models import LSTMVAE, vae_loss
    n = int(cfg["samples_scale"] * len(x))
    if n < 2:
        raise ValueError(f"target client has too little data ({len(x)}) for a proxy of {n} samples")
    idx = random.sample(range(len(x)), n)
    xs, ys = x[idx], y[idx]
    seq_len, in_dim = xs.shape[1], xs.shape[2]
    vae = LSTMVAE(input_dim=in_dim, hidden_dim=cfg["vae_hidden"], latent_dim=cfg["vae_latent"]).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=cfg["vae_lr"])
    bs = cfg["gen_bs"]
    for _ in range(cfg["vae_epochs"]):
        vae.train()
        for i in range(0, n, bs):
            xb = xs[i:i + bs]
            rec, mu, logvar = vae(xb)
            loss = vae_loss(xb, rec, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
    vae.eval()
    with torch.no_grad():
        mu, _ = vae.encode(xs)
        direction = _pca_shift(mu, cfg)
        z_new = mu + cfg["pca_alpha"] * direction
        out = vae.decode(z_new, seq_len=seq_len).clamp(0, 1)
    return out.detach(), ys
