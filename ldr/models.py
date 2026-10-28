"""Networks.

SIDUNet         the Learning-to-See-in-the-Dark baseline (Chen et al. 2018)
BridgeUNet      SID U-Net + bottleneck bridge refinement, FiLM on (ratio, noise), conditional
                refinement of the e4 skip and teacher-latent distillation from the long exposure
DDBMFeatureUNet the report's feature-space bridge: Brownian-bridge mixing of short and long
                encoder features at levels 3-5, time-conditioned decoder, x1-prediction
BridgeDenoiser  the network of the true pixel-space DDBM (ldr/ddbm.py)

All take the packed 4-channel short exposure at H/2 x W/2 and return RGB at H x W.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CH = (32, 64, 128, 256, 512)


def conv2(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.LeakyReLU(0.2, True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.LeakyReLU(0.2, True))


def pad_to(x, m=16):
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    return F.pad(x, (0, pw, 0, ph), mode="reflect"), (h, w)


class Encoder(nn.Module):
    def __init__(self, cin=4, ch=CH):
        super().__init__()
        self.blocks = nn.ModuleList()
        c = cin
        for co in ch:
            self.blocks.append(conv2(c, co))
            c = co

    def forward(self, x):
        feats = []
        for i, b in enumerate(self.blocks):
            if i:
                x = F.max_pool2d(x, 2)
            x = b(x)
            feats.append(x)
        return feats  # e1..e5


class Decoder(nn.Module):
    """Transposed-conv up, concat skip, two convs; final 1x1 to 12 channels + depth-to-space."""

    def __init__(self, ch=CH, temb: int = 0):
        super().__init__()
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.film = nn.ModuleList()
        for i in range(len(ch) - 1, 0, -1):
            self.ups.append(nn.ConvTranspose2d(ch[i], ch[i - 1], 2, stride=2))
            self.blocks.append(conv2(2 * ch[i - 1], ch[i - 1]))
            if temb:
                self.film.append(nn.Linear(temb, 2 * ch[i - 1]))
        self.out = nn.Conv2d(ch[0], 12, 1)

    def forward(self, feats, emb=None):
        x = feats[-1]
        for k, (up, blk) in enumerate(zip(self.ups, self.blocks)):
            x = up(x)
            x = blk(torch.cat([x, feats[-2 - k]], 1))
            if emb is not None:
                g, b = self.film[k](emb).chunk(2, 1)
                x = x * (1 + g[..., None, None]) + b[..., None, None]
        return F.pixel_shuffle(self.out(x), 2)


class SIDUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc, self.dec = Encoder(), Decoder()

    def forward(self, x, cond=None, long=None):
        x, (h, w) = pad_to(x)
        return {"rgb": self.dec(self.enc(x))[..., :2 * h, :2 * w]}


# ------------------------------------------------------------------ Bridge U-Net


class FiLMResBlock(nn.Module):
    """Residual 3x3 block whose features are scaled and shifted by the condition."""

    def __init__(self, c, cdim):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.film = nn.Linear(cdim, 2 * c)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x, cond):
        g, b = self.film(cond).chunk(2, 1)
        h = F.leaky_relu(self.c1(x), 0.2)
        h = h * (1 + g[..., None, None]) + b[..., None, None]
        return x + self.c2(h)


class CondMLP(nn.Module):
    def __init__(self, cin=2, d=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cin, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU())

    def forward(self, c):
        return self.net(c)


class BridgeUNet(nn.Module):
    """Encoder -> bridge (n FiLM residual blocks on e5, one on the e4 skip) -> decoder.

    With `long` given (training), the long exposure goes through the same encoder without
    gradient; the refined e4/e5 are pulled toward its e4/e5 (teacher-latent distillation).
    """

    def __init__(self, n_bridge: int = 2, film: bool = True, cdim: int = 64):
        super().__init__()
        self.enc, self.dec = Encoder(), Decoder()
        self.use_film = film
        self.cond = CondMLP(2, cdim)
        self.bridge = nn.ModuleList([FiLMResBlock(CH[4], cdim) for _ in range(n_bridge)])
        self.skip4 = FiLMResBlock(CH[3], cdim)

    def forward(self, x, cond, long=None):
        x, (h, w) = pad_to(x)
        c = self.cond(cond if self.use_film else torch.zeros_like(cond))
        f = self.enc(x)
        e5 = f[4]
        for blk in self.bridge:
            e5 = blk(e5, c)
        e4 = self.skip4(f[3], c)
        out = {"rgb": self.dec(f[:3] + [e4, e5])[..., :2 * h, :2 * w]}
        if long is not None:
            with torch.no_grad():
                t = self.enc(pad_to(long)[0])
            out["distill"] = sum(F.l1_loss(F.normalize(s, dim=1), F.normalize(tt, dim=1))
                                 for s, tt in ((e4, t[3]), (e5, t[4]))) / 2
        return out


# ------------------------------------------------------------------ DDBM-feature (report variant)


def timestep_embedding(t, dim=128):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * 1000 * freqs[None]
    return torch.cat([a.sin(), a.cos()], 1)


def rgb_to_pseudo_raw(y):
    """Report's re-encoding step: mosaic an sRGB estimate back to a linear RGGB pack."""
    lin = y.clamp(0, 1) ** 2.2
    r, g, b = lin[:, 0:1], lin[:, 1:2], lin[:, 2:3]
    return torch.cat([r[..., 0::2, 0::2], g[..., 0::2, 1::2], b[..., 1::2, 1::2], g[..., 1::2, 0::2]], 1)


class DDBMFeatureUNet(nn.Module):
    """f_t = (1-t) f_short + t f_long + sigma_max sqrt(t(1-t)) eps at levels 3-5;
    levels 1-2 always come from the short exposure. Predicts the clean RGB at every t."""

    def __init__(self, sigma_max: float = 0.1, levels=(2, 3, 4), temb: int = 128):
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder(temb=temb)
        self.temb = nn.Sequential(nn.Linear(temb, temb), nn.SiLU(), nn.Linear(temb, temb))
        self.sigma_max, self.levels, self.tdim = sigma_max, levels, temb

    def decode(self, fs, fl, t, noise=True):
        feats = list(fs)
        tt = t.view(-1, 1, 1, 1)
        for i in self.levels:
            mix = (1 - tt) * fs[i] + tt * fl[i]
            if noise:
                mix = mix + self.sigma_max * (tt * (1 - tt)).sqrt() * torch.randn_like(mix)
            feats[i] = mix
        return self.dec(feats, self.temb(timestep_embedding(t, self.tdim)))

    def forward(self, x, cond=None, long=None, steps: int = 3):
        x, (h, w) = pad_to(x)
        fs = self.enc(x)
        n = x.shape[0]
        if long is not None:  # training: random t, true long features
            t = torch.empty(n, device=x.device).uniform_(0.01, 0.99)
            fl = self.enc(pad_to(long)[0])
            return {"rgb": self.decode(fs, fl, t)[..., :2 * h, :2 * w]}
        y = self.decode(fs, fs, torch.zeros(n, device=x.device), noise=False)
        for k in range(1, steps + 1):
            fl = self.enc(rgb_to_pseudo_raw(y))
            y = self.decode(fs, fl, torch.full((n,), k / steps, device=x.device), noise=False)
        return {"rgb": y[..., :2 * h, :2 * w]}


# ------------------------------------------------------------------ true DDBM denoiser


class BridgeDenoiser(nn.Module):
    """D(x_t, t, x_T, raw): a time-conditioned U-Net on pixel-unshuffled (12-channel, H/2)
    images. Inputs: x_t and x_T (both 3-channel sRGB in [-1, 1]) and the packed short raw.
    Output: the predicted clean image x_0 (x_0-parameterisation of the bridge score)."""

    def __init__(self, ch=(64, 128, 256, 384, 512), temb: int = 256):
        super().__init__()
        self.enc = Encoder(cin=12 + 12 + 4, ch=ch)
        self.dec = Decoder(ch=ch, temb=temb)
        self.temb = nn.Sequential(nn.Linear(temb + 64, temb), nn.SiLU(), nn.Linear(temb, temb))
        self.cond = CondMLP(2, 64)
        self.tdim = temb

    def forward(self, xt, t, xT, raw, cond):
        inp = torch.cat([F.pixel_unshuffle(xt, 2), F.pixel_unshuffle(xT, 2), raw], 1)
        inp, (h, w) = pad_to(inp)
        emb = self.temb(torch.cat([timestep_embedding(t, self.tdim), self.cond(cond)], 1))
        return self.dec(self.enc(inp), emb)[..., :2 * h, :2 * w]


MODELS = {"sid": SIDUNet, "bridge": BridgeUNet, "ddbm_feature": DDBMFeatureUNet}
