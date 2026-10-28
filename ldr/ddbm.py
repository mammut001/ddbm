"""Pixel-space Denoising Diffusion Bridge Model (Zhou et al., ICLR 2024), VE / Brownian bridge.

The bridge runs between x_0 = clean sRGB target and x_T = a naive rendering of the amplified
short exposure (both in [-1, 1], time T = 1). With base variance s2 * t, the bridge marginal is

    x_t | x_0, x_T  ~  N((1 - t) x_0 + t x_T,  s2 t (1 - t) I)

and the network D predicts x_0 (the score of the bridge is (mean(D) - x_t) / var, so an L2
loss on x_0 is denoising bridge score matching up to a per-t weight). Sampling walks t from 1
to 0 with the exact Markov-bridge posterior

    x_s | x_t, x_0  ~  N((1 - s/t) x_0 + (s/t) x_t,  s2 s (t - s) / t I)

(ancestral, eta = 1), or its deterministic implicit counterpart (eta = 0, DBIM, Zheng et al.
2024), which reuses the predicted noise. Each step costs one network evaluation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def naive_render(short, wb):
    """Packed amplified RGGB in [0, 1] -> full-resolution sRGB guess in [-1, 1]."""
    rgb = torch.stack([short[:, 0], (short[:, 1] + short[:, 3]) / 2, short[:, 2]], 1)
    rgb = (rgb * wb[:, :, None, None]).clamp(0, 1) ** (1 / 2.2)
    rgb = F.interpolate(rgb, scale_factor=2, mode="bilinear", align_corners=False)
    return rgb * 2 - 1


class Bridge:
    def __init__(self, s2: float = 0.5, t_eps: float = 1e-3):
        self.s2, self.t_eps = s2, t_eps

    def marginal(self, x0, xT, t):
        tt = t.view(-1, 1, 1, 1)
        mean = (1 - tt) * x0 + tt * xT
        std = (self.s2 * tt * (1 - tt)).sqrt()
        return mean, std

    def loss(self, net, x0, xT, raw, cond):
        t = torch.rand(x0.shape[0], device=x0.device) * (1 - 2 * self.t_eps) + self.t_eps
        mean, std = self.marginal(x0, xT, t)
        xt = mean + std * torch.randn_like(x0)
        pred = net(xt, t, xT, raw, cond)
        return F.mse_loss(pred, x0), pred

    @torch.no_grad()
    def sample(self, net, xT, raw, cond, steps: int = 10, eta: float = 1.0):
        ts = torch.linspace(1 - self.t_eps, 0, steps + 1, device=xT.device)
        n = xT.shape[0]
        x = xT.clone()
        x0 = xT
        for i in range(steps):
            t, s = ts[i], ts[i + 1]
            x0 = net(x, t.expand(n), xT, raw, cond).clamp(-1, 1)
            if s <= 0:
                return x0
            mean_s = (1 - s) * x0 + s * xT
            std_s = (self.s2 * s * (1 - s)).sqrt()
            if eta > 0 or i == 0:
                # exact posterior of the Brownian bridge pinned at x0 (time 0) and x (time t)
                m = (1 - s / t) * x0 + (s / t) * x
                v = self.s2 * s * (t - s) / t
                x = m + v.sqrt() * torch.randn_like(x)
            else:
                mean_t = (1 - t) * x0 + t * xT
                eps = (x - mean_t) / (self.s2 * t * (1 - t)).sqrt()
                x = mean_s + std_s * eps
        return x0
