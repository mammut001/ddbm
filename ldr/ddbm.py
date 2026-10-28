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


class KarrasBridge:
    """DDBM as released (Zhou et al., `ddbm/karras_diffusion.py`, VE mode): sigma_t = t on
    [sigma_min, sigma_max], bridge marginal
        x_t = a x_T + b x_0 + c eps,  a = s^2/S^2,  b = 1 - a,  c^2 = s^2 (1 - a)   (S = sigma_max)
    EDM-style preconditioning D = c_skip x_t + c_out F(c_in x_t, c_noise, x_T), loss weight
    1 / c_out^2, sigma ~ U(sigma_min, sigma_max - 1e-4), and the hybrid sampler: a stochastic
    Euler-Maruyama 'churn' step followed by a Heun step of the probability-flow ODE on a
    Karras (rho = 7) schedule."""

    def __init__(self, sigma_min=0.002, sigma_max=80.0, sigma_data=0.5, cov_xy=0.0, rho=7.0):
        self.smin, self.smax, self.sd, self.cov, self.rho = sigma_min, sigma_max, sigma_data, cov_xy, rho

    def scalings(self, s):
        S, sd, cov = self.smax, self.sd, self.cov
        a = s ** 2 / S ** 2
        A = a ** 2 * sd ** 2 + (1 - a) ** 2 * sd ** 2 + 2 * a * (1 - a) * cov + s ** 2 * (1 - a)
        c_in = A.rsqrt()
        c_skip = ((1 - a) * sd ** 2 + a * cov) / A
        c_out = (a ** 2 * (sd ** 4 - cov ** 2) + sd ** 2 * s ** 2 * (1 - a)).sqrt() * c_in
        return c_skip, c_out, c_in

    def denoise(self, net, xt, s, xT, raw, cond):
        c_skip, c_out, c_in = [v.view(-1, 1, 1, 1) for v in self.scalings(s.float())]
        c_noise = 0.25 * torch.log(s.float() + 1e-44)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=xt.is_cuda):
            f = net(c_in * xt, c_noise, xT, raw, cond)
        return c_skip * xt + c_out * f.float(), c_out

    def loss(self, net, x0, xT, raw, cond):
        s = torch.rand(x0.shape[0], device=x0.device) * (self.smax - 1e-4 - self.smin) + self.smin
        a = (s ** 2 / self.smax ** 2).view(-1, 1, 1, 1)
        std = (s.view(-1, 1, 1, 1) ** 2 * (1 - a)).sqrt()
        xt = a * xT + (1 - a) * x0 + std * torch.randn_like(x0)
        d, c_out = self.denoise(net, xt, s, xT, raw, cond)
        w = 1 / c_out ** 2
        return (w * (d - x0) ** 2).mean(), d

    def sigmas(self, n, device):
        r = torch.linspace(0, 1, n, device=device)
        lo, hi = self.smin ** (1 / self.rho), (self.smax - 1e-4) ** (1 / self.rho)
        s = (hi + r * (lo - hi)) ** self.rho
        return torch.cat([s, s.new_zeros(1)])

    def drift(self, x, s, d, xT, stochastic):
        g2 = 2 * s
        score_x0 = (d - x) / s ** 2                     # grad log p(x_t | x_0)
        score_T = (xT - x) / (self.smax ** 2 - s ** 2)  # grad log p(x_T | x_t), the h-transform
        if stochastic:
            return -g2 * score_x0, g2
        return -0.5 * g2 * (score_x0 - score_T)

    @torch.no_grad()
    def sample(self, net, xT, raw, cond, steps: int = 10, churn: float = 0.33, **_):
        x = xT.clone()
        ss = self.sigmas(steps, xT.device)
        n = xT.shape[0]
        nfe = 0
        for i in range(len(ss) - 1):
            si, sn = ss[i], ss[i + 1]
            if churn > 0:  # stochastic Euler-Maruyama part of the step
                sh = (sn - si) * churn + si
                d, _ = self.denoise(net, x, si.expand(n), xT, raw, cond)
                dr, g2 = self.drift(x, si, d, xT, True)
                dt = sh - si
                x = x + dr * dt + torch.randn_like(x) * dt.abs().sqrt() * g2.sqrt()
                nfe += 1
            else:
                sh = si
            d, _ = self.denoise(net, x, sh.expand(n), xT, raw, cond)
            dr = self.drift(x, sh, d, xT, False)
            nfe += 1
            dt = sn - sh
            if sn == 0:
                x = x + dr * dt
            else:  # Heun correction
                x2 = x + dr * dt
                d2, _ = self.denoise(net, x2, sn.expand(n), xT, raw, cond)
                dr2 = self.drift(x2, sn, d2, xT, False)
                x = x + 0.5 * (dr + dr2) * dt
                nfe += 1
        self.nfe = nfe
        return x.clamp(-1, 1)


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
