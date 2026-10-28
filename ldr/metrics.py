from __future__ import annotations

import torch
from pytorch_msssim import ms_ssim, ssim


def psnr(a, b):
    mse = ((a - b) ** 2).flatten(1).mean(1)
    return (10 * torch.log10(1.0 / mse.clamp_min(1e-10)))


@torch.no_grad()
def scores(pred, gt):
    pred, gt = pred.float().clamp(0, 1), gt.float().clamp(0, 1)
    return {"psnr": psnr(pred, gt).mean().item(),
            "ssim": ssim(pred, gt, data_range=1.0, size_average=True).item(),
            "msssim": ms_ssim(pred, gt, data_range=1.0, size_average=True).item(),
            "l1": (pred - gt).abs().mean().item()}


def msssim_loss(pred, gt):
    return 1 - ms_ssim(pred.float().clamp(0, 1), gt.float(), data_range=1.0, size_average=True)
