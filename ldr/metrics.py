from __future__ import annotations

import torch
from pytorch_msssim import ms_ssim, ssim


def psnr(a, b):
    mse = ((a - b) ** 2).flatten(1).mean(1)
    return (10 * torch.log10(1.0 / mse.clamp_min(1e-10)))


_LPIPS = None


def lpips_alex(pred, gt):
    """LPIPS (AlexNet, v0.1) on [0, 1] images; the network is built on first use."""
    global _LPIPS
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net="alex", verbose=False).to(pred.device).eval()
    return _LPIPS(pred * 2 - 1, gt * 2 - 1).mean()


@torch.no_grad()
def scores(pred, gt, with_lpips=False):
    pred, gt = pred.float().clamp(0, 1), gt.float().clamp(0, 1)
    out = {"psnr": psnr(pred, gt).mean().item(),
           "ssim": ssim(pred, gt, data_range=1.0, size_average=True).item(),
           "msssim": ms_ssim(pred, gt, data_range=1.0, size_average=True).item(),
           "l1": (pred - gt).abs().mean().item()}
    if with_lpips:
        out["lpips"] = lpips_alex(pred, gt).item()
    return out


def msssim_loss(pred, gt):
    return 1 - ms_ssim(pred.float().clamp(0, 1), gt.float(), data_range=1.0, size_average=True)
