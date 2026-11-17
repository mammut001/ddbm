"""The SID side of colab/diag_naive.py: brightness and colour of the naive rendering x_T and of the SID U-Net output
against the target on the SID Sony test pairs (two per scene), to compare with the ELD cameras where the naive-start
DDBM fails. If x_T is close to the target on Sony, the 2.8 to 4.3 times found on ELD is a transfer problem.

    python colab/diag_naive_sid.py
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/content/ldr-src")
import main as M  # noqa: E402
from ldr.data import PairDataset, spread  # noqa: E402

OUT = Path("/content/results")
OUT.mkdir(parents=True, exist_ok=True)
unet, ck = M.load_ckpt(Path("/content/runs/sid/best.pt"))
unet.eval()
cache, pairs = M.sid_split(type("A", (), {"cache": "/content/cache"})(), "test")
pairs = spread(pairs, 2)
ds = PairDataset(cache, pairs, ps=None, train=False)
rows = []
for i, p in enumerate(pairs):
    b = {k: (v[None].to(M.DEV) if torch.is_tensor(v) else v) for k, v in ds[i].items()}
    with torch.no_grad():
        xt = (M.bridge_start(b, None).float() + 1) / 2
        un = M.predict(unet, b, ck["model"]).float()
    gt = b["gt"].float()

    def stat(x):
        ch = (x.mean(dim=(0, 2, 3)) / gt.mean(dim=(0, 2, 3)).clamp_min(1e-6)).tolist()
        mse = ((x.clamp(0, 1) - gt) ** 2).mean().item()
        return {"mean_ratio": float(x.mean() / gt.mean().clamp_min(1e-6)), "rgb_ratio": ch,
                "psnr": -10 * torch.log10(torch.tensor(max(mse, 1e-12))).item()}
    rows.append({"short": p.short, "camera": "SonyA7S2", "scene": p.scene, "ratio": p.ratio, "gt_mean": float(gt.mean()),
                 "xT": stat(xt), "unet": stat(un)})
    print(i, p.short, f"xT mean ratio {rows[-1]['xT']['mean_ratio']:.2f} psnr {rows[-1]['xT']['psnr']:.1f}", flush=True)
(OUT / "diag_naive_sid.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
