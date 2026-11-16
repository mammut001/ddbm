"""Why does the naive-start DDBM fail on other cameras? Compare the bridge start x_T (the naive rendering) and the
SID U-Net output with the target, per ELD camera: brightness and colour (per-channel mean ratio to the target) and
PSNR of each. If x_T is off in brightness or colour on the cameras where the DDBM fails and not on SID's Sony, the
naive rendering is out of distribution there.

    python colab/diag_naive.py
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/content/ldr-src")
import main as M  # noqa: E402
from ldr.data import PairDataset  # noqa: E402

OUT = Path("/content/results")
OUT.mkdir(parents=True, exist_ok=True)
unet, ck = M.load_ckpt(Path("/content/runs/sid/best.pt"))
unet.eval()
a = type("A", (), {"cache": "/content/cache"})()
rows = []
for cam in M.ELD_CAMERAS:
    cache, pairs = M.eld(a, [cam], M.ELD_TEST, M.ELD_EVAL_IDS)
    ds = PairDataset(cache, pairs, ps=None, train=False)
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
        rows.append({"short": p.short, "camera": cam, "scene": p.scene, "ratio": p.ratio, "gt_mean": float(gt.mean()),
                     "xT": stat(xt), "unet": stat(un), "wb": [float(v) for v in torch.as_tensor(b["wb"]).flatten()[:4]]})
        print(cam, i, p.short, f"xT mean ratio {rows[-1]['xT']['mean_ratio']:.2f} psnr {rows[-1]['xT']['psnr']:.1f}", flush=True)
(OUT / "diag_naive.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
