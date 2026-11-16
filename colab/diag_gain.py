"""Is the brightness (and colour) of the naive rendering what breaks the naive-start DDBM on other cameras?

Oracle test: the naive rendering x_T of every ELD test image is rescaled so that its mean matches the target's, once
over all channels ("gain") and once per channel ("rgb gain"), and the one-step estimate of the naive-start DDBM
(runs/ddbm_vp) is scored on the rescaled start. The oracle uses the target, so it is a diagnosis and not a method.
If the PSNR recovers towards the SID U-Net's, the start image is the problem.

    python colab/diag_gain.py
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/content/ldr-src")
import main as M  # noqa: E402
from ldr.data import PairDataset  # noqa: E402
from ldr.metrics import scores  # noqa: E402
from ldr.models import BridgeDenoiser  # noqa: E402

OUT = Path("/content/results")
OUT.mkdir(parents=True, exist_ok=True)
ck = torch.load("/content/runs/ddbm_vp/best.pt", map_location=M.DEV)
net = BridgeDenoiser(**ck.get("kwargs", {})).to(M.DEV).eval()
net.load_state_dict(ck["ema"])
bridge = M.make_bridge(ck["args"])
unet, uck = M.load_ckpt(Path("/content/runs/sid/best.pt"))
unet.eval()


@torch.no_grad()
def readout(xT, b):
    s = bridge.sigmas(2, xT.device)[:1]
    d, _ = bridge.denoise(net, xT, s.expand(xT.shape[0]), xT, b["short"], b["cond"])
    return (d.float().clamp(-1, 1) + 1) / 2


a = type("A", (), {"cache": "/content/cache"})()
rows = []
for cam in M.ELD_CAMERAS:
    cache, pairs = M.eld(a, [cam], M.ELD_TEST, M.ELD_EVAL_IDS)
    ds = PairDataset(cache, pairs, ps=None, train=False)
    for i, p in enumerate(pairs):
        b = {k: (v[None].to(M.DEV) if torch.is_tensor(v) else v) for k, v in ds[i].items()}
        with torch.no_grad():
            x = (M.bridge_start(b, None).float() + 1) / 2
            gt = b["gt"].float()
            g = gt.mean() / x.mean().clamp_min(1e-6)
            gc = gt.mean(dim=(0, 2, 3), keepdim=True) / x.mean(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
            est = {"naive": readout(x * 2 - 1, b),
                   "gain": readout((x * g).clamp(0, 1) * 2 - 1, b),
                   "rgb_gain": readout((x * gc).clamp(0, 1) * 2 - 1, b),
                   "unet": M.predict(unet, b, uck["model"])}
        rows.append({"short": p.short, "camera": cam, "scene": p.scene,
                     "scores": {n: scores(v, b["gt"], True) for n, v in est.items()}})
        print(cam, i, " ".join(f"{n}:{rows[-1]['scores'][n]['psnr']:.2f}" for n in est), flush=True)
(OUT / "diag_gain.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
