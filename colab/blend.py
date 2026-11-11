"""Blend the SID U-Net with the full DDBM, x = (1 - w) * U-Net + w * DDBM, on full-resolution SID pairs.

Every weight in WEIGHTS is scored on the same DDBM sample (40 steps, seeded by the image name). Each
finished image is appended to /content/results/blend_<split>.jsonl, and a rerun continues from there.
Pairs are interleaved across scenes and exposure ratios, so every prefix of the run covers all scenes.

    python colab/blend.py val 2     # two pairs per validation scene, to choose the weight
    python colab/blend.py test      # all test pairs
"""

import json
import sys
import zlib
from pathlib import Path

import torch

sys.path.insert(0, "/content/ldr-src")
import main as M  # noqa: E402
from ldr.data import PairDataset, spread  # noqa: E402
from ldr.metrics import scores  # noqa: E402
from ldr.models import BridgeDenoiser  # noqa: E402

RUNS, OUT = Path("/content/runs"), Path("/content/results")
WEIGHTS = (0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1)


split = sys.argv[1]
cache, pairs = M.sid_split(type("A", (), {"cache": "/content/cache"})(), split)
pairs = spread(pairs, int(sys.argv[2]) if len(sys.argv) > 2 else None)

unet, uck = M.load_ckpt(RUNS / "sid" / "best.pt")
unet.eval()
ck = torch.load(RUNS / "ddbm_vp" / "best.pt", map_location=M.DEV)
ddbm = BridgeDenoiser(**ck.get("kwargs", {})).to(M.DEV).eval()
ddbm.load_state_dict(ck["ema"])
bridge = M.make_bridge(ck["args"])

out = OUT / f"blend_{split}.jsonl"
OUT.mkdir(parents=True, exist_ok=True)
rows = []
if out.exists():
    for line in out.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # a line cut off when the machine stopped
            pass
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
done = {r["short"] for r in rows}

ds = PairDataset(cache, pairs, ps=None, train=False)
print(f"{split}: {len(pairs)} pairs, {len(done)} already done", flush=True)
for i, p in enumerate(pairs):
    if p.short in done:
        continue
    b = {k: (v[None].to(M.DEV) if torch.is_tensor(v) else v) for k, v in ds[i].items()}
    u = M.predict(unet, b, uck["model"])
    torch.manual_seed(zlib.crc32(p.short.encode()))
    d = M.predict(ddbm, b, "ddbm", ddbm_steps=40, bridge=bridge)
    s = {str(w): scores((1 - w) * u + w * d, b["gt"], True) for w in WEIGHTS}
    with open(out, "a") as fh:
        fh.write(json.dumps({"short": p.short, "scene": p.scene, "ratio": p.ratio, "scores": s}) + "\n")
    print(i, p.short, " ".join(f"{w}:{s[str(w)]['psnr']:.2f}/{s[str(w)]['lpips']:.3f}" for w in (0, 0.25, 0.5, 1)),
          flush=True)
