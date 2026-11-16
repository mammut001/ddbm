"""Ensemble control for the one-step average of the two DDBMs (colab/multisample.py readout).

The average of the two DDBMs' one-step estimates beats the SID U-Net, but any two different networks gain
from averaging. This script scores, on all test pairs, the first SID U-Net (runs/sid, seed 0), a second one
trained with another seed (runs/sid_s1), and the average of the two, so that the ensemble gain of two
U-Nets can be put next to the 0.58 dB of the DDBM pair.

Finished images are appended to /content/results/ens_unet.jsonl, and a rerun continues from there.

    python colab/unet_ensemble.py [--limit N]
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/content/ldr-src")
import main as M  # noqa: E402
from ldr.data import PairDataset, spread  # noqa: E402
from ldr.metrics import scores  # noqa: E402

RUNS, OUT = Path("/content/runs"), Path("/content/results")
NETS = {"unet_a": "sid", "unet_b": "sid_s1"}

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--out", default="ens_unet.jsonl")
args = ap.parse_args()

cache, pairs = M.sid_split(type("A", (), {"cache": "/content/cache"})(), "test")
pairs = spread(pairs, None)[:args.limit]
nets = {}
for tag, run in NETS.items():
    net, ck = M.load_ckpt(RUNS / run / "best.pt")
    nets[tag] = (net.eval(), ck["model"])

OUT.mkdir(parents=True, exist_ok=True)
out = OUT / args.out
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
print(f"ensemble control: {len(pairs)} pairs, {len(done)} already done", flush=True)
for i, p in enumerate(pairs):
    if p.short in done:
        continue
    b = {k: (v[None].to(M.DEV) if torch.is_tensor(v) else v) for k, v in ds[i].items()}
    with torch.no_grad():
        est = {tag: M.predict(net, b, kind) for tag, (net, kind) in nets.items()}
    est["unet_ab"] = (est["unet_a"] + est["unet_b"]) / 2
    s = {name: scores(x, b["gt"], True) for name, x in est.items()}
    with open(out, "a") as fh:
        fh.write(json.dumps({"short": p.short, "scene": p.scene, "ratio": p.ratio, "scores": s}) + "\n")
    print(i, p.short, " ".join(f"{n}:{s[n]['psnr']:.2f}/{s[n]['lpips']:.3f}" for n in est), flush=True)
