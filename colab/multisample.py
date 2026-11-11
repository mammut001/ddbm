"""Average several samples of the two full DDBMs: how close does their posterior mean come to regression?

`mean` draws K samples (40 steps) from each full DDBM for some test pairs and scores, on the same samples:
single samples, the mean of the first 2, 4, ... samples of one model, the mean over both models, the
ensemble of that mean with the SID U-Net, and blends between a mean and one sample. `readout` scores the
one-step estimate D(x_T, T) of each DDBM, the network's own guess of the posterior mean, on all test pairs.
Finished images are appended to /content/results/ms_<mode>.jsonl, and a rerun continues from there.

    python colab/multisample.py readout
    python colab/multisample.py mean --per-scene 2 --k 4
"""

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/content/ldr-src")
import main as M  # noqa: E402
from ldr.data import PairDataset, spread  # noqa: E402
from ldr.metrics import scores  # noqa: E402
from ldr.models import BridgeDenoiser  # noqa: E402

RUNS, OUT = Path("/content/runs"), Path("/content/results")
DDBMS = {"naive": "ddbm_vp", "ustart": "ddbm_vp_unet"}  # x_T: the naive rendering, the SID U-Net output
CROPS = {"10074_00_0.1s", "10003_00_0.04s", "10003_00_0.1s"}  # the test images of the report's figures

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["readout", "mean"])
ap.add_argument("--per-scene", type=int, default=None)
ap.add_argument("--k", type=int, default=4, help="samples per DDBM, a power of two")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--out", default=None)
args = ap.parse_args()

cache, pairs = M.sid_split(type("A", (), {"cache": "/content/cache"})(), "test")
pairs = spread(pairs, args.per_scene)[:args.limit]
unet, uck = M.load_ckpt(RUNS / "sid" / "best.pt")
unet.eval()
nets = {}
for tag, run in DDBMS.items():
    ck = torch.load(RUNS / run / "best.pt", map_location=M.DEV)
    net = BridgeDenoiser(**ck.get("kwargs", {})).to(M.DEV).eval()
    net.load_state_dict(ck["ema"])
    nets[tag] = (net, M.make_bridge(ck["args"]), unet if ck["args"].get("unet_start") else None)


@torch.no_grad()
def readout(net, bridge, start, b):
    """D(x_T, T): the network's estimate of x_0 at the start of the bridge, one evaluation."""
    xT = M.bridge_start(b, start)
    s = bridge.sigmas(2, xT.device)[:1]
    d, _ = bridge.denoise(net, xT, s.expand(xT.shape[0]), xT, b["short"], b["cond"])
    return (d.float().clamp(-1, 1) + 1) / 2


def sample(tag, b, short, k):
    net, bridge, start = nets[tag]
    # the first sample of the naive start repeats the sample of colab/blend.py
    torch.manual_seed(zlib.crc32((short if (tag, k) == ("naive", 0) else f"{short}/{tag}/{k}").encode()))
    return M.predict(net, b, "ddbm", ddbm_steps=40, bridge=bridge, start=start)


def mean(xs):
    return sum(xs) / len(xs)


@torch.no_grad()
def estimates_readout(b, short):
    u = M.predict(unet, b, uck["model"])
    r = {tag: readout(*nets[tag], b) for tag in DDBMS}
    return {"unet": u, "naive_r": r["naive"], "ustart_r": r["ustart"], "unet+naive_r": (u + r["naive"]) / 2,
            "naive_r+ustart_r": (r["naive"] + r["ustart"]) / 2}


@torch.no_grad()
def estimates_mean(b, short):
    est = estimates_readout(b, short)
    u = est["unet"]
    S = {tag: [sample(tag, b, short, k) for k in range(args.k)] for tag in DDBMS}
    for tag in DDBMS:
        est.update({f"{tag}_s{k}": x for k, x in enumerate(S[tag])})
        n = 2
        while n <= args.k:
            est[f"{tag}_m{n}"] = mean(S[tag][:n])
            n *= 2
    n = 1
    while n <= args.k:  # n samples of each model
        est[f"both_m{2 * n}"] = mean(S["naive"][:n] + S["ustart"][:n])
        n *= 2
    m, s0 = est[f"both_m{2 * args.k}"], S["naive"][0]
    for w in (0.25, 0.5, 0.75):
        est[f"unet+mean_w{w}"] = (1 - w) * u + w * m             # ensemble of the U-Net and the sample mean
        est[f"mean+sample_w{w}"] = (1 - w) * m + w * s0          # from the DDBMs' mean towards one sample
        est[f"ens+sample_w{w}"] = (1 - w) * (u + m) / 2 + w * s0  # from the ensemble towards one sample
    for w in (0.25, 0.5, 0.625, 0.75):
        est[f"unet+sample_w{w}"] = (1 - w) * u + w * s0          # the blend of colab/blend.py
    return est


def crop_png(x, path):
    """The centre crop of colab/figures.py."""
    h, w = x.shape[-2:]
    y0, x0 = h // 2 - 384, w // 2 - 512
    a = x[0, :, y0:y0 + 768, x0:x0 + 1024].permute(1, 2, 0).float().clamp(0, 1).cpu().numpy()
    Image.fromarray((a * 255).astype(np.uint8)).save(path)


SHOW = ("unet", "naive_r", "naive_s0", f"naive_m{args.k}", f"both_m{2 * args.k}", "unet+mean_w0.5")
out = OUT / (args.out or f"ms_{args.mode}.jsonl")
(OUT / "fig").mkdir(parents=True, exist_ok=True)
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
print(f"{args.mode}: {len(pairs)} pairs, {len(done)} already done", flush=True)
for i, p in enumerate(pairs):
    if p.short in done:
        continue
    b = {k: (v[None].to(M.DEV) if torch.is_tensor(v) else v) for k, v in ds[i].items()}
    est = (estimates_mean if args.mode == "mean" else estimates_readout)(b, p.short)
    s = {name: scores(x, b["gt"], True) for name, x in est.items()}
    if args.mode == "mean" and p.short in CROPS:
        for name in ("naive_s0", f"naive_m{args.k}", f"both_m{2 * args.k}", "unet+mean_w0.5"):
            crop_png(est[name], OUT / "fig" / f"ms_{p.short}_{name}.png")
    with open(out, "a") as fh:
        fh.write(json.dumps({"short": p.short, "scene": p.scene, "ratio": p.ratio, "scores": s}) + "\n")
    print(i, p.short, " ".join(f"{n}:{s[n]['psnr']:.2f}/{s[n]['lpips']:.3f}" for n in SHOW if n in s), flush=True)
