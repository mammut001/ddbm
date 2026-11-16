"""Zero-shot transfer to other cameras: do the DDBM estimates keep their edge over regression on sensors never seen?

Scores, on the ELD test scenes of every ELD camera (scenes 7-10, ratios 100 and 200 as in the ELD protocol), without any
adaptation: the SID U-Net (seed 0), the second SID U-Net (seed 1), their average, the one-step estimates of the two full
DDBMs and their average (the best result of the report on SID). All models were trained on SID Sony only.
Finished images are appended to /content/results/crosscam.jsonl and a rerun continues from there.

    python colab/crosscam.py [--limit N]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/content/ldr-src")
import main as M  # noqa: E402
from ldr.data import PairDataset  # noqa: E402
from ldr.metrics import scores  # noqa: E402
from ldr.models import BridgeDenoiser  # noqa: E402

RUNS, OUT = Path("/content/runs"), Path("/content/results")
DDBMS = {"naive": "ddbm_vp", "ustart": "ddbm_vp_unet"}

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--out", default="crosscam.jsonl")
args = ap.parse_args()

unets = {}
for tag, run in {"unet_a": "sid", "unet_b": "sid_s1"}.items():
    net, ck = M.load_ckpt(RUNS / run / "best.pt")
    unets[tag] = (net.eval(), ck["model"])
nets = {}
for tag, run in DDBMS.items():
    ck = torch.load(RUNS / run / "best.pt", map_location=M.DEV)
    net = BridgeDenoiser(**ck.get("kwargs", {})).to(M.DEV).eval()
    net.load_state_dict(ck["ema"])
    nets[tag] = (net, M.make_bridge(ck["args"]), unets["unet_a"][0] if ck["args"].get("unet_start") else None)


@torch.no_grad()
def readout(net, bridge, start, b):
    """D(x_T, T): the network's own estimate of x_0 at the start of the bridge, one evaluation."""
    xT = M.bridge_start(b, start)
    s = bridge.sigmas(2, xT.device)[:1]
    d, _ = bridge.denoise(net, xT, s.expand(xT.shape[0]), xT, b["short"], b["cond"])
    return (d.float().clamp(-1, 1) + 1) / 2


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

a = type("A", (), {"cache": "/content/cache"})()
for cam in M.ELD_CAMERAS:
    cache, pairs = M.eld(a, [cam], M.ELD_TEST, M.ELD_EVAL_IDS)
    pairs = pairs[:args.limit]
    ds = PairDataset(cache, pairs, ps=None, train=False)
    print(f"{cam}: {len(pairs)} pairs, {len(done)} done in total", flush=True)
    for i, p in enumerate(pairs):
        if p.short in done:
            continue
        b = {k: (v[None].to(M.DEV) if torch.is_tensor(v) else v) for k, v in ds[i].items()}
        with torch.no_grad():
            est = {tag: M.predict(net, b, kind) for tag, (net, kind) in unets.items()}
            est["unet_ab"] = (est["unet_a"] + est["unet_b"]) / 2
            r = {tag: readout(*nets[tag], b) for tag in DDBMS}
        est.update({"naive_r": r["naive"], "ustart_r": r["ustart"], "ddbm_pair": (r["naive"] + r["ustart"]) / 2})
        s = {name: scores(x, b["gt"], True) for name, x in est.items()}
        with open(out, "a") as fh:
            fh.write(json.dumps({"short": p.short, "camera": cam, "scene": p.scene, "ratio": p.ratio, "scores": s}) + "\n")
        print(cam, i, p.short, " ".join(f"{n}:{s[n]['psnr']:.2f}/{s[n]['lpips']:.3f}" for n in est), flush=True)
