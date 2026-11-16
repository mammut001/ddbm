"""Few-shot adaptation to a new camera, regression U-Net against the DDBM that starts from its output.

For each ELD camera and each K (scenes 1..K of that camera as support, scenes 7-10 for test, full images) the SID U-Net is
fine-tuned with L1 (the recipe of `main.py fewshot`), and the full DDBM started from the U-Net output (runs/ddbm_vp_unet)
is fine-tuned with its bridge loss for the same number of steps, its start image coming from the frozen SID U-Net. Scored:
the adapted U-Net (U), the one-step estimate of the adapted DDBM (D), their average (UD), and the one-step estimate of the
unadapted DDBM fed with the adapted U-Net's output (DU). K=0 is the zero-shot result of colab/crosscam.py.
One line per (camera, K) is appended to /content/results/fewshot_ddbm.jsonl.

    python colab/fewshot_ddbm.py [--shots 0 1 3 6] [--steps 200]
"""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/content/ldr-src")
import main as M  # noqa: E402
from ldr.data import PairDataset  # noqa: E402
from ldr.metrics import scores  # noqa: E402
from ldr.models import BridgeDenoiser  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shots", type=int, nargs="+", default=[0, 1, 3, 6])
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--ps", type=int, default=512)
ap.add_argument("--out", default="fewshot_ddbm.jsonl")
ap.add_argument("--objective", choices=["bridge", "readout"], default="bridge",
                help="bridge: the bridge loss; readout: L1 between the one-step estimate and the target")
args = ap.parse_args()

RUNS, OUT = Path("/content/runs"), Path("/content/results")
OUT.mkdir(parents=True, exist_ok=True)
out = OUT / args.out
a = type("A", (), {"cache": "/content/cache"})()
unet0, uck = M.load_ckpt(RUNS / "sid" / "best.pt")
unet0.eval().requires_grad_(False)
dck = torch.load(RUNS / "ddbm_vp_unet" / "best.pt", map_location=M.DEV)
bridge = M.make_bridge(dck["args"])


def fresh_ddbm():
    net = BridgeDenoiser(**dck.get("kwargs", {})).to(M.DEV)
    net.load_state_dict(dck["ema"])
    return net


@torch.no_grad()
def readout(net, start, b):
    net.eval()
    xT = M.bridge_start(b, start)
    s = bridge.sigmas(2, xT.device)[:1]
    d, _ = bridge.denoise(net, xT, s.expand(xT.shape[0]), xT, b["short"], b["cond"])
    return (d.float().clamp(-1, 1) + 1) / 2


def finetune_unet(ds):
    model = copy.deepcopy(unet0).requires_grad_(True).train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for _ in range(args.steps):
        b = M.batch_of(ds, random.sample(range(len(ds)), min(args.bs, len(ds))))
        loss = M.task_loss(model, dict(model.named_parameters()), b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model.eval().requires_grad_(False)


def finetune_ddbm(ds):
    net = fresh_ddbm().train()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0)
    for _ in range(args.steps):
        b = M.batch_of(ds, random.sample(range(len(ds)), min(args.bs, len(ds))))
        x0, xT = b["gt"] * 2 - 1, M.bridge_start(b, unet0)
        with torch.autocast(**M.AMP):
            if args.objective == "bridge":
                loss, _ = bridge.loss(net, x0, xT, b["short"], b["cond"])
            else:  # the network's own estimate of x_0 at the start of the bridge, as scored
                s = bridge.sigmas(2, xT.device)[:1]
                d, _ = bridge.denoise(net, xT, s.expand(xT.shape[0]), xT, b["short"], b["cond"])
                loss = F.l1_loss((d.float().clamp(-1, 1) + 1) / 2, b["gt"])
        opt.zero_grad(set_to_none=True)
        loss.float().backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    return net.eval().requires_grad_(False)


done = set()
if out.exists():
    for line in out.read_text().splitlines():
        try:
            r = json.loads(line)
            done.add((r["camera"], r["k"]))
        except json.JSONDecodeError:
            pass
base_ddbm = fresh_ddbm().eval().requires_grad_(False)
for cam in M.ELD_CAMERAS:
    for k in args.shots:
        if (cam, k) in done:
            continue
        M.seed_all(0)
        cache, test = M.eld(a, [cam], M.ELD_TEST, M.ELD_EVAL_IDS)
        if k:
            _, sup = M.eld(a, [cam], M.ELD_TRAIN[:k])
            ds = PairDataset(cache, sup, ps=args.ps)
            unet, ddbm = finetune_unet(ds), finetune_ddbm(ds)
        else:
            unet, ddbm = unet0, base_ddbm
        tds = PairDataset(cache, test, ps=None, train=False)
        rows = []
        for i, p in enumerate(test):
            b = {kk: (v[None].to(M.DEV) if torch.is_tensor(v) else v) for kk, v in tds[i].items()}
            with torch.no_grad():
                u = M.predict(unet, b, uck["model"])
                d = readout(ddbm, unet0, b)
                du = readout(base_ddbm, unet, b)
            est = {"U": u, "D": d, "UD": (u + d) / 2, "DU": du}
            rows.append({"short": p.short, "scene": p.scene, "scores": {n: scores(x, b["gt"], True) for n, x in est.items()}})
        with open(out, "a") as fh:
            fh.write(json.dumps({"camera": cam, "k": k, "steps": args.steps, "lr": args.lr, "rows": rows}) + "\n")
        m = {n: sum(r["scores"][n]["psnr"] for r in rows) / len(rows) for n in ("U", "D", "UD", "DU")}
        print(cam, k, " ".join(f"{n}:{v:.2f}" for n, v in m.items()), flush=True)
