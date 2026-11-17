"""A grid of few-shot adaptation recipes for the SID U-Net on new cameras, with seeds.

Data: SIED Sony (A7R III) from colab/sied_prep.py ("sied": K support images, 180 test images) and the ELD cameras
("eld": K support scenes of each camera, scenes 7-10 for test). For every recipe, seed and K the U-Net is fine-tuned and
scored, alone and with the unadapted DDBM (runs/ddbm_vp_unet) fed with its output, on the same test images. Recipe =
which part is trained (all / dec / enc), the loss (l1, or l1 + 0.1 MS-SSIM as in training), cosine or constant learning rate,
learning rate, steps. One line per (data, recipe, seed, k) is appended to /content/results/adapt_grid_<data>.jsonl
(per-image PSNR and LPIPS included, for paired comparisons); a rerun continues where it stopped.

    python colab/adapt_grid.py sied --seeds 0 1 2
    python colab/adapt_grid.py eld --seeds 0 1 2
"""
import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/content/ldr-src")
import main as M  # noqa: E402
from ldr import raw as R  # noqa: E402
from ldr.data import Cache, PairDataset  # noqa: E402
from ldr.metrics import msssim_loss, scores  # noqa: E402
from ldr.models import BridgeDenoiser  # noqa: E402

# name: (part, loss, schedule, lr, steps, weight decay)
RECIPES = {
    "base": ("all", "l1", "const", 2e-5, 200, 0.0),                # the recipe of the report
    "cos": ("all", "l1ms", "cos", 2e-5, 200, 0.0),
    "cos_short": ("all", "l1ms", "cos", 5e-5, 100, 0.0),
    "dec": ("dec", "l1ms", "cos", 1e-4, 200, 0.0),
    "dec_long": ("dec", "l1ms", "cos", 1e-4, 400, 0.0),
    "enc": ("enc", "l1ms", "cos", 2e-5, 200, 0.0),
    "wd": ("all", "l1ms", "cos", 5e-5, 200, 1e-2),
}

ap = argparse.ArgumentParser()
ap.add_argument("data", choices=["sied", "eld"])
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--recipes", nargs="+", default=list(RECIPES))
ap.add_argument("--ks", type=int, nargs="+", default=None)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--ps", type=int, default=512)
ap.add_argument("--crop", type=int, default=1024)
args = ap.parse_args()

RUNS, OUT = Path("/content/runs"), Path("/content/results")
OUT.mkdir(parents=True, exist_ok=True)
out = OUT / f"adapt_grid_{args.data}.jsonl"
unet0, uck = M.load_ckpt(RUNS / "sid" / "best.pt")
unet0.eval().requires_grad_(False)
kind = uck["model"]
dck = torch.load(RUNS / "ddbm_vp_unet" / "best.pt", map_location=M.DEV)
ddbm = BridgeDenoiser(**dck.get("kwargs", {})).to(M.DEV).eval()
ddbm.load_state_dict(dck["ema"])
ddbm.requires_grad_(False)
bridge = M.make_bridge(dck["args"])


@torch.no_grad()
def readout(start, b):
    xT = M.bridge_start(b, start)
    s = bridge.sigmas(2, xT.device)[:1]
    d, _ = bridge.denoise(ddbm, xT, s.expand(xT.shape[0]), xT, b["short"], b["cond"])
    return (d.float().clamp(-1, 1) + 1) / 2


def evaluate(model, cache, test):
    ds = PairDataset(cache, test, ps=args.crop if args.data == "sied" else None, train=False)
    rows = []
    for i in range(len(test)):
        b = {k: (v[None].to(M.DEV) if torch.is_tensor(v) else v) for k, v in ds[i].items()}
        with torch.no_grad():
            u = M.predict(model, b, kind)
            d = readout(model, b)
        su, sd = scores(u, b["gt"], True), scores(d, b["gt"], True)
        rows.append([su["psnr"], su["lpips"], sd["psnr"], sd["lpips"]])
    return rows


def adapt(cache, support, recipe, seed):
    part, loss_kind, sched, lr, steps, wd = RECIPES[recipe]
    M.seed_all(seed)
    random.seed(seed)
    model = copy.deepcopy(unet0).train()
    for n, p in model.named_parameters():
        p.requires_grad_(part == "all" or n.startswith(part))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    lam = (lambda t: 0.5 * (1 + math.cos(math.pi * t / steps))) if sched == "cos" else (lambda t: 1.0)
    sc = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    ds = PairDataset(cache, support, ps=args.ps)
    for _ in range(steps):
        b = M.batch_of(ds, random.sample(range(len(ds)), min(args.bs, len(ds))))
        with torch.autocast(**M.AMP):
            pred = model(b["short"], b["cond"])["rgb"].float()
        loss = F.l1_loss(pred, b["gt"])
        if loss_kind == "l1ms":
            loss = loss + 0.1 * msssim_loss(pred, b["gt"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sc.step()
    return model.eval().requires_grad_(False)


done = set()
if out.exists():
    for line in out.read_text().splitlines():
        try:
            r = json.loads(line)
            done.add((r["recipe"], r["seed"], r["k"], r.get("camera", "")))
        except json.JSONDecodeError:
            pass

if args.data == "sied":
    cdir = Path("/content/cache/custom/sied")
    cache = Cache(cdir)
    load = lambda f: [R.Pair(**p) for p in json.loads((cdir / f).read_text())]
    pool, test = load("pairs_train.json"), load("pairs_test.json")
    tasks = [("", cache, lambda k: pool[:k], test, k) for k in (args.ks or [10, 30, 100])]
else:
    a = type("A", (), {"cache": "/content/cache"})()
    tasks = []
    for cam in M.ELD_CAMERAS:
        for k in (args.ks or [1, 3, 6]):
            cache, test = M.eld(a, [cam], M.ELD_TEST, M.ELD_EVAL_IDS)
            tasks.append((cam, cache, (lambda k, cam=cam: M.eld(a, [cam], M.ELD_TRAIN[:k])[1]), test, k))

# zero-shot reference, once per dataset and camera
for cam, cache, sup, test, k in tasks[::len(args.ks or [1, 3, 6])] if args.data == "eld" else tasks[:1]:
    if ("zero", 0, 0, cam) not in done:
        rows = evaluate(unet0, cache, test)
        with open(out, "a") as fh:
            fh.write(json.dumps({"data": args.data, "camera": cam, "recipe": "zero", "seed": 0, "k": 0, "rows": rows}) + "\n")
for recipe in args.recipes:
    for seed in args.seeds:
        for cam, cache, sup, test, k in tasks:
            if (recipe, seed, k, cam) in done:
                continue
            model = adapt(cache, sup(k), recipe, seed)
            rows = evaluate(model, cache, test)
            with open(out, "a") as fh:
                fh.write(json.dumps({"data": args.data, "camera": cam, "recipe": recipe, "seed": seed, "k": k, "rows": rows}) + "\n")
            m = [sum(r[i] for r in rows) / len(rows) for i in range(4)]
            print(args.data, cam, recipe, "seed", seed, "K", k, "U %.2f/%.3f DU %.2f/%.3f" % tuple(m), flush=True)
