"""Noise calibration and synthetic pairs for the LRID phone sensor: can a calibrated noise model replace real noisy frames?

1. Calibrate a heteroscedastic Gaussian noise model var(x) = K x + sigma^2 on normalised raw values (0..1 between black and
   white level): sigma^2 per channel from dark frames (bias/6400, temporal variance over a few frames, so that the fixed
   pattern drops out) and the shot-noise gain K from ONE real noisy/clean pair (scene 000, digital gain 1) by a least-squares
   fit of the residual variance against the clean signal in intensity bins.
2. Fine-tune the SID U-Net on the clean raw images of K training scenes with noise drawn fresh at every step from that model
   (same brightness and exposure ratio as the real noisy frames of the scene, same recipe as the real-pair adaptation) and
   score it on the real noisy test frames, with and without the frozen DDBM on its output. The real-pair numbers of
   colab/custom_camera.py adapt are the reference.

Needs the cache of colab/lrid_prep.py (with clean/<scene>.npy) and the dark frames in /content/lrid/bias/6400.

    python colab/lrid_synth.py --ks 1 2 4 12 --steps 200
"""
import argparse
import copy
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np
import rawpy
import torch

sys.path.insert(0, "/content/ldr-src")
sys.path.insert(0, "/content/ldr-src/colab")
import main as M  # noqa: E402
import custom_camera as cc  # noqa: E402
from ldr import raw as R  # noqa: E402
from ldr.data import Cache, condition, noise_level  # noqa: E402
from ldr.models import BridgeDenoiser  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 12])
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--ps", type=int, default=512)
ap.add_argument("--crop", type=int, default=1024)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--noise", choices=["gauss", "dark", "dark_fpn", "dark_fpn_poisson"], default="gauss",
                help="gauss: Gaussian read noise; dark: real dark-frame residuals as read noise (keeps spatial correlation and row noise); "
                     "_fpn: plus the fixed pattern of the dark frames; _poisson: Poisson instead of Gaussian shot noise")
ap.add_argument("--k-scale", type=float, default=1.0, help="multiplies the calibrated K (a sensitivity check)")
ap.add_argument("--out", default="/content/results/lrid_synth.json")
ap.add_argument("--mix", type=int, default=0, help="mix test: K real scenes (--ks) + synthetic noise on this many scenes, half and half")
ap.add_argument("--cross", action="store_true", help="cross test: zero-shot, real-pair and synthetic models on real and on synthetic noisy test frames")
args = ap.parse_args()

CDIR = Path("/content/cache/custom/lrid")
BLACK, WHITE = 64.0, 1023.0
cache = Cache(CDIR)
load = lambda f: [R.Pair(**p) for p in json.loads((CDIR / f).read_text())]
train_pairs, test_pairs = load("pairs_train.json"), load("pairs_test.json")

# ---- 1. calibration
frames = sorted(glob.glob("/content/lrid/bias/6400/*.dng"))[:12]
bias = []
for f in frames:
    with rawpy.imread(f) as r:
        p, _ = R.pack_bayer(r)
    bias.append((p.astype(np.float32) - BLACK) / (WHITE - BLACK))
bias = np.stack(bias)                                   # (n, H, W, 4)
bias_mean = bias.mean(axis=0)                            # fixed pattern (dark shading), (H, W, 4)
bias_res = ((bias - bias_mean[None]) * np.sqrt(len(bias) / (len(bias) - 1))).astype(np.float32)  # temporal residuals: the read noise itself
sigma2 = bias.var(axis=0, ddof=1).mean(axis=(0, 1))     # per channel temporal variance
print("read-noise sigma^2 per channel (normalised units):", sigma2, "sd:", np.sqrt(sigma2), flush=True)
p0 = next(p for p in train_pairs if p.scene.endswith("/000") and p.short.endswith("_g1_s"))
noisy = cache.norm_raw(cache.load("short", p0.short), p0.short)
clean = np.load(CDIR / "clean" / "000.npy").astype(np.float32)
f_scale = float(noisy.mean() / max(clean.mean(), 1e-9))
res = noisy - f_scale * clean
x = f_scale * clean
edges = np.quantile(x, np.linspace(0, 1, 21))
bx, bv = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (x >= lo) & (x <= hi)
    if m.sum() > 1000:
        bx.append(x[m].mean())
        bv.append(res[m].var())
bx, bv = np.array(bx), np.array(bv)
s2 = float(sigma2.mean())
Kfit = float(np.sum(bx * (bv - s2)) / np.sum(bx * bx))   # least squares through var - sigma^2 = K x
print("shot-noise gain K = %.5f (scene 000, gain 1); residual variance range %.2e..%.2e" % (Kfit, bv.min(), bv.max()), flush=True)
K = Kfit * args.k_scale
CAL = {"sigma2_per_channel": [float(v) for v in sigma2], "K": Kfit, "K_used": K, "f_scale_scene000": f_scale}

# ---- 2. synthetic fine-tuning
unet0, ck = M.load_ckpt("/content/runs/sid/best.pt")
kind = ck["model"]
dck = torch.load("/content/runs/ddbm_vp_unet/best.pt", map_location=M.DEV)
net = BridgeDenoiser(**dck.get("kwargs", {})).to(M.DEV).eval()
net.load_state_dict(dck["ema"])
ddbm = (net, M.make_bridge(dck["args"]))
scene_ids = sorted({p.scene for p in train_pairs})
by_scene = {s: [p for p in train_pairs if p.scene == s] for s in scene_ids}
clean_cache = {}


def clean_of(sc):
    if sc not in clean_cache:
        clean_cache[sc] = np.load(CDIR / "clean" / f"{sc.split('/')[-1]}.npy").astype(np.float32)
    return clean_cache[sc]


def synth_item(p, ps, loc=None):
    """A noisy crop made from the clean raw of the pair's scene, at the brightness of the real noisy frame of that pair."""
    cl = clean_of(p.scene)
    h, w = cl.shape[:2]
    y, x_ = loc if loc else (random.randint(0, h - ps), random.randint(0, w - ps))
    crop = cl[y:y + ps, x_:x_ + ps]
    real = cache.norm_raw(cache.load("short", p.short)[y:y + ps, x_:x_ + ps], p.short)
    f = float(real.mean() / max(crop.mean(), 1e-9))     # brightness of the real frame relative to the clean one
    sig = crop * f
    if args.noise == "gauss":
        std = np.sqrt(np.maximum(K * sig, 0) + sigma2[None, None, :])
        noisy_ = sig + np.random.randn(*sig.shape).astype(np.float32) * std
    else:
        if args.noise.endswith("poisson"):  # signal in photoelectron-like units 1 / K
            shot = np.random.poisson(np.maximum(sig, 0) / K).astype(np.float32) * K
        else:
            shot = sig + np.random.randn(*sig.shape).astype(np.float32) * np.sqrt(np.maximum(K * sig, 0))
        bh, bw = bias_res.shape[1:3]
        by, bx_ = (y, x_) if loc else (random.randint(0, bh - ps), random.randint(0, bw - ps))
        read = bias_res[random.randrange(len(bias_res))][by:by + ps, bx_:bx_ + ps]
        noisy_ = shot + read
        if "fpn" in args.noise:
            noisy_ = noisy_ + bias_mean[by:by + ps, bx_:bx_ + ps]
    noisy_ = np.clip(noisy_, 0, 1).astype(np.float32)
    s = np.minimum(noisy_ * p.ratio, 1.0)
    gt = cache.load("gt", p.long)[2 * y:2 * y + 2 * ps, 2 * x_:2 * x_ + 2 * ps].astype(np.float32) / 65535.0
    f_ = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)
    return {"short": f_(s), "gt": f_(gt), "cond": torch.from_numpy(condition(p.ratio, noise_level(noisy_)))}


def finetune(k_scenes, seed):
    M.seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    pool = [p for s in scene_ids[:k_scenes] for p in by_scene[s]]
    model = copy.deepcopy(unet0).requires_grad_(True).train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for _ in range(args.steps):
        items = [synth_item(random.choice(pool), args.ps) for _ in range(args.bs)]
        b = M.to({k: torch.stack([it[k] for it in items]) for k in items[0]})
        loss = M.task_loss(model, dict(model.named_parameters()), b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model.eval().requires_grad_(False)



def eval_synth(model, ps=1024):
    """PSNR of a model on test frames made from the clean test raw with the calibrated noise (fixed seed, centre crops)."""
    random.seed(1234)
    np.random.seed(1234)
    vals = []
    for p in test_pairs:
        h, w = clean_of(p.scene).shape[:2]
        it = synth_item(p, ps, loc=((h - ps) // 2, (w - ps) // 2))
        b = M.to({k: v[None] for k, v in it.items()})
        with torch.no_grad():
            pred = M.predict(model, b, kind)
        vals.append(cc.scores(pred, b["gt"], False)["psnr"])
    return float(np.mean(vals))


def finetune_real(k_scenes, seed):
    from ldr.data import PairDataset
    M.seed_all(seed)
    random.seed(seed)
    pool = [p for s in scene_ids[:k_scenes] for p in by_scene[s]]
    ds = PairDataset(cache, pool, ps=args.ps)
    model = copy.deepcopy(unet0).requires_grad_(True).train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for _ in range(args.steps):
        b = M.batch_of(ds, random.sample(range(len(ds)), min(args.bs, len(ds))))
        loss = M.task_loss(model, dict(model.named_parameters()), b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model.eval().requires_grad_(False)



def finetune_mix(k_real, k_synth, seed):
    """Half of every batch from the real pairs of k_real scenes, half synthetic from the clean raw of k_synth scenes."""
    from ldr.data import PairDataset
    M.seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    real_pool = [p for s in scene_ids[:k_real] for p in by_scene[s]]
    synth_pool = [p for s in scene_ids[:k_synth] for p in by_scene[s]]
    ds = PairDataset(cache, real_pool, ps=args.ps)
    model = copy.deepcopy(unet0).requires_grad_(True).train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    h = args.bs // 2
    for _ in range(args.steps):
        br = M.batch_of(ds, random.sample(range(len(ds)), min(h, len(ds))))
        items = [synth_item(random.choice(synth_pool), args.ps) for _ in range(args.bs - h)]
        bs_ = M.to({k: torch.stack([it[k] for it in items]) for k in items[0]})
        b = {k: torch.cat([br[k], bs_[k]]) for k in ("short", "gt", "cond")}
        loss = M.task_loss(model, dict(model.named_parameters()), b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model.eval().requires_grad_(False)


if args.mix:
    res = {}
    for seed in args.seeds:
        for name, fn in (("real", lambda: finetune_real(args.ks[0], seed)), ("mix", lambda: finetune_mix(args.ks[0], args.mix, seed))):
            m = fn()
            sm = cc.summarize(cc.score_all(m, kind, cache, test_pairs, args.crop, ddbm))
            res[f"{name}_seed{seed}"] = sm
            print("mix", name, "seed", seed, json.dumps({n: (round(d["psnr"], 2), round(d["lpips"], 3)) for n, d in sm.items()}), flush=True)
            Path(args.out).write_text(json.dumps(res))
    raise SystemExit

if args.cross:
    res = {"zero": {"real": cc.summarize(cc.score_all(unet0.eval(), kind, cache, test_pairs, args.crop, None))["unet"]["psnr"],
                    "synth": eval_synth(unet0.eval())}}
    print("cross zero", res["zero"], flush=True)
    for seed in args.seeds:
        for name, fn in (("real", finetune_real), ("synth", finetune)):
            m = fn(args.ks[0], seed)
            res[f"{name}_seed{seed}"] = {"real": cc.summarize(cc.score_all(m, kind, cache, test_pairs, args.crop, None))["unet"]["psnr"],
                                         "synth": eval_synth(m)}
            print("cross", name, "seed", seed, res[f"{name}_seed{seed}"], flush=True)
            Path(args.out).write_text(json.dumps(res))
    raise SystemExit

out = {"calibration": CAL, "steps": args.steps, "k_scale": args.k_scale, "runs": {}}
for k in args.ks:
    for seed in args.seeds:
        model = finetune(k, seed)
        rows = cc.score_all(model, kind, cache, test_pairs, args.crop, ddbm)
        out["runs"][f"scenes{k}_seed{seed}"] = {"summary": cc.summarize(rows)}
        s = out["runs"][f"scenes{k}_seed{seed}"]["summary"]
        print("scenes", k, "seed", seed, json.dumps({n: (round(d["psnr"], 2), round(d["lpips"], 3)) for n, d in s.items()}), flush=True)
        Path(args.out).write_text(json.dumps(out))
