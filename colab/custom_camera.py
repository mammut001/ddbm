"""Use the SID-trained U-Net on a camera of your own (any Bayer RAW that rawpy reads, e.g. a Sony A7CR).

Folder layout, one folder per static scene (tripod):

    mycam/scene-01/ref.ARW        a long exposure at low ISO (the clean reference; or the file with the largest
                                  exposure x ISO is taken as the reference)
    mycam/scene-01/short_01.ARW   any number of short or high-ISO frames of the same scene
    mycam/scene-02/...

    python colab/custom_camera.py prep    --root mycam --name a7cr
    python colab/custom_camera.py adapt   --name a7cr --init runs/sid/best.pt --k 3 [--ddbm-ckpt runs/ddbm_vp_unet/best.pt]
    python colab/custom_camera.py enhance --ckpt runs/custom_a7cr/adapted.pt --inputs a.ARW b.ARW --ratio 100 --out out

prep caches the RAWs and the pairs (exposure ratio = reference exposure x ISO over short exposure x ISO, as in ELD).
adapt fine-tunes the U-Net on the first K scenes (the recipe of `main.py fewshot`: 200 Adam steps) and scores the
zero-shot and adapted model on the remaining scenes (centre crop, PSNR / SSIM / MS-SSIM / LPIPS); with --ddbm-ckpt it also scores the unadapted DDBM
fed with the adapted U-Net's output, which kept the U-Net's PSNR and lowered LPIPS in the report. enhance writes
8-bit PNGs of whole images (tiled, so a 60 MP file fits).
"""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/content/ldr-src")
import main as M  # noqa: E402
from ldr import raw as R  # noqa: E402
from ldr.data import Cache, PairDataset, condition, noise_level  # noqa: E402
from ldr.metrics import scores  # noqa: E402
from ldr.models import BridgeDenoiser  # noqa: E402

EXT = {".arw", ".dng", ".nef", ".cr2", ".cr3", ".orf", ".rw2", ".pef", ".srw"}
CACHE = Path("/content/cache/custom")


def gain(m):
    return m["exposure"] * m.get("iso", 1.0)


def cmd_prep(a):
    root, name = Path(a.root), a.name
    out = CACHE / name
    jobs, scenes = [], {}
    for sdir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(f for f in sdir.iterdir() if f.suffix.lower() in EXT)
        ref = [f for f in files if f.stem.lower().startswith("ref")]
        if not ref:  # no file called ref*: the largest exposure x ISO is the reference
            meta = {f: R.exif(f) for f in files}
            ref = [max(files, key=lambda f: gain({"exposure": meta[f].get("exposure", 0), **meta[f]}))] if files else []
        if not ref or len(files) < 2:
            print(f"skip {sdir.name}: needs a reference and at least one short frame")
            continue
        scenes[sdir.name] = (ref[0], [f for f in files if f != ref[0]])
        for f in files:
            jobs.append((f, f"{name}_{sdir.name}_{f.stem}", f == ref[0]))
    meta = R.build_cache(jobs, out, a.workers)
    pairs = []
    for sc, (ref, shorts) in scenes.items():
        lstem = f"{name}_{sc}_{ref.stem}"
        for f in shorts:
            sstem = f"{name}_{sc}_{f.stem}"
            if "exposure" not in meta[lstem] or "exposure" not in meta[sstem] or meta[sstem]["exposure"] != meta[sstem]["exposure"]:
                print(f"skip {f}: no exposure time in the EXIF")
                continue
            ratio = gain(meta[lstem]) / gain(meta[sstem])
            if ratio < 1.5:
                print(f"skip {f}: ratio {ratio:.2f} to the reference is too small to be a short exposure")
                continue
            pairs.append({"short": sstem, "long": lstem, "scene": f"{name}/{sc}", "ratio": min(ratio, 300.0), "camera": name})
    (out / "pairs.json").write_text(json.dumps(pairs))
    print(f"{name}: {len(scenes)} scenes, {len(pairs)} pairs, ratios {sorted({round(p['ratio']) for p in pairs})}")


def load_pairs(name):
    cache = Cache(CACHE / name)
    pairs = [R.Pair(**p) for p in json.loads((CACHE / name / "pairs.json").read_text())]
    return cache, pairs


@torch.no_grad()
def readout(net, bridge, start, b):
    xT = M.bridge_start(b, start)
    s = bridge.sigmas(2, xT.device)[:1]
    d, _ = bridge.denoise(net, xT, s.expand(xT.shape[0]), xT, b["short"], b["cond"])
    return (d.float().clamp(-1, 1) + 1) / 2


def score_all(unet, kind, cache, pairs, crop, ddbm):
    rows = []
    ds = PairDataset(cache, pairs, ps=crop, train=False)
    for i, p in enumerate(pairs):
        b = {k: (v[None].to(M.DEV) if torch.is_tensor(v) else v) for k, v in ds[i].items()}
        with torch.no_grad():
            est = {"unet": M.predict(unet, b, kind)}
            if ddbm:
                est["unet+ddbm"] = readout(ddbm[0], ddbm[1], unet, b)
        rows.append({"short": p.short, "scene": p.scene, "ratio": p.ratio, "scores": {n: scores(x, b["gt"], True) for n, x in est.items()}})
    return rows


def summarize(rows):
    names = rows[0]["scores"].keys()
    return {n: {m: sum(r["scores"][n][m] for r in rows) / len(rows) for m in ("psnr", "ssim", "msssim", "lpips")} for n in names}


def cmd_adapt_ks(a):
    """--train-pairs / --test-pairs: fixed lists (pairs_*.json in the cache); one adaptation per K in --ks."""
    cache = Cache(CACHE / a.name)
    load = lambda f: [R.Pair(**p) for p in json.loads((CACHE / a.name / f).read_text())]
    train_all, test = load(a.train_pairs), load(a.test_pairs)
    unet0, ck = M.load_ckpt(a.init)
    kind = ck["model"]
    ddbm = None
    if a.ddbm_ckpt:
        dck = torch.load(a.ddbm_ckpt, map_location=M.DEV)
        net = BridgeDenoiser(**dck.get("kwargs", {})).to(M.DEV).eval()
        net.load_state_dict(dck["ema"])
        ddbm = (net, M.make_bridge(dck["args"]))
    out = Path(a.out or f"/content/runs/custom_{a.name}")
    out.mkdir(parents=True, exist_ok=True)
    res = {"steps": a.steps, "lr": a.lr, "n_test": len(test), "ks": {}}
    for k in a.ks:
        M.seed_all(a.seed)
        model = copy.deepcopy(unet0).requires_grad_(True).train()
        if k:
            ds = PairDataset(cache, train_all[:k], ps=a.ps)
            opt = torch.optim.Adam(model.parameters(), lr=a.lr)
            for step in range(a.steps):
                b = M.batch_of(ds, random.sample(range(len(ds)), min(a.bs, len(ds))))
                loss = M.task_loss(model, dict(model.named_parameters()), b)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        model.eval().requires_grad_(False)
        rows = score_all(model, kind, cache, test, a.crop, ddbm)
        res["ks"][str(k)] = {"summary": summarize(rows), "rows": rows}
        print("K", k, json.dumps(summarize(rows)), flush=True)
        (out / "result_ks.json").write_text(json.dumps(res))
        if k == a.ks[-1]:
            sd = model.state_dict()
            torch.save({"state": sd, "ema": sd, "model": kind, "kwargs": ck.get("kwargs", {}), "step": a.steps, "args": {}}, out / "adapted.pt")


def cmd_adapt(a):
    if a.train_pairs:
        return cmd_adapt_ks(a)
    cache, pairs = load_pairs(a.name)
    scenes = sorted({p.scene for p in pairs})
    sup = a.support or scenes[:a.k]
    sup = [s if "/" in s else f"{a.name}/{s}" for s in sup]
    train = [p for p in pairs if p.scene in sup]
    test = [p for p in pairs if p.scene not in sup]
    print(f"support scenes {sup} ({len(train)} pairs), test scenes {sorted({p.scene for p in test})} ({len(test)} pairs)")
    M.seed_all(a.seed)
    unet0, ck = M.load_ckpt(a.init)
    kind = ck["model"]
    ddbm = None
    if a.ddbm_ckpt:
        dck = torch.load(a.ddbm_ckpt, map_location=M.DEV)
        net = BridgeDenoiser(**dck.get("kwargs", {})).to(M.DEV).eval()
        net.load_state_dict(dck["ema"])
        ddbm = (net, M.make_bridge(dck["args"]))
    res = {"support": sup, "steps": a.steps, "lr": a.lr}
    if test:
        res["zero_shot"] = summarize(score_all(unet0.eval(), kind, cache, test, a.crop, ddbm))
    model = copy.deepcopy(unet0).requires_grad_(True).train()
    ds = PairDataset(cache, train, ps=a.ps)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    for step in range(a.steps):
        b = M.batch_of(ds, random.sample(range(len(ds)), min(a.bs, len(ds))))
        loss = M.task_loss(model, dict(model.named_parameters()), b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 20 == 0:
            print(f"step {step} L1 {loss.item():.4f}", flush=True)
    model.eval().requires_grad_(False)
    if test:
        res["adapted"] = summarize(score_all(model, kind, cache, test, a.crop, ddbm))
    out = Path(a.out or f"/content/runs/custom_{a.name}")
    out.mkdir(parents=True, exist_ok=True)
    sd = model.state_dict()
    torch.save({"state": sd, "ema": sd, "model": kind, "kwargs": ck.get("kwargs", {}), "step": a.steps, "args": {}}, out / "adapted.pt")
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


@torch.no_grad()
def tiles(fn, x, tile=1024, pad=64):
    """Apply fn to a (1, C, H, W) packed tensor in overlapping tiles; fn maps packed tiles to RGB at twice the size."""
    _, _, h, w = x.shape
    out = None
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            ya, yb = max(y0 - pad, 0), min(y0 + tile + pad, h)
            xa, xb = max(x0 - pad, 0), min(x0 + tile + pad, w)
            y = fn(x[:, :, ya:yb, xa:xb])
            if out is None:
                out = torch.zeros(1, y.shape[1], 2 * h, 2 * w)
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            out[:, :, 2 * y0:2 * y1, 2 * x0:2 * x1] = y[:, :, 2 * (y0 - ya):2 * (y1 - ya), 2 * (x0 - xa):2 * (x1 - xa)].float().cpu()
    return out


def cmd_enhance(a):
    import rawpy
    from PIL import Image
    unet, ck = M.load_ckpt(a.ckpt)
    unet.eval()
    ddbm = None
    if a.ddbm_ckpt:
        dck = torch.load(a.ddbm_ckpt, map_location=M.DEV)
        net = BridgeDenoiser(**dck.get("kwargs", {})).to(M.DEV).eval()
        net.load_state_dict(dck["ema"])
        ddbm = (net, M.make_bridge(dck["args"]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in a.inputs:
        with rawpy.imread(str(f)) as raw:
            packed, lv = R.pack_bayer(raw)
        s = np.maximum(packed.astype(np.float32) - lv["black"], 0) / (lv["white"] - lv["black"])
        sigma = noise_level(s)
        s = np.minimum(s * a.ratio, 1.0)
        cond = torch.from_numpy(condition(a.ratio, sigma))[None].to(M.DEV)
        x = torch.from_numpy(np.ascontiguousarray(s)).permute(2, 0, 1)[None]

        def run_unet(t):
            with torch.autocast(**M.AMP):
                return unet(t.to(M.DEV), cond)["rgb"].float().clamp(0, 1)

        def run_ddbm(t):
            t = t.to(M.DEV)
            return readout(ddbm[0], ddbm[1], unet, {"short": t, "cond": cond})

        for tag, fn in [("unet", run_unet)] + ([("unet_ddbm", run_ddbm)] if ddbm else []):
            y = tiles(fn, x, a.tile)[0].permute(1, 2, 0).clamp(0, 1).numpy()
            Image.fromarray((y * 255 + 0.5).astype(np.uint8)).save(out / f"{Path(f).stem}_{tag}.png")
        print("wrote", f, flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--root", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(func=cmd_prep)
    p = sub.add_parser("adapt")
    p.add_argument("--name", required=True)
    p.add_argument("--init", default="/content/runs/sid/best.pt")
    p.add_argument("--k", type=int, default=3, help="number of support scenes (the first K by name)")
    p.add_argument("--support", nargs="*", help="support scenes by name; overrides --k")
    p.add_argument("--train-pairs", help="pairs_*.json of the support pool (first K are used), with --test-pairs and --ks")
    p.add_argument("--test-pairs")
    p.add_argument("--ks", type=int, nargs="*", default=[0, 10, 30, 100])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--ps", type=int, default=512)
    p.add_argument("--crop", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ddbm-ckpt")
    p.add_argument("--out")
    p.set_defaults(func=cmd_adapt)
    p = sub.add_parser("enhance")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--ratio", type=float, required=True, help="amplification, e.g. 100 for a frame 100x darker than the reference")
    p.add_argument("--out", default="/content/out")
    p.add_argument("--tile", type=int, default=1024)
    p.add_argument("--ddbm-ckpt")
    p.set_defaults(func=cmd_enhance)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
