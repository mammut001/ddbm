#!/usr/bin/env python3
"""Learning to See in the Dark with bridge refinement: every experiment of the report.

  prep-sid    cache SID Sony RAWs as uint16 packs (+ sRGB targets)
  prep-eld    cache ELD cameras the same way
  train       supervised training on SID: --model sid | bridge | ddbm_feature
  train-ddbm  pixel-space DDBM (true bridge score matching) on SID
  finetune    adapt a SID checkpoint to ELD cameras (all trainable layers, or --freeze-encoder)
  meta        MAML over scene-disjoint SID tasks with simulated sensor responses
  fewshot     K-shot adaptation on ELD cameras from a given initialisation
  eval        full-image PSNR / SSIM / MS-SSIM on SID test or ELD held-out scenes

Every command writes JSON lines to <run>/log.jsonl so the report figures are reproducible.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import DataLoader

from ldr import raw as R
from ldr.data import Cache, PairDataset, by_scene
from ldr.ddbm import Bridge, naive_render
from ldr.metrics import msssim_loss, scores
from ldr.models import MODELS, BridgeDenoiser

DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = dict(device_type="cuda", dtype=torch.bfloat16, enabled=DEV == "cuda")
ELD_CAMERAS = ["CanonEOS70D", "CanonEOS700D", "NikonD850"]
ELD_TRAIN, ELD_TEST = list(range(1, 7)), list(range(7, 11))
ELD_EVAL_IDS = [4, 9, 14, 5, 10, 15]  # ratios 100 and 200, the ELD paper's protocol


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


class Log:
    def __init__(self, run: Path):
        run.mkdir(parents=True, exist_ok=True)
        self.f = (run / "log.jsonl").open("a")

    def __call__(self, **kw):
        kw["time"] = time.time()
        self.f.write(json.dumps(kw) + "\n")
        self.f.flush()
        print(json.dumps(kw), flush=True)


class EMA:
    def __init__(self, model, decay=0.999):
        self.m = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.m.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.lerp_(p, 1 - self.decay)


def to(b):
    return {k: (v.to(DEV, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def build(name, **kw):
    return MODELS[name](**kw).to(DEV)


def load_ckpt(path, model_name=None):
    ck = torch.load(path, map_location=DEV)
    m = build(model_name or ck["model"], **ck.get("kwargs", {}))
    m.load_state_dict(ck["ema"] if "ema" in ck else ck["state"])
    return m, ck


def plain(args):
    return {k: v for k, v in vars(args).items() if k != "func"}


def save(path, model, ema, args, step, name, kwargs=None):
    torch.save({"state": model.state_dict(), "ema": ema.m.state_dict(), "model": name,
                "kwargs": kwargs or {}, "step": step, "args": plain(args)}, path)


# ----------------------------------------------------------------------------- data


def sid_split(args, split):
    cache = Cache(Path(args.cache) / "sid")
    pairs = json.loads((Path(args.cache) / "sid" / "pairs.json").read_text())[split]
    return cache, [R.Pair(**p) for p in pairs]


def eld(args, cameras, scenes, img_ids=None):
    cache = Cache(Path(args.cache) / "eld")
    pairs = []
    for c in cameras:
        pairs += R.eld_pairs(Path(args.cache) / "eld", c, scenes, img_ids)
    return cache, pairs


def loader(ds, bs, workers, shuffle=True):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, pin_memory=True,
                      drop_last=shuffle, persistent_workers=workers > 0)


def forever(dl):
    while True:
        yield from dl


# ----------------------------------------------------------------------------- prep


def cmd_prep_sid(args):
    root = Path(args.sid_raw)
    pairs = R.sid_pairs(root)
    jobs = []
    seen = set()
    for split in pairs.values():
        for p in split:
            jobs.append((root / "short" / f"{p.short}.ARW", p.short, False))
            if p.long not in seen:
                seen.add(p.long)
                jobs.append((root / "long" / f"{p.long}.ARW", p.long, True))
    out = Path(args.cache) / "sid"
    R.build_cache(jobs, out, args.workers)
    (out / "pairs.json").write_text(json.dumps({k: [vars(p) for p in v] for k, v in pairs.items()}))
    print({k: (len(v), len(by_scene(v))) for k, v in pairs.items()})


def cmd_prep_eld(args):
    root = Path(args.eld_raw)
    jobs = []
    for cam in args.cameras:
        for sdir in sorted((root / cam).glob("scene-*")):
            for f in sorted(sdir.iterdir()):
                if f.suffix.lower() not in (".cr2", ".nef", ".arw", ".dng"):
                    continue
                idx = int(f.stem.split("_")[-1])
                stem = f"{cam}_{sdir.name}_IMG_{idx:04d}"
                jobs.append((f, stem, idx in R.ELD_GT_IDS))
    R.build_cache(jobs, Path(args.cache) / "eld", args.workers)


# ----------------------------------------------------------------------------- evaluation


@torch.no_grad()
def predict(model, b, kind, steps=3, ddbm_steps=10, eta=1.0, bridge=None):
    with torch.autocast(**AMP):
        if kind == "ddbm":
            xT = naive_render(b["short"], b["wb"])
            y = bridge.sample(model, xT, b["short"], b["cond"], steps=ddbm_steps, eta=eta)
            return ((y.float() + 1) / 2).clamp(0, 1)
        if kind == "ddbm_feature":
            return model(b["short"], b["cond"], steps=steps)["rgb"].float().clamp(0, 1)
        return model(b["short"], b["cond"])["rgb"].float().clamp(0, 1)


@torch.no_grad()
def evaluate(model, cache, pairs, kind, crop=None, limit=None, **kw):
    model.eval()
    ds = PairDataset(cache, pairs[:limit] if limit else pairs, ps=crop, train=False)
    rows = []
    for i in range(len(ds)):
        b = {k: (v[None].to(DEV) if torch.is_tensor(v) else v) for k, v in ds[i].items()}
        pred = predict(model, b, kind, **kw)
        s = scores(pred, b["gt"])
        p = ds.pairs[i]
        rows.append({"short": p.short, "scene": p.scene, "ratio": p.ratio, "camera": p.camera, **s})
    model.train()
    agg = {k: float(np.mean([r[k] for r in rows])) for k in ("psnr", "ssim", "msssim", "l1")}
    return agg, rows


def cmd_eval(args):
    if args.kind == "ddbm":
        ck = torch.load(args.ckpt, map_location=DEV)
        model = BridgeDenoiser(**ck.get("kwargs", {})).to(DEV)
        model.load_state_dict(ck["ema"])
        bridge = Bridge(s2=ck["args"]["s2"])
    else:
        model, ck = load_ckpt(args.ckpt)
        args.kind = ck["model"]
        bridge = None
    if args.data == "sid":
        cache, pairs = sid_split(args, "test")
    else:
        cache, pairs = eld(args, [args.data], ELD_TEST, ELD_EVAL_IDS)
    t0 = time.time()
    agg, rows = evaluate(model, cache, pairs, args.kind, crop=args.crop, limit=args.limit,
                         steps=args.steps, ddbm_steps=args.ddbm_steps, eta=args.eta, bridge=bridge)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ckpt": args.ckpt, "data": args.data, "n": len(rows), **agg,
                               "seconds": time.time() - t0, "rows": rows}))
    print(json.dumps({"n": len(rows), **agg}))


# ----------------------------------------------------------------------------- supervised


def objective(model, b, args, step, kind):
    with torch.autocast(**AMP):
        out = model(b["short"], b["cond"], long=b["long"] if kind != "sid" else None)
    pred = out["rgb"].float()
    l1 = F.l1_loss(pred, b["gt"])
    loss = l1
    parts = {"l1": l1.item()}
    if kind == "ddbm_feature" or step >= args.ms_start:
        ms = msssim_loss(pred, b["gt"])
        loss = loss + args.ms_weight * ms
        parts["msssim_loss"] = ms.item()
    if "distill" in out:
        loss = loss + args.distill * out["distill"].float()
        parts["distill"] = out["distill"].item()
    return loss, parts


def run_supervised(args, model, kind, train_set, val, kwargs):
    run = Path(args.run)
    log = Log(run)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 500) * (0.5 * (1 + math.cos(math.pi * min(s, args.iters) / args.iters)) * 0.98 + 0.02))
    ema = EMA(model, args.ema)
    start, best = 0, -1.0
    if (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=DEV)
        model.load_state_dict(ck["state"])
        ema.m.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best = ck["step"], ck.get("best", -1.0)
        print(f"resumed at {start}")
    it = forever(loader(train_set, args.bs, args.workers))
    t0 = time.time()
    model.train()
    for step in range(start, args.iters):
        b = to(next(it))
        loss, parts = objective(model, b, args, step, kind)
        if not torch.isfinite(loss) or (step > 1000 and loss.item() > 5 * args._run_loss):
            log(step=step, skipped=loss.item())
            opt.zero_grad(set_to_none=True)
            continue
        args._run_loss = 0.99 * args._run_loss + 0.01 * loss.item() if step else loss.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        ema.update(model)
        if step % 100 == 0:
            log(step=step, loss=loss.item(), lr=sched.get_last_lr()[0],
                it_s=(step - start + 1) / (time.time() - t0), **parts)
        if (step + 1) % args.val_every == 0 or step + 1 == args.iters:
            cache, vpairs = val
            agg, _ = evaluate(ema.m, cache, vpairs, kind, crop=args.val_crop)
            log(step=step + 1, val=agg)
            if agg["psnr"] > best:
                best = agg["psnr"]
                save(run / "best.pt", model, ema, args, step + 1, kind, kwargs)
            ck = {"state": model.state_dict(), "ema": ema.m.state_dict(), "opt": opt.state_dict(),
                  "sched": sched.state_dict(), "step": step + 1, "best": best, "model": kind,
                  "kwargs": kwargs, "args": plain(args)}
            torch.save(ck, run / "last.pt")
    save(run / "final.pt", model, ema, args, args.iters, kind, kwargs)


def model_kwargs(args):
    if args.model == "bridge":
        return {"n_bridge": args.n_bridge, "film": not args.no_film}
    if args.model == "ddbm_feature":
        return {"sigma_max": args.sigma_max}
    return {}


def cmd_train(args):
    seed_all(args.seed)
    cache, train = sid_split(args, "train")
    _, valp = sid_split(args, "val")
    kw = model_kwargs(args)
    if args.model == "bridge" and args.no_distill:
        args.distill = 0.0
    model = build(args.model, **kw)
    print(f"{args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")
    ds = PairDataset(cache, train, ps=args.ps, repeat=50)
    run_supervised(args, model, args.model, ds, (cache, valp[:args.val_n]), kw)


def cmd_finetune(args):
    seed_all(args.seed)
    model, ck = load_ckpt(args.init)
    kind = ck["model"]
    if args.freeze_encoder:
        for p in model.enc.parameters():
            p.requires_grad_(False)
    cache, train = eld(args, ELD_CAMERAS, ELD_TRAIN)
    _, valp = eld(args, ELD_CAMERAS, ELD_TEST, ELD_EVAL_IDS)
    ds = PairDataset(cache, train, ps=args.ps, repeat=100)
    run_supervised(args, model, kind, ds, (cache, valp), ck.get("kwargs", {}))


# ----------------------------------------------------------------------------- DDBM


def cmd_train_ddbm(args):
    seed_all(args.seed)
    run = Path(args.run)
    log = Log(run)
    cache, train = sid_split(args, "train")
    _, valp = sid_split(args, "val")
    kw = {}
    net = BridgeDenoiser().to(DEV)
    print(f"ddbm: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params")
    bridge = Bridge(s2=args.s2)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0)
    ema = EMA(net, args.ema)
    start, best = 0, -1.0
    if (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=DEV)
        net.load_state_dict(ck["state"])
        ema.m.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        start, best = ck["step"], ck.get("best", -1.0)
    it = forever(loader(PairDataset(cache, train, ps=args.ps, repeat=50), args.bs, args.workers))
    t0 = time.time()
    for step in range(start, args.iters):
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1, (step + 1) / 1000)
        b = to(next(it))
        x0 = b["gt"] * 2 - 1
        xT = naive_render(b["short"], b["wb"])
        with torch.autocast(**AMP):
            loss, _ = bridge.loss(net, x0, xT, b["short"], b["cond"])
        opt.zero_grad(set_to_none=True)
        loss.float().backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        ema.update(net)
        if step % 100 == 0:
            log(step=step, loss=loss.item(), it_s=(step - start + 1) / (time.time() - t0))
        if (step + 1) % args.val_every == 0 or step + 1 == args.iters:
            agg, _ = evaluate(ema.m, cache, valp[:args.val_n], "ddbm", crop=args.val_crop,
                              ddbm_steps=args.ddbm_steps, bridge=bridge)
            log(step=step + 1, val=agg)
            state = {"state": net.state_dict(), "ema": ema.m.state_dict(), "opt": opt.state_dict(),
                     "step": step + 1, "best": max(best, agg["psnr"]), "kwargs": kw,
                     "args": plain(args)}
            if agg["psnr"] > best:
                best = agg["psnr"]
                torch.save(state, run / "best.pt")
            torch.save(state, run / "last.pt")


# ----------------------------------------------------------------------------- meta-learning


def batch_of(ds, idx):
    items = [ds[i] for i in idx]
    return to({k: (torch.stack([x[k] for x in items]) if torch.is_tensor(items[0][k]) else [x[k] for x in items])
               for k in items[0]})


def task_loss(model, params, b):
    with torch.autocast(**AMP):
        out = functional_call(model, params, (b["short"], b["cond"]))
    return F.l1_loss(out["rgb"].float(), b["gt"])


def adapt(model, params, b, steps, lr, create_graph):
    for _ in range(steps):
        loss = task_loss(model, params, b)
        grads = torch.autograd.grad(loss, list(params.values()), create_graph=create_graph, allow_unused=True)
        params = {k: (v - lr * g if g is not None else v) for (k, v), g in zip(params.items(), grads)}
    return params


def sample_task(cache, scenes, args):
    """One task = one simulated sensor (per-channel colour gains, G1 = G2) and one exposure
    ratio; support and query are drawn from *different* scenes, so the query target is
    never seen during adaptation."""
    g = np.exp(np.random.uniform(-args.gain_range, args.gain_range, 3)).astype(np.float32)
    gains = np.array([g[0], g[1], g[2], g[1]], dtype=np.float32)
    ratio = random.choice([100.0, 250.0, 300.0])
    pool = [s for s, ps in scenes.items() if any(abs(p.ratio - ratio) < 1 for p in ps)]
    chosen = random.sample(pool, args.k_shot + args.q_scenes)
    pick = lambda s: random.choice([p for p in scenes[s] if abs(p.ratio - ratio) < 1])
    sup = [pick(s) for s in chosen[:args.k_shot]]
    qry = [pick(s) for s in chosen[args.k_shot:]]
    mk = lambda ps: PairDataset(cache, ps, ps=args.ps, gains=gains)
    return mk(sup), mk(qry)


def cmd_meta(args):
    seed_all(args.seed)
    run = Path(args.run)
    log = Log(run)
    model, ck = load_ckpt(args.init)
    kind, kw = ck["model"], ck.get("kwargs", {})
    cache, train = sid_split(args, "train")
    scenes = by_scene(train)
    log(event="tasks", train_scenes=len(scenes), train_pairs=len(train),
        shorts_per_scene=[min(len(v) for v in scenes.values()), max(len(v) for v in scenes.values())],
        k_shot=args.k_shot, q_scenes=args.q_scenes, inner_steps=args.inner_steps, meta_batch=args.meta_batch)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = EMA(model, 0.99)
    model.train()
    t0 = time.time()
    for it in range(args.iters):
        opt.zero_grad(set_to_none=True)
        qlosses, slosses = [], []
        for _ in range(args.meta_batch):
            sup, qry = sample_task(cache, scenes, args)
            bs, bq = batch_of(sup, range(len(sup))), batch_of(qry, range(len(qry)))
            params = dict(model.named_parameters())
            slosses.append(task_loss(model, params, bs).item())
            fast = adapt(model, params, bs, args.inner_steps, args.inner_lr, not args.first_order)
            if args.first_order:
                fast = {k: v.detach().requires_grad_(True) for k, v in fast.items()}
                ql = task_loss(model, fast, bq)
                grads = torch.autograd.grad(ql, list(fast.values()), allow_unused=True)
                for p, g in zip(model.parameters(), grads):
                    if g is not None:
                        p.grad = g / args.meta_batch if p.grad is None else p.grad + g / args.meta_batch
            else:
                ql = task_loss(model, fast, bq)
                (ql / args.meta_batch).backward()
            qlosses.append(ql.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)
        if it % 10 == 0:
            log(step=it, query_l1=float(np.mean(qlosses)), support_l1_before=float(np.mean(slosses)),
                it_s=(it + 1) / (time.time() - t0))
        if (it + 1) % args.save_every == 0 or it + 1 == args.iters:
            save(run / "meta.pt", model, ema, args, it + 1, kind, kw)


def cmd_fewshot(args):
    """K-shot adaptation to each new ELD camera; support = K scenes from 1-6, test = 7-10."""
    seed_all(args.seed)
    out = []
    for cam in ELD_CAMERAS:
        for k in args.shots:
            model, ck = load_ckpt(args.init)
            kind = ck["model"]
            cache, sup = eld(args, [cam], ELD_TRAIN[:k]) if k else (Cache(Path(args.cache) / "eld"), [])
            _, test = eld(args, [cam], ELD_TEST, ELD_EVAL_IDS)
            if k:
                ds = PairDataset(cache, sup, ps=args.ps)
                if args.mode == "inner":  # exactly the meta-learning inner loop
                    params = {n: p.detach().clone().requires_grad_(True) for n, p in model.named_parameters()}
                    for _ in range(args.steps):
                        b = batch_of(ds, random.sample(range(len(ds)), min(args.bs, len(ds))))
                        params = adapt(model, params, b, 1, args.inner_lr, False)
                        params = {n: v.detach().requires_grad_(True) for n, v in params.items()}
                    model.load_state_dict({n: v.detach() for n, v in params.items()}, strict=False)
                else:  # ordinary fine-tuning with Adam
                    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                    model.train()
                    for _ in range(args.steps):
                        b = batch_of(ds, random.sample(range(len(ds)), min(args.bs, len(ds))))
                        loss = task_loss(model, dict(model.named_parameters()), b)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
            agg, _ = evaluate(model, cache, test, kind, crop=args.crop)
            row = {"camera": cam, "shots": k, "support_pairs": len(sup), "test_pairs": len(test),
                   "init": args.init, "mode": args.mode, "steps": args.steps, **agg}
            out.append(row)
            print(json.dumps(row), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------- CLI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/content/cache")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    sub = ap.add_subparsers(required=True)

    p = sub.add_parser("prep-sid"); p.set_defaults(func=cmd_prep_sid)
    p.add_argument("--sid-raw", default="/content/Sony")
    p = sub.add_parser("prep-eld"); p.set_defaults(func=cmd_prep_eld)
    p.add_argument("--eld-raw", default="/content/ELD")
    p.add_argument("--cameras", nargs="+", default=ELD_CAMERAS)

    def common(p, iters, lr, ps=512, bs=8):
        p.add_argument("--run", required=True)
        p.add_argument("--iters", type=int, default=iters)
        p.add_argument("--lr", type=float, default=lr)
        p.add_argument("--ps", type=int, default=ps)
        p.add_argument("--bs", type=int, default=bs)
        p.add_argument("--ema", type=float, default=0.999)
        p.add_argument("--val-every", type=int, default=2000)
        p.add_argument("--val-crop", type=int, default=512)
        p.add_argument("--val-n", type=int, default=60)
        p.add_argument("--ms-start", type=int, default=10**9)
        p.add_argument("--ms-weight", type=float, default=0.1)
        p.add_argument("--distill", type=float, default=0.1)
        p.set_defaults(_run_loss=1.0)

    p = sub.add_parser("train"); p.set_defaults(func=cmd_train); common(p, 60000, 2e-4)
    p.add_argument("--model", choices=list(MODELS), required=True)
    p.add_argument("--n-bridge", type=int, default=2)
    p.add_argument("--no-film", action="store_true")
    p.add_argument("--no-distill", action="store_true")
    p.add_argument("--sigma-max", type=float, default=0.1)

    p = sub.add_parser("finetune"); p.set_defaults(func=cmd_finetune); common(p, 8000, 5e-5)
    p.add_argument("--init", required=True)
    p.add_argument("--freeze-encoder", action="store_true")

    p = sub.add_parser("train-ddbm"); p.set_defaults(func=cmd_train_ddbm); common(p, 80000, 1e-4, ps=256, bs=16)
    p.add_argument("--s2", type=float, default=0.5)
    p.add_argument("--ddbm-steps", type=int, default=10)

    p = sub.add_parser("meta"); p.set_defaults(func=cmd_meta)
    p.add_argument("--run", required=True)
    p.add_argument("--init", required=True)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--inner-lr", type=float, default=1e-3)
    p.add_argument("--inner-steps", type=int, default=3)
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--q-scenes", type=int, default=8)
    p.add_argument("--meta-batch", type=int, default=2)
    p.add_argument("--ps", type=int, default=256)
    p.add_argument("--gain-range", type=float, default=0.35)
    p.add_argument("--first-order", action="store_true")
    p.add_argument("--save-every", type=int, default=250)

    p = sub.add_parser("fewshot"); p.set_defaults(func=cmd_fewshot)
    p.add_argument("--init", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shots", type=int, nargs="+", default=[0, 1, 3, 5])
    p.add_argument("--mode", choices=["inner", "adam"], default="adam")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--inner-lr", type=float, default=1e-3)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--ps", type=int, default=512)
    p.add_argument("--crop", type=int, default=1024)

    p = sub.add_parser("eval"); p.set_defaults(func=cmd_eval)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--kind", default="bridge", choices=["sid", "bridge", "ddbm_feature", "ddbm"])
    p.add_argument("--data", default="sid")
    p.add_argument("--out", required=True)
    p.add_argument("--crop", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--ddbm-steps", type=int, default=10)
    p.add_argument("--eta", type=float, default=1.0)

    args = ap.parse_args()
    if not hasattr(args, "sid_raw"):
        args.sid_raw = "/content/Sony"
    args.func(args)


if __name__ == "__main__":
    main()
