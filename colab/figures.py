"""Report figures: training curves, qualitative crops on SID test, and clean inference timing.

    python colab/figures.py curves | crops | bench
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/content/ldr-src")
import main as M  # noqa: E402
from ldr.data import PairDataset  # noqa: E402
from ldr.ddbm import Bridge, KarrasBridge, naive_render  # noqa: E402
from ldr.models import BridgeDenoiser  # noqa: E402

R, OUT = Path("/content/runs"), Path("/content/results/fig")
OUT.mkdir(parents=True, exist_ok=True)
NAMES = {"sid": "SID U-Net", "bridge": "Bridge U-Net", "bridge_nodistill": "Bridge w/o distill",
         "bridge_nofilm": "Bridge w/o FiLM", "ddbm_feature": "DDBM-Feature", "ddbm": "DDBM (ours, VE-BB)",
         "ddbm_official": "DDBM (official, σmax=80)",
         "ddbm_official_s1": "DDBM (official, σmax=1)"}


def load(name):
    p = R / name / "best.pt"
    if not p.exists():
        return None, None, None
    if name.startswith("ddbm") and name != "ddbm_feature":
        ck = torch.load(p, map_location=M.DEV)
        net = BridgeDenoiser().to(M.DEV)
        net.load_state_dict(ck["ema"])
        br = KarrasBridge(sigma_max=ck["args"].get("bridge_sigma_max", 80.0)) if ck["args"].get("official") else Bridge(s2=ck["args"]["s2"])
        return net.eval(), "ddbm", br
    m, ck = M.load_ckpt(p)
    return m.eval(), ck["model"], None


def curves():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for name, label in NAMES.items():
        f = R / name / "log.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(line) for line in f.open()]
        tr = [(r["step"], r["loss"]) for r in rows if "loss" in r and "val" not in r]
        va = [(r["step"], r["val"]) for r in rows if "val" in r]
        if tr:
            s, l = zip(*tr)
            k = max(1, len(l) // 100)
            ax[0].plot(s[::k], np.convolve(l, np.ones(k) / k, "same")[::k], label=label, lw=1)
        if va:
            s, v = zip(*va)
            ax[1].plot(s, [x["psnr"] for x in v], label=label, lw=1.2, marker=".")
            ax[2].plot(s, [x["msssim"] for x in v], label=label, lw=1.2, marker=".")
    ax[0].set_yscale("log")
    for a, t in zip(ax, ["training loss", "val PSNR (dB)", "val MS-SSIM"]):
        a.set_title(t)
        a.set_xlabel("step")
        a.grid(alpha=.3)
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "curves.pdf")
    # meta-learning curve
    f = R / "meta" / "log.jsonl"
    if f.exists():
        rows = [json.loads(line) for line in f.open() if "query_l1" in line]
        s = [r["step"] for r in rows]
        fig, a = plt.subplots(figsize=(4.5, 3))
        k = 10
        sm = lambda v: np.convolve(v, np.ones(k) / k, "valid")
        a.plot(s[k - 1:], sm([r["support_l1_before"] for r in rows]), label="support L1 before adaptation")
        a.plot(s[k - 1:], sm([r["query_l1"] for r in rows]), label="query L1 after adaptation")
        a.set_xlabel("meta step")
        a.legend(fontsize=7)
        a.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(OUT / "meta_curve.pdf")


def to_img(t):
    return (t[0].permute(1, 2, 0).float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


@torch.no_grad()
def crops():
    from PIL import Image
    cache, pairs = M.sid_split(type("A", (), {"cache": "/content/cache"})(), "test")
    pick = []
    for want in (300.0, 250.0, 100.0):
        pick += [p for p in pairs if abs(p.ratio - want) < 1][::40][:2]
    ds = PairDataset(cache, pick, ps=None, train=False)
    models = {n: load(n) for n in NAMES}
    for i, p in enumerate(pick):
        b = {k: (v[None].to(M.DEV) if torch.is_tensor(v) else v) for k, v in ds[i].items()}
        h, w = b["gt"].shape[-2:]
        y0, x0 = h // 2 - 384, w // 2 - 512
        cut = lambda t: t[..., y0:y0 + 768, x0:x0 + 1024]
        dark = ((naive_render(b["short"] / p.ratio, b["wb"]) + 1) / 2).clamp(0, 1)
        Image.fromarray(to_img(cut(dark))).save(OUT / f"{i}_input_dark.png")
        Image.fromarray(to_img(cut((naive_render(b["short"], b["wb"]) + 1) / 2))).save(OUT / f"{i}_input_amp.png")
        Image.fromarray(to_img(cut(b["gt"]))).save(OUT / f"{i}_gt.png")
        for n, (m, kind, br) in models.items():
            if m is None:
                continue
            y = M.predict(m, b, kind, steps=3, ddbm_steps=10, bridge=br)
            Image.fromarray(to_img(cut(y))).save(OUT / f"{i}_{n}.png")
        (OUT / f"{i}_meta.json").write_text(json.dumps({"short": p.short, "ratio": p.ratio}))


@torch.no_grad()
def bench():
    """Network time per full SID image (1424x2128 packed -> 2848x4256 RGB), GPU only."""
    x = torch.rand(1, 4, 1424, 2128, device=M.DEV)
    b = {"short": x, "wb": torch.tensor([[2.0, 1.0, 1.5]], device=M.DEV),
         "cond": torch.tensor([[0.9, 0.5]], device=M.DEV)}
    rows = {}
    # timing does not depend on the weights, so lost checkpoints are replaced by fresh networks
    fresh = {"sid": ("sid", {}), "bridge": ("bridge", {}), "bridge_nodistill": ("bridge", {}),
             "bridge_nofilm": ("bridge", {"film": False}), "ddbm_feature": ("ddbm_feature", {})}
    for n in NAMES:
        m, kind, br = load(n)
        if m is None and n in fresh:
            kind, kw = fresh[n]
            m, br = M.build(kind, **kw).eval(), None
        elif m is None:
            m, kind = BridgeDenoiser().to(M.DEV).eval(), "ddbm"
            br = {"ddbm": Bridge(), "ddbm_official": KarrasBridge()}.get(n, KarrasBridge(sigma_max=1.0))
        settings = [("", {})]
        if kind == "ddbm":
            settings = [(f" N={s}", {"ddbm_steps": s}) for s in (5, 10)]
        if kind == "ddbm_feature":
            settings = [(" N=0", {"steps": 0}), (" N=3", {"steps": 3})]
        for tag, kw in settings:
            for _ in range(2):
                M.predict(m, b, kind, bridge=br, **kw)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(5):
                M.predict(m, b, kind, bridge=br, **kw)
            torch.cuda.synchronize()
            rows[NAMES[n] + tag] = {"seconds": (time.time() - t0) / 5,
                                    "params_M": sum(p.numel() for p in m.parameters()) / 1e6,
                                    "nfe": getattr(br, "nfe", kw.get("ddbm_steps", 1 + kw.get("steps", 0)))}
            print(NAMES[n] + tag, rows[NAMES[n] + tag], flush=True)
    (OUT.parent / "bench.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    {"curves": curves, "crops": crops, "bench": bench}[sys.argv[1]]()
