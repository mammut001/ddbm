"""LRID_simplified (Redmi K30 phone, Sony IMX686; Feng et al., PMN) in the cache format of ldr/raw.py, indoor_x3 subset.

Per scene: noisy ISO-6400 DNG frames at digital gains 1, 2 and 4 (one frame kept each) and a multi-frame fused clean raw
(npy/GT_align_ours/<scene>.npy, 10-bit units with black level 64). The target is rendered from the clean raw: bilinear
demosaic at full resolution, white balance from metadata, a per-scene exposure that brings the mean linear value to
0.18, sRGB gamma, 16 bit. The ratio of a pair is the exposure of that rendering divided by the brightness the noisy
frame already has (mean clean / mean noisy, from the clean raw), so ratio x noisy has the target's brightness; it uses
the clean raw, so the study measures adaptation, not blind use. The long frame is a zero placeholder.

    python colab/lrid_prep.py --root /content/lrid/indoor_x3 --n-test 10
writes /content/cache/custom/lrid/{short,long,gt,meta.json,pairs_train.json,pairs_test.json,pairs.json}.
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import rawpy
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, "/content/ldr-src")
from ldr import raw as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/content/lrid/indoor_x3")
ap.add_argument("--n-test", type=int, default=10)
ap.add_argument("--out", default="/content/cache/custom/lrid")
args = ap.parse_args()
root, out = Path(args.root), Path(args.out)
for d in ("short", "long", "gt"):
    (out / d).mkdir(parents=True, exist_ok=True)
BLACK, WHITE = 64.0, 1023.0


def demosaic_rggb(x):
    """x: (H, W) float tensor, pattern R G / G B; returns (3, H, W) bilinear."""
    h, w = x.shape
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    mr = ((yy % 2 == 0) & (xx % 2 == 0)).float()
    mb = ((yy % 2 == 1) & (xx % 2 == 1)).float()
    mg = 1 - mr - mb
    k_rb = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 4
    k_g = torch.tensor([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=torch.float32) / 4
    conv = lambda t, k: F.conv2d(F.pad(t[None, None], (1, 1, 1, 1), mode="reflect"), k[None, None])[0, 0]
    return torch.stack([conv(x * mr, k_rb), conv(x * mg, k_g), conv(x * mb, k_rb)])


scenes = sorted(os.listdir(root / "npy" / "GT_align_ours"))
scenes = [s[:-4] for s in scenes]
zero = out / "long" / "zero.npy"
meta, pairs = {}, {"train": [], "test": []}
split_at = len(scenes) - args.n_test
for si, sc in enumerate(scenes):
    split = "train" if si < split_at else "test"
    gt_raw = np.load(root / "npy" / "GT_align_ours" / f"{sc}.npy").astype(np.float32)
    h, w = gt_raw.shape
    if not zero.exists():
        np.save(zero, np.zeros((h // 2, w // 2, 4), np.uint16))
    gtn = np.clip((gt_raw - BLACK) / (WHITE - BLACK), 0, 1)
    (out / "clean").mkdir(exist_ok=True)  # the clean raw, packed R, G1, B, G2 and normalised, for noise synthesis
    np.save(out / "clean" / f"{sc}.npy", np.stack([gtn[0::2, 0::2], gtn[0::2, 1::2], gtn[1::2, 1::2], gtn[1::2, 0::2]], -1).astype(np.float16))
    f1 = sorted(glob.glob(str(root / "6400" / "1" / sc / "*.dng")))[0]
    with rawpy.imread(f1) as r:  # the white balance the camera wrote into the noisy frame of this scene
        wb = np.array(r.camera_whitebalance[:3], dtype=np.float32)
    wb = wb / max(wb[1], 1e-6)
    rgb = demosaic_rggb(torch.from_numpy(gtn)).numpy() * wb[:, None, None]
    lum = float(rgb.mean())
    expo = 0.18 / max(lum, 1e-6)
    srgb = np.clip(rgb * expo, 0, 1) ** (1 / 2.2)
    l = f"lrid_{sc}_l"
    np.save(out / "gt" / f"{l}.npy", (srgb.transpose(1, 2, 0) * 65535).astype(np.uint16))
    link = out / "long" / f"{l}.npy"
    if not link.exists():
        link.symlink_to(zero)
    meta[l] = {"black": BLACK, "white": WHITE, "wb": [1.0, 1.0, 1.0]}
    for g in (1, 2, 4):
        f = sorted(glob.glob(str(root / "6400" / str(g) / sc / "*.dng")))[0]
        with rawpy.imread(f) as r:
            packed, lv = R.pack_bayer(r)
        s = f"lrid_{sc}_g{g}_s"
        np.save(out / "short" / f"{s}.npy", packed)
        noisy_mean = float(np.maximum(packed.astype(np.float32) - BLACK, 0).mean() / (WHITE - BLACK))
        gt_mean = float(gtn.mean())
        ratio = float(np.clip(expo * gt_mean / max(noisy_mean, 1e-6), 1.0, 300.0))
        meta[s] = {"black": BLACK, "white": WHITE, "wb": [float(v) for v in wb]}
        pairs[split].append({"short": s, "long": l, "scene": f"lrid/{sc}", "ratio": ratio, "camera": "RedmiK30"})
    print(sc, split, f"expo {expo:.2f}", flush=True)
(out / "meta.json").write_text(json.dumps(meta))
for k in ("train", "test"):
    (out / f"pairs_{k}.json").write_text(json.dumps(pairs[k]))
(out / "pairs.json").write_text(json.dumps(pairs["train"] + pairs["test"]))
r = sorted(p["ratio"] for p in pairs["train"] + pairs["test"])
print("ratios: min %.1f median %.1f max %.1f; train %d test %d pairs" % (r[0], r[len(r) // 2], r[-1], len(pairs["train"]), len(pairs["test"])))
