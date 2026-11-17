"""SIED Sony subset (Jiang et al., ICCV 2025: Sony A7R III, 3840x2160 crops) in the cache format of ldr/raw.py.

The low-light frames are bare uint16 mosaics (black 512, white 16383, packed top-left, top-right, bottom-right,
bottom-left like R, G1, B, G2 in the authors' loader) and the references are sRGB JPGs. There is no exposure ratio in the
files, so it is estimated per image as (mean linearised reference) / (mean normalised mosaic), clipped to [1, 300]; this
uses the reference, so it is meant for evaluating adaptation, not for blind use. The long frame of a pair is a zero
placeholder (the U-Net never reads it), the target is the JPG scaled to 16 bit.

    python colab/sied_prep.py --zips /content --level 0.001 --n-train 100
writes /content/cache/custom/sied/{short,long,gt,meta.json,pairs_train.json,pairs_test.json,pairs.json}.
"""
import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--zips", default="/content", help="folder holding Sony_gt.zip and Sony_<level>.zip")
ap.add_argument("--level", default="0.001")
ap.add_argument("--n-train", type=int, default=100)
ap.add_argument("--out", default="/content/cache/custom/sied")
args = ap.parse_args()
RAW = "https://raw.githubusercontent.com/JianghaiSCU/SIED/main/data/"
out = Path(args.out)
for d in ("short", "long", "gt"):
    (out / d).mkdir(parents=True, exist_ok=True)


def ids(split):
    txt = urllib.request.urlopen(f"{RAW}{split}_sony_{args.level}.txt", timeout=60).read().decode()
    return [Path(line.split()[0]).stem for line in txt.splitlines() if line.strip()]


test_ids = ids("test")
train_all = ids("train")
step = max(len(train_all) // args.n_train, 1)
train_ids = train_all[::step][:args.n_train]  # spread over the list, so over the scenes
print(f"test {len(test_ids)}, train {len(train_ids)} of {len(train_all)}")
zr = zipfile.ZipFile(Path(args.zips) / f"Sony_{args.level}.zip")
zg = zipfile.ZipFile(Path(args.zips) / "Sony_gt.zip")
zero = out / "long" / "zero.npy"
if not zero.exists():
    np.save(zero, np.zeros((1080, 1920, 4), np.uint16))
meta, pairs = {}, {"train": [], "test": []}
for split, lst in (("train", train_ids), ("test", test_ids)):
    for i, n in enumerate(lst):
        s, l = f"sied_{n}_s", f"sied_{n}_l"
        mosaic = np.frombuffer(zr.read(f"{args.level}/{n}.raw"), np.uint16).reshape(2160, 3840)
        packed = np.stack([mosaic[0::2, 0::2], mosaic[0::2, 1::2], mosaic[1::2, 1::2], mosaic[1::2, 0::2]], axis=-1)
        gt = np.asarray(Image.open(io.BytesIO(zg.read(f"gt/{n}.jpg"))).convert("RGB"))
        np.save(out / "short" / f"{s}.npy", packed)
        np.save(out / "gt" / f"{l}.npy", gt.astype(np.uint16) * 257)
        link = out / "long" / f"{l}.npy"
        if not link.exists():
            link.symlink_to(zero)
        norm = np.maximum(packed.astype(np.float32) - 512, 0) / (16383 - 512)
        lin = (gt.astype(np.float32) / 255.0) ** 2.2
        ratio = float(np.clip(lin.mean() / max(norm.mean(), 1e-6), 1.0, 300.0))
        meta[s] = {"black": 512.0, "white": 16383.0, "wb": [1.0, 1.0, 1.0]}
        meta[l] = {"black": 512.0, "white": 16383.0, "wb": [1.0, 1.0, 1.0]}
        pairs[split].append({"short": s, "long": l, "scene": f"sied/{n}", "ratio": ratio, "camera": "SonyA7R3"})
        if i % 20 == 0:
            print(split, i, n, f"ratio {ratio:.1f}", flush=True)
(out / "meta.json").write_text(json.dumps(meta))
(out / "pairs_train.json").write_text(json.dumps(pairs["train"]))
(out / "pairs_test.json").write_text(json.dumps(pairs["test"]))
(out / "pairs.json").write_text(json.dumps(pairs["train"] + pairs["test"]))
r = sorted(p["ratio"] for p in pairs["test"])
print("test ratios: min %.1f median %.1f max %.1f" % (r[0], r[len(r) // 2], r[-1]))
