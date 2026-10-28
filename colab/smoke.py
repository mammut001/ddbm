"""Synthetic cache + a few iterations of every command, to catch bugs before real data."""

import json
import subprocess
from pathlib import Path

import numpy as np

root = Path("/content/smoke")
rng = np.random.default_rng(0)
H, W = 300, 420  # packed size; GT is 2x


def write(cache, stems_short, stems_long):
    for d in ("short", "long", "gt"):
        (cache / d).mkdir(parents=True, exist_ok=True)
    meta = {}
    for s in stems_short:
        np.save(cache / "short" / f"{s}.npy", rng.integers(512, 900, (H, W, 4), dtype=np.uint16))
        meta[s] = {"black": 512, "white": 16383, "wb": [2.0, 1.0, 1.5], "exposure": 0.1}
    for s in stems_long:
        np.save(cache / "long" / f"{s}.npy", rng.integers(512, 9000, (H, W, 4), dtype=np.uint16))
        np.save(cache / "gt" / f"{s}.npy", rng.integers(0, 65535, (2 * H, 2 * W, 3), dtype=np.uint16))
        meta[s] = {"black": 512, "white": 16383, "wb": [2.0, 1.0, 1.5], "exposure": 10.0}
    (cache / "meta.json").write_text(json.dumps(meta))


pairs = {"train": [], "val": [], "test": []}
shorts, longs = [], []
for split, first, n in (("train", "0", 20), ("val", "2", 2), ("test", "1", 2)):
    for k in range(n):
        scene = f"{first}{k:04d}"
        longs.append(f"{scene}_00_10s")
        for j, (exp, ratio) in enumerate(((0.1, 100.0), (0.04, 250.0), (0.033, 300.0))):
            st = f"{scene}_{j:02d}_{exp}s"
            shorts.append(st)
            pairs[split].append({"short": st, "long": f"{scene}_00_10s", "scene": scene,
                                 "ratio": ratio, "camera": "SonyA7S2"})
write(root / "sid", shorts, longs)
(root / "sid" / "pairs.json").write_text(json.dumps(pairs))

es, el = [], []
for cam in ("CanonEOS70D", "CanonEOS700D", "NikonD850"):
    for s in range(1, 11):
        for i in range(1, 17):
            (el if i in (1, 6, 11, 16) else es).append(f"{cam}_scene-{s}_IMG_{i:04d}")
write(root / "eld", es, el)
meta = json.loads((root / "eld" / "meta.json").read_text())
for k in meta:
    meta[k]["exposure"] = 10.0 if k in el else 0.05
(root / "eld" / "meta.json").write_text(json.dumps(meta))

C = f"cd /content/ldr-src && python main.py --cache {root} --workers 2"
R = "/content/smoke/runs"
cmds = [
    f"{C} train --model sid --run {R}/sid --iters 30 --bs 2 --ps 128 --val-every 15 --val-n 2 --val-crop 128 --ms-start 10",
    f"{C} train --model bridge --run {R}/bridge --iters 30 --bs 2 --ps 128 --val-every 15 --val-n 2 --val-crop 128 --ms-start 10",
    f"{C} train --model ddbm_feature --run {R}/ddbmf --iters 30 --bs 2 --ps 128 --val-every 15 --val-n 2 --val-crop 128",
    f"{C} train-ddbm --run {R}/ddbm --iters 30 --bs 2 --ps 128 --val-every 15 --val-n 2 --val-crop 128 --ddbm-steps 4",
    f"{C} eval --ckpt {R}/bridge/best.pt --out {R}/eval_bridge.json",
    f"{C} eval --ckpt {R}/ddbmf/best.pt --out {R}/eval_ddbmf.json",
    f"{C} eval --ckpt {R}/ddbm/best.pt --kind ddbm --ddbm-steps 4 --out {R}/eval_ddbm.json",
    f"{C} eval --ckpt {R}/ddbm/best.pt --kind ddbm --ddbm-steps 4 --eta 0 --out {R}/eval_ddbm0.json",
    f"{C} finetune --init {R}/bridge/best.pt --run {R}/ft --iters 20 --bs 2 --ps 128 --val-every 10 --ms-start 5 --val-crop 128",
    f"{C} meta --init {R}/bridge/best.pt --run {R}/meta --iters 4 --ps 64 --k-shot 2 --q-scenes 2 --save-every 2",
    f"{C} meta --init {R}/bridge/best.pt --run {R}/meta_fo --iters 4 --ps 64 --k-shot 2 --q-scenes 2 --save-every 2 --first-order",
    f"{C} fewshot --init {R}/meta/meta.pt --out {R}/fs.json --shots 0 2 --steps 3 --ps 64 --crop 128",
    f"{C} fewshot --init {R}/meta/meta.pt --out {R}/fs_inner.json --shots 2 --steps 3 --ps 64 --crop 128 --mode inner",
    f"{C} eval --ckpt {R}/ft/best.pt --data NikonD850 --crop 128 --out {R}/eval_eld.json",
]
for c in cmds:
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    print(("OK  " if r.returncode == 0 else "FAIL") + " " + c.split("main.py")[1][:90])
    if r.returncode:
        print("\n".join((r.stdout + r.stderr).strip().splitlines()[-25:]))
        break
    print("    " + " | ".join(t[:160] for t in tail))
