"""Ensemble control, offline: two SID U-Nets (seeds 0 and 1) and their average against the two DDBMs' one-step average.

Reads results/results/ens_unet.jsonl (colab/unet_ensemble.py) and ms_readout.jsonl (colab/multisample.py), joins them
by image name and prints the mean of each metric over the common pairs, plus the paired PSNR gain with a bootstrap
interval for the pair average over the first U-Net.

    python colab/compare_ens.py
"""
import json
import random
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results" / "results"
KEYS = ["psnr", "ssim", "msssim", "lpips"]


def rows(name):
    return {r["short"]: r["scores"] for r in map(json.loads, (RES / name).read_text().splitlines()) if r}


ens, ms = rows("ens_unet.jsonl"), rows("ms_readout.jsonl")
common = sorted(set(ens) & set(ms))
print(f"{len(common)} common pairs (ens {len(ens)}, readout {len(ms)})")
table = {
    "U-Net A (seed 0)": [ens[k]["unet_a"] for k in common],
    "U-Net B (seed 1)": [ens[k]["unet_b"] for k in common],
    "U-Net A+B average": [ens[k]["unet_ab"] for k in common],
    "U-Net (readout file)": [ms[k]["unet"] for k in common],
    "DDBM pair, one-step average": [ms[k]["naive_r+ustart_r"] for k in common],
}
print(f"{'method':32s}" + "".join(f"{m:>9s}" for m in KEYS))
for name, sc in table.items():
    print(f"{name:32s}" + "".join(f"{sum(s[m] for s in sc) / len(sc):9.3f}" for m in KEYS))

random.seed(0)
for a, b in [("U-Net A+B average", "U-Net A (seed 0)"), ("DDBM pair, one-step average", "U-Net A (seed 0)"),
             ("DDBM pair, one-step average", "U-Net A+B average")]:
    d = [x["psnr"] - y["psnr"] for x, y in zip(table[a], table[b])]
    bs = sorted(sum(random.choices(d, k=len(d))) / len(d) for _ in range(2000))
    print(f"PSNR {a} - {b}: {sum(d) / len(d):+.3f} dB  95% CI [{bs[50]:+.3f}, {bs[1949]:+.3f}]")
