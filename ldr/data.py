"""Paired crops from the uint16 cache.

A sample is (packed short x ratio, packed long, sRGB target, condition). The packed arrays
are memory-mapped so a 512x512 crop reads only its own bytes.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .raw import Pair


class Cache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())

    def load(self, kind: str, stem: str) -> np.ndarray:
        return np.load(self.root / kind / f"{stem}.npy", mmap_mode="r")

    def norm_raw(self, arr: np.ndarray, stem: str) -> np.ndarray:
        m = self.meta[stem]
        return np.maximum(arr.astype(np.float32) - m["black"], 0) / (m["white"] - m["black"])


def noise_level(x: np.ndarray) -> float:
    """Robust noise sigma of a normalised packed input (MAD of the green difference image)."""
    d = x[..., 1] - x[..., 3]
    return float(np.median(np.abs(d - np.median(d))) * 1.4826 / math.sqrt(2))


def condition(ratio: float, sigma: float) -> np.ndarray:
    """Two-number FiLM condition: log exposure ratio and log noise, both roughly in [0, 1]."""
    return np.array([math.log(ratio) / math.log(300.0),
                     (math.log10(max(sigma, 1e-5)) + 5.0) / 5.0], dtype=np.float32)


def augment(arrs, flips):
    hf, vf, tr = flips
    out = []
    for a in arrs:
        if hf:
            a = a[:, ::-1]
        if vf:
            a = a[::-1]
        if tr:
            a = a.transpose(1, 0, 2)
        out.append(np.ascontiguousarray(a))
    return out


class PairDataset(Dataset):
    """Random aligned crops for training (ps = packed crop size), whole images otherwise.

    gains: optional 4-vector multiplied into the packed channels, which simulates another
    sensor's colour response for a meta-learning task (the target stays the same).
    """

    def __init__(self, cache: Cache, pairs: list[Pair], ps: int | None = 512, train: bool = True,
                 repeat: int = 1, gains: np.ndarray | None = None):
        self.cache, self.pairs, self.ps, self.train = cache, pairs, ps, train
        self.repeat, self.gains = repeat, gains

    def __len__(self):
        return len(self.pairs) * self.repeat

    def __getitem__(self, i):
        p = self.pairs[i % len(self.pairs)]
        short = self.cache.load("short", p.short)
        long = self.cache.load("long", p.long)
        gt = self.cache.load("gt", p.long)
        h, w = short.shape[:2]
        if self.ps:
            ps = self.ps
            y = random.randint(0, h - ps) if self.train else (h - ps) // 2
            x = random.randint(0, w - ps) if self.train else (w - ps) // 2
            short = short[y:y + ps, x:x + ps]
            long = long[y:y + ps, x:x + ps]
            gt = gt[2 * y:2 * y + 2 * ps, 2 * x:2 * x + 2 * ps]
        else:
            gt = gt[:2 * h, :2 * w]
        s = self.cache.norm_raw(short, p.short)
        l = self.cache.norm_raw(long, p.long)
        if self.gains is not None:
            s = s * self.gains
            l = l * self.gains
        sigma = noise_level(s)
        s = np.minimum(s * p.ratio, 1.0)
        t = gt.astype(np.float32) / 65535.0
        if self.train:
            s, l, t = augment([s, l, t], [random.random() < .5 for _ in range(3)])
        wb = np.array(self.cache.meta[p.short]["wb"], dtype=np.float32)
        wb = wb / max(wb[1], 1e-6)
        f = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)
        return {"short": f(s), "long": f(l), "gt": f(t), "wb": torch.from_numpy(wb),
                "cond": torch.from_numpy(condition(p.ratio, sigma)), "ratio": p.ratio}


def by_scene(pairs: list[Pair]) -> dict[str, list[Pair]]:
    out: dict[str, list[Pair]] = {}
    for p in pairs:
        out.setdefault(p.scene, []).append(p)
    return out
