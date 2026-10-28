"""RAW reading, Bayer packing and the SID / ELD pair lists.

Everything is cached once as uint16 .npy so training never decodes a RAW file:
  short/<stem>.npy  packed Bayer (H/2, W/2, 4) in RGGB order, raw counts
  long/<stem>.npy   packed Bayer of the long exposure (teacher input), raw counts
  gt/<stem>.npy     long exposure through rawpy (camera WB, 16-bit sRGB), (H, W, 3)
  meta.json         black/white level, white-balance gains and exposure per file
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# Index of R, G1, B, G2 inside a 2x2 CFA tile, for every Bayer layout rawpy reports.
RGGB = [0, 1, 3, 2]  # rawpy colour ids: 0 R, 1 G, 2 B, 3 G2


def pack_bayer(raw) -> tuple[np.ndarray, dict]:
    """rawpy object -> (H/2, W/2, 4) uint16 in R, G1, B, G2 order, plus levels."""
    im = raw.raw_image_visible.astype(np.uint16)
    pattern = raw.raw_pattern  # 2x2 colour ids
    if pattern.shape != (2, 2):
        raise ValueError("not a Bayer sensor (X-Trans is unsupported)")
    h, w = im.shape[0] // 2 * 2, im.shape[1] // 2 * 2
    im = im[:h, :w]
    planes = {}
    g_seen = 0
    for dy in range(2):
        for dx in range(2):
            c = int(pattern[dy, dx])
            if c in (1, 3):
                key = "G1" if g_seen == 0 else "G2"
                g_seen += 1
            else:
                key = "R" if c == 0 else "B"
            planes[key] = im[dy::2, dx::2]
    packed = np.stack([planes["R"], planes["G1"], planes["B"], planes["G2"]], axis=-1)
    black = float(np.mean(raw.black_level_per_channel))
    white = float(raw.white_level)
    wb = [float(x) for x in raw.camera_whitebalance[:3]]
    return packed, {"black": black, "white": white, "wb": wb}


def gt_rgb(raw) -> np.ndarray:
    """Long exposure -> 16-bit sRGB exactly as SID builds its targets."""
    return raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)


@dataclass
class Pair:
    short: str      # cache stem of the short exposure
    long: str       # cache stem of the long exposure (shared by every short of a scene)
    scene: str      # scene id; a task in meta-learning
    ratio: float    # long / short exposure, clipped at 300 like SID
    camera: str = "SonyA7S2"


# ---------------------------------------------------------------- SID Sony

SID_NAME = re.compile(r"(\d{5})_(\d{2})_([\d.]+)s\.ARW$")


def sid_pairs(root: Path) -> dict[str, list[Pair]]:
    """Split by SID's own scene-id convention: 0xxxx train, 1xxxx test, 2xxxx validation."""
    longs = {}
    for f in sorted((root / "long").glob("*.ARW")):
        m = SID_NAME.match(f.name)
        longs[m.group(1)] = (f.stem, float(m.group(3)))
    out = {"train": [], "val": [], "test": []}
    split = {"0": "train", "1": "test", "2": "val"}
    for f in sorted((root / "short").glob("*.ARW")):
        m = SID_NAME.match(f.name)
        scene, exp = m.group(1), float(m.group(3))
        if scene not in longs:
            continue
        lstem, lexp = longs[scene]
        out[split[scene[0]]].append(Pair(f.stem, lstem, scene, min(lexp / exp, 300.0)))
    return out


# ---------------------------------------------------------------- ELD

ELD_GT_IDS = np.array([1, 6, 11, 16])


def eld_pairs(root: Path, camera: str, scenes, img_ids=None) -> list[Pair]:
    """ELD protocol (Wei et al.): within each scene, images 1/6/11/16 are references and
    the frames after each reference are its short exposures; the ratio comes from EXIF
    exposure times written to meta.json during preparation."""
    meta = json.loads((root / "meta.json").read_text())
    pairs = []
    for s in scenes:
        for i in (img_ids or [i for i in range(1, 17) if i not in ELD_GT_IDS]):
            g = int(ELD_GT_IDS[np.argmin(np.abs(i - ELD_GT_IDS) + (ELD_GT_IDS > i) * 99)])
            short = f"{camera}_scene-{s}_IMG_{i:04d}"
            long = f"{camera}_scene-{s}_IMG_{g:04d}"
            if short not in meta or long not in meta:
                continue
            # ELD defines the ratio on ISO x exposure time: references are shot at a lower ISO
            gain = lambda m: m["exposure"] * m.get("iso", 1.0)
            ratio = gain(meta[long]) / gain(meta[short])
            pairs.append(Pair(short, long, f"{camera}/scene-{s}", min(ratio, 300.0), camera))
    return pairs


# ---------------------------------------------------------------- caching


def exif(path: Path) -> dict:
    """Exposure time (s) and ISO from EXIF; SID file names carry the exposure as a fallback."""
    out = {}
    try:
        import exifread
        with open(path, "rb") as fh:
            tags = exifread.process_file(fh, details=False)
        v = tags["EXIF ExposureTime"].values[0]
        out["exposure"] = float(v.num) / float(v.den)
        iso = tags.get("EXIF ISOSpeedRatings") or tags.get("EXIF RecommendedExposureIndex")
        if iso is not None:
            out["iso"] = float(iso.values[0])
    except Exception:
        m = SID_NAME.match(path.name)
        out["exposure"] = float(m.group(3)) if m else float("nan")
    return out


def _cache_one(job):
    src, stem, out, want_gt = job
    import rawpy
    res = {}
    with rawpy.imread(str(src)) as raw:
        packed, levels = pack_bayer(raw)
        res.update(levels)
        kind = "long" if want_gt else "short"
        p = out / kind / f"{stem}.npy"
        if not p.exists():
            np.save(p, packed)
        if want_gt:
            g = out / "gt" / f"{stem}.npy"
            if not g.exists():
                np.save(g, gt_rgb(raw))
    res.update(exif(Path(src)))
    return stem, res


def build_cache(jobs: list[tuple[Path, str, bool]], out: Path, workers: int = os.cpu_count() or 4):
    """jobs: (raw path, cache stem, is_long). Writes arrays and merges meta.json."""
    for d in ("short", "long", "gt"):
        (out / d).mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    todo = [(src, stem, out, is_long) for src, stem, is_long in jobs if stem not in meta]
    with Pool(workers) as pool:
        for i, (stem, res) in enumerate(pool.imap_unordered(_cache_one, todo, chunksize=2)):
            meta[stem] = res
            if i % 100 == 0:
                print(f"cached {i + 1}/{len(todo)}", flush=True)
    meta_path.write_text(json.dumps(meta))
    return meta
