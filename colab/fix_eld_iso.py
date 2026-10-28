"""Add EXIF exposure and ISO to an existing ELD meta.json (arrays are unchanged)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/content/ldr-src")
from ldr.raw import exif

raw, cache = Path("/content/ELDraw/ELD"), Path("/content/cache/eld")
meta = json.loads((cache / "meta.json").read_text())
for stem in meta:
    cam, scene, _, idx = stem.split("_")
    f = next((raw / cam / scene).glob(f"IMG_{idx}.*"))
    meta[stem].update(exif(f))
(cache / "meta.json").write_text(json.dumps(meta))
from ldr import raw as R
for cam in ["CanonEOS70D", "CanonEOS700D", "NikonD850"]:
    ps = R.eld_pairs(cache, cam, range(1, 11))
    by = {}
    for p in ps:
        by.setdefault(int(p.short.split("_")[-1]) % 5, set()).add(round(p.ratio))
    print(cam, len(ps), {k: sorted(v) for k, v in sorted(by.items())})
