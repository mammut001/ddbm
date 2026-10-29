"""Numbers quoted in the report: split sizes, shorts per training scene, parameter counts."""
import collections
import sys

sys.path.insert(0, "/content/ldr-src")
from pathlib import Path  # noqa: E402

import ldr.raw as R  # noqa: E402
from ldr.models import BridgeDenoiser, BridgeUNet, DDBMFeatureUNet, SIDUNet  # noqa: E402

sp = R.sid_pairs(Path("/content/Sony"))
for k, v in sp.items():
    print(k, "scenes", len({p.scene for p in v}), "pairs", len(v))
c = collections.Counter(p.scene for p in sp["train"])
print("shorts per train scene: min", min(c.values()), "max", max(c.values()))
n = lambda m: f"{sum(p.numel() for p in m.parameters()) / 1e6:.2f}M"
print("sid", n(SIDUNet()), "bridge", n(BridgeUNet()), "bridge_nofilm", n(BridgeUNet(film=False)),
      "ddbm_feature", n(DDBMFeatureUNet()), "ddbm", n(BridgeDenoiser()))
