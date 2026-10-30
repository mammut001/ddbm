"""Sanity checks for the released-recipe bridges, run on CPU before a long training run.

With an oracle network that makes D return the true x_0, each sampler must carry x_T to x_0;
the VP marginal must start at x_0 and end at x_T, and the preconditioning must stay finite.
Exits with status 1 if a check fails.
"""

import sys

import torch

sys.path.insert(0, "/content/ldr-src")
from ldr.ddbm import KarrasBridge, VPBridge  # noqa: E402

torch.manual_seed(0)
x0 = torch.rand(2, 3, 32, 32) * 1.6 - 0.8
xT = (x0 + 0.3 * torch.randn_like(x0)).clamp(-1, 1)


def oracle(br):
    def net(xin, c_noise, xT_, raw, cond):
        s = (4 * c_noise).exp()
        c_skip, c_out, c_in = [v.view(-1, 1, 1, 1) for v in br.scalings(s)]
        return (x0 - c_skip * xin / c_in) / c_out
    return net


fails = []
for name, br in [("ve sigma_max=80", KarrasBridge()), ("ve sigma_max=1", KarrasBridge(sigma_max=1.0)),
                 ("vp", VPBridge())]:
    s = torch.linspace(br.smin, br.smax - 1e-4, 200)
    sc = torch.stack(br.scalings(s))
    if not torch.isfinite(sc).all() or (sc[1] <= 0).any():
        fails.append(f"{name}: scalings not finite or c_out <= 0")
    for churn in (0.0, 0.33):
        for steps in (10, 40):
            x = br.sample(oracle(br), xT, None, None, steps=steps, churn=churn)
            rmse = (x - x0).pow(2).mean().sqrt().item()
            print(f"{name:16s} churn {churn:.2f} N={steps:2d} NFE={br.nfe:3d}  rmse to x0 {rmse:.2e}")
            if rmse > 2e-2:
                fails.append(f"{name} churn {churn} N={steps}: rmse {rmse:.3f}")

vp = VPBridge()
a, b, c = vp.coefs(torch.tensor([vp.smin, vp.smax - 1e-4]))
print(f"vp marginal at sigma_min: a={a[0]:.2e} b={b[0]:.4f} c={c[0]:.2e};  at sigma_max: a={a[1]:.4f} b={b[1]:.2e} c={c[1]:.2e}")
if not (a[0] < 1e-3 and abs(b[0] - 1) < 1e-3 and c[0] < 1e-2 and abs(a[1] - 1) < 1e-3 and b[1] < 1e-3 and c[1] < 2e-2):
    fails.append("vp marginal endpoints")

print("FAIL: " + "; ".join(fails) if fails else "ALL BRIDGE CHECKS PASSED")
sys.exit(1 if fails else 0)
