"""Write report/numbers.tex (and the sigma_max=1 val curve) from the files synced into results/."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES, REP = ROOT / "results" / "results", ROOT / "report"
N = {"SIDTRAINSCENES": 161, "SIDTRAINPAIRS": 1865, "SIDVALSCENES": 20, "SIDVALPAIRS": 234,
     "SIDTESTSCENES": 50, "SIDTESTPAIRS": 598, "ELDFTPAIRS": 216, "METAITERS": "2{,}000",
     "METABATCH": 2, "METATASKS": "4{,}000", "SONECOMMENT": ""}


def js(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def f(x, d):
    return f"{x:.{d}f}" if x is not None else "--"


stats = (RES / "stats.txt").read_text() if (RES / "stats.txt").exists() else ""
m = re.search(r"min (\d+) max (\d+)", stats)
N["SIDMINSHORT"], N["SIDMAXSHORT"] = m.groups() if m else ("--", "--")
par = dict(re.findall(r"(\w+) ([\d.]+M)", stats))
for key, name in [("PSID", "sid"), ("PBRIDGE", "bridge"), ("PNOFILM", "bridge_nofilm"),
                  ("PFEAT", "ddbm_feature"), ("PDDBM", "ddbm")]:
    N[key] = par.get(name, "--")
N["BRIDGEPARAMS"], N["DDBMPARAMS"] = N["PBRIDGE"], N["PDDBM"]

# third run (retrained after the VM loss): SID U-Net seed 0, simplified DDBM and sigma_max=1 DDBM at 60k, with LPIPS;
# fourth run: the full DDBM (VP bridge, clipped sampler) with 20 and the released 40 steps
for key, name in [("SONEFIVE", "ddbm_official_s1_test_n5.json"), ("SONETEN", "ddbm_official_s1_test_n10.json"),
                  ("SONETWENTY", "ddbm_official_s1_test_n20.json"), ("SIDRE", "sid_test.json"),
                  ("DDBMREFIVE", "ddbm_test_n5.json"), ("DDBMRETEN", "ddbm_test_n10.json"),
                  ("VPTWENTY", "ddbm_vp_test_n20.json"), ("VPFORTY", "ddbm_vp_test_n40.json")]:
    d = js(name)
    N[key + "PSNR"], N[key + "SSIM"] = f(d and d["psnr"], 2), f(d and d["ssim"], 3)
    N[key + "MS"], N[key + "LPIPS"] = f(d and d["msssim"], 3), f(d and d.get("lpips"), 3)
# second seed of the SID U-Net, from the second run (its json was overwritten by the third run)
N.update(SIDRERUNPSNR="28.61", SIDRERUNSSIM="0.796", SIDRERUNMS="0.876", SIDSEEDGAP=f(abs(28.6145 - 28.6197), 2))

for key, run in [("SONE", "ddbm_official_s1"), ("VP", "ddbm_vp")]:
    log = ROOT / "results" / "runs" / run / "log.jsonl"
    rows = [json.loads(line) for line in log.open()] if log.exists() else []
    vals = [r for r in rows if "val" in r]
    best = max(vals, key=lambda v: v["val"]["psnr"]) if vals else None
    at60 = [v for v in vals if v["step"] == 60000]
    N[key + "VAL"] = f(best and best["val"]["psnr"], 2)
    N[key + "VALSIXTY"] = f(at60[0]["val"]["psnr"] if at60 else None, 2)
    N[key + "BESTSTEP"] = f"{best['step'] // 1000}k" if best else "--"
    N[key + "ITERS"] = f"{rows[-1]['step'] // 1000}k" if rows else "--"
    (REP / "data" / f"val_{run}.dat").write_text(
        "step psnr\n" + "".join(f"{v['step']} {v['val']['psnr']:.2f}\n" for v in vals) if vals else "step psnr\n2000 nan\n")

bench = js("bench.json") or {}
b = lambda k: f(bench[k]["seconds"], 3) if k in bench else "--"
N.update(TSID=b("SID U-Net"), TBRIDGE=b("Bridge U-Net"), TNOFILM=b("Bridge w/o FiLM"),
         TFEATZERO=b("DDBM-Feature N=0"), TFEATTHREE=b("DDBM-Feature N=3"),
         TDDBMFIVE=b("DDBM (ours, VE-BB) N=5"), TDDBMTEN=b("DDBM (ours, VE-BB) N=10"),
         TOFFFIVE=b("DDBM (official, σmax=1) N=5"), TOFFTEN=b("DDBM (official, σmax=1) N=10"))
tsid = bench.get("SID U-Net", {}).get("seconds")
bench = js("bench_n20_40.json") or {}
N.update(TVPTWENTY=b("DDBM (full, VP bridge) N=20"), TVPFORTY=b("DDBM (full, VP bridge) N=40"))
tvp = bench.get("DDBM (full, VP bridge) N=40", {}).get("seconds")

# fifth run: x = (1 - w) U-Net + w DDBM (N=40) for every w on the same sample; w is chosen on validation
def blend(split):
    p = RES / f"blend_{split}.jsonl"
    rows = [json.loads(line) for line in p.open()] if p.exists() else []
    return {w: {k: sum(r["scores"][w][k] for r in rows) / len(rows) for k in ("psnr", "ssim", "msssim", "lpips")}
            for w in (rows[0]["scores"] if rows else [])}


val, test = blend("val"), blend("test")
if val and test:
    # best validation PSNR, best validation LPIPS, and the smallest weight of the grid
    for key, w in [("BLENDQ", max(val, key=lambda w: val[w]["psnr"])),
                   ("BLENDL", min(val, key=lambda w: val[w]["lpips"])), ("BLENDE", "0.125")]:
        N[key + "W"] = w
        N[key + "PSNR"], N[key + "SSIM"] = f(test[w]["psnr"], 2), f(test[w]["ssim"], 3)
        N[key + "MS"], N[key + "LPIPS"] = f(test[w]["msssim"], 3), f(test[w]["lpips"], 3)
    N["BLENDVALPSNR"], N["BLENDVALUNET"] = f(val[N["BLENDQW"]]["psnr"], 2), f(val["0"]["psnr"], 2)
    (REP / "data" / "blend_test.dat").write_text(
        "w psnr lpips\n" + "".join(f"{w} {v['psnr']:.3f} {v['lpips']:.4f}\n" for w, v in test.items()))
N["TBLEND"] = f(tsid + tvp, 3) if tsid and tvp else "--"
N["PBLEND"] = "--" if "--" in (N["PSID"], N["PDDBM"]) else f"{float(N['PSID'][:-1]) + float(N['PDDBM'][:-1]):.2f}M"
# the other models in the PSNR-LPIPS plane of the blend figure
(REP / "data" / "pd_models.csv").write_text("label;lpips;psnr\n" + "".join(
    f"{label};{d['lpips']:.4f};{d['psnr']:.3f}\n" for label, d in
    [("simplified DDBM", js("ddbm_test_n5.json")), ("VE $\\sigma_{\\max}=1$", js("ddbm_official_s1_test_n20.json")),
     ("full DDBM, $N=20$", js("ddbm_vp_test_n20.json"))] if d))

(REP / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(N.items())))
print(open(REP / "numbers.tex").read())
