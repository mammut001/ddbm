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

for key, name in [("SONEFIVE", "ddbm_official_s1_test_n5.json"), ("SONETEN", "ddbm_official_s1_test_n10.json"),
                  ("SIDRERUN", "sid_test.json")]:
    d = js(name)
    N[key + "PSNR"], N[key + "SSIM"] = f(d and d["psnr"], 2), f(d and d["ssim"], 3)
    N[key + "MS"] = f(d and d["msssim"], 3)
d = js("sid_test.json")
N["SIDSEEDGAP"] = f(d and abs(d["psnr"] - 28.6197), 2)

log = ROOT / "results" / "runs" / "ddbm_official_s1" / "log.jsonl"
vals = [json.loads(line) for line in log.open() if '"val"' in line] if log.exists() else []
N["SONEVAL"] = f(max(v["val"]["psnr"] for v in vals), 2) if vals else "--"
(REP / "data" / "val_ddbm_official_s1.dat").write_text(
    "step psnr\n" + "".join(f"{v['step']} {v['val']['psnr']:.2f}\n" for v in vals) if vals else "step psnr\n2000 nan\n")

bench = js("bench.json") or {}
b = lambda k: f(bench[k]["seconds"], 3) if k in bench else "--"
N.update(TSID=b("SID U-Net"), TBRIDGE=b("Bridge U-Net"), TNOFILM=b("Bridge w/o FiLM"),
         TFEATZERO=b("DDBM-Feature N=0"), TFEATTHREE=b("DDBM-Feature N=3"),
         TDDBMFIVE=b("DDBM (ours, VE-BB) N=5"), TDDBMTEN=b("DDBM (ours, VE-BB) N=10"),
         TOFFFIVE=b("DDBM (official, σmax=1) N=5"), TOFFTEN=b("DDBM (official, σmax=1) N=10"))

(REP / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(N.items())))
print(open(REP / "numbers.tex").read())
