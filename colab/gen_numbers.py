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
# fourth run: the full DDBM (VP bridge, clipped sampler) with 20 and the released 40 steps;
# sixth run: the same DDBM started from the SID U-Net output, final checkpoint
for key, name in [("SONEFIVE", "ddbm_official_s1_test_n5.json"), ("SONETEN", "ddbm_official_s1_test_n10.json"),
                  ("SONETWENTY", "ddbm_official_s1_test_n20.json"), ("SIDRE", "sid_test.json"),
                  ("DDBMREFIVE", "ddbm_test_n5.json"), ("DDBMRETEN", "ddbm_test_n10.json"),
                  ("VPTWENTY", "ddbm_vp_test_n20.json"), ("VPFORTY", "ddbm_vp_test_n40.json"),
                  ("VPUNET", "ddbm_vp_unet_test_n40.json")]:
    d = js(name)
    N[key + "PSNR"], N[key + "SSIM"] = f(d and d["psnr"], 2), f(d and d["ssim"], 3)
    N[key + "MS"], N[key + "LPIPS"] = f(d and d["msssim"], 3), f(d and d.get("lpips"), 3)
# second seed of the SID U-Net, from the second run (its json was overwritten by the third run)
N.update(SIDRERUNPSNR="28.61", SIDRERUNSSIM="0.796", SIDRERUNMS="0.876", SIDSEEDGAP=f(abs(28.6145 - 28.6197), 2))

for key, run in [("SONE", "ddbm_official_s1"), ("VP", "ddbm_vp"), ("VPU", "ddbm_vp_unet")]:
    log = ROOT / "results" / "runs" / run / "log.jsonl"
    rows = [json.loads(line) for line in log.open()] if log.exists() else []
    vals = [r for r in rows if "val" in r]
    best = max(vals, key=lambda v: v["val"]["psnr"]) if vals else None
    at60 = [v for v in vals if v["step"] == 60000]
    N[key + "VAL"] = f(best and best["val"]["psnr"], 2)
    N[key + "VALSIXTY"] = f(at60[0]["val"]["psnr"] if at60 else None, 2)
    N[key + "BESTSTEP"] = f"{best['step'] // 1000}k" if best else "--"
    N[key + "VALLAST"] = f(vals[-1]["val"]["psnr"] if vals else None, 2)
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
bench = js("bench_n40.json") or {}
N["TVPUNET"] = b("DDBM (full, from U-Net) N=40")

# The SID authors ask to leave three misaligned test scenes out of quantitative results. The report keeps
# all pairs, because the first run's per-image scores are lost, and quotes these for comparison.
MISALIGNED = {"10034", "10045", "10172"}


def aligned(rows, get=lambda r: r["psnr"]):
    keep = [get(r) for r in rows if r["scene"] not in MISALIGNED]
    return sum(keep) / len(keep), len(keep)


for key, name in [("SID", "sid_test.json"), ("VP", "ddbm_vp_test_n40.json")]:
    d = js(name)
    N[key + "ALIGNEDPSNR"] = f(d and aligned(d["rows"])[0], 2)
    if d:
        N["ALIGNEDPAIRS"] = aligned(d["rows"])[1]

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
    rows = [json.loads(line) for line in (RES / "blend_test.jsonl").open()]
    N["BLENDALIGNEDPSNR"] = f(aligned(rows, lambda r: r["scores"][N["BLENDQW"]]["psnr"])[0], 2)
    (REP / "data" / "blend_test.dat").write_text(
        "w psnr lpips\n" + "".join(f"{w} {v['psnr']:.3f} {v['lpips']:.4f}\n" for w, v in test.items()))
N["TBLEND"] = f(tsid + tvp, 3) if tsid and tvp else "--"
N["PBLEND"] = "--" if "--" in (N["PSID"], N["PDDBM"]) else f"{float(N['PSID'][:-1]) + float(N['PDDBM'][:-1]):.2f}M"
# the other models in the PSNR-LPIPS plane of the blend figure
(REP / "data" / "pd_models.csv").write_text("label;lpips;psnr\n" + "".join(
    f"{label};{d['lpips']:.4f};{d['psnr']:.3f}\n" for label, d in
    [("simplified DDBM", js("ddbm_test_n5.json")), ("VE $\\sigma_{\\max}=1$", js("ddbm_official_s1_test_n20.json")),
     ("full DDBM, $N=20$", js("ddbm_vp_test_n20.json"))] if d))


# seventh run (colab/multisample.py): one-step estimates D(x_T, T) of the two full DDBMs on all test pairs, and
# means of samples on 100 test pairs (two per scene), four 40-step samples from each DDBM
import math

MSM = ("psnr", "ssim", "msssim", "lpips")


def ms_rows(name):
    p = RES / name
    return [json.loads(line) for line in p.open()] if p.exists() else []


def ms_get(r, name):
    if name.endswith("_single"):  # the average over the four draws of one image
        return {k: sum(r["scores"][f"{name[:-7]}_s{i}"][k] for i in range(4)) / 4 for k in MSM}
    return r["scores"][name]


def ms_mean(rows, name):
    return {k: sum(ms_get(r, name)[k] for r in rows) / len(rows) for k in MSM}


mse, db = (lambda p: 10 ** (-p / 10)), (lambda x: -10 * math.log10(max(x, 1e-12)))
one, sub = ms_rows("ms_readout.jsonl"), ms_rows("ms_mean.jsonl")
if one and sub:
    nfe_t = float(N["TVPFORTY"]) / 119  # seconds per network evaluation of the 40-step sampler
    t_unet = float(N["TSID"])
    for key, name, secs in [("STEPNAIVE", "naive_r", nfe_t), ("STEPUNET", "ustart_r", nfe_t + t_unet),
                            ("STEPAVG", "naive_r+ustart_r", 2 * nfe_t + t_unet)]:
        m = ms_mean(one, name)
        N[key + "PSNR"], N[key + "SSIM"], N[key + "MS"], N[key + "LPIPS"] = (f(m["psnr"], 2), f(m["ssim"], 3), f(m["msssim"], 3), f(m["lpips"], 3))
        N["T" + key] = f(secs, 3)
    N["PENS"] = f"{float(N['PSID'][:-1]) + 2 * float(N['PDDBM'][:-1]):.2f}M"
    names = {"unet": "unet", "single": "naive_single", "two": "naive_m2", "four": "naive_m4", "eight": "both_m8", "ens": "unet+mean_w0.5",
             "step": "naive_r", "stepavg": "naive_r+ustart_r", "stepu": "ustart_r"}
    S = {k: ms_mean(sub, v) for k, v in names.items()}
    for k, m in S.items():
        for met, tag in zip(MSM, ("PSNR", "SSIM", "MS", "LPIPS")):
            N[f"MS{k.upper()}{tag}"] = f(m[met], 2 if met == "psnr" else 3)
    # MSE(K) = M + V / K per image, from single draws and the mean of four draws; the mean of two is held out
    for tag in ("naive", "ustart"):
        V, M = [], []
        for r in sub:
            m1 = sum(mse(r["scores"][f"{tag}_s{i}"]["psnr"]) for i in range(4)) / 4
            m4 = mse(r["scores"][f"{tag}_m4"]["psnr"])
            v = (m1 - m4) * 4 / 3
            V.append(v)
            M.append(m4 - v / 4)
        inf = sum(db(m) for m in M) / len(M)
        pred2 = sum(db(m + v / 2) for m, v in zip(M, V)) / len(M)
        meas2 = sum(r["scores"][f"{tag}_m2"]["psnr"] for r in sub) / len(sub)
        share = sum(v / (m + v) for m, v in zip(M, V)) / len(M)
        single = sum(ms_get(r, f"{tag}_single")["psnr"] for r in sub) / len(sub)
        N[f"MSINF{tag.upper()}"], N[f"MSERR{tag.upper()}"] = f(inf, 2), f(abs(pred2 - meas2), 3)
        N[f"MSSHARE{tag.upper()}"], N[f"MSGAP{tag.upper()}"] = f(share, 2), f(inf - single, 1)
    # rows of the sample-averaging table, and the path of the figure
    NFE = {"unet": 1, "step": 1, "stepu": 2, "stepavg": 3, "single": 119, "two": 238, "four": 476, "eight": 953, "ens": 953}
    LAB = {"unet": "SID U-Net", "step": "One-step estimate, full DDBM", "stepu": "One-step estimate, full DDBM from the U-Net",
           "stepavg": "Average of the two one-step estimates", "single": "One sample ($N=40$), full DDBM",
           "two": "Mean of 2 samples", "four": "Mean of 4 samples", "eight": "Mean of 4 samples of each of the two DDBMs",
           "ens": "U-Net + mean of the 8 samples, $1:1$"}
    best = {met: (max if met != "lpips" else min)(m[met] for m in S.values()) for met in MSM}
    rows = []
    for k in ("unet", "step", "stepu", "stepavg", "single", "two", "four", "eight", "ens"):
        cells = [(f"{S[k][met]:.2f}" if met == "psnr" else f"{S[k][met]:.3f}") for met in MSM]
        cells = [f"\\textbf{{{c}}}" if S[k][met] == best[met] else c for c, met in zip(cells, MSM)]
        rows.append(f"{LAB[k]} & {NFE[k]} & " + " & ".join(cells) + " \\\\")
        if k in ("unet", "stepavg"):
            rows[-1] += "\n\\midrule" if k == "stepavg" else ""
    # the last row has no \\ of its own: a file that ends in one breaks the row structure right before \bottomrule
    (REP / "data" / "ms_table.tex").write_text("\n".join(rows).rstrip()[:-2].rstrip() + "%\n")
    path = [("1", (ms_mean(sub, "naive_single"), ms_mean(sub, "ustart_single")))]
    pts = [(1, [ms_mean(sub, "naive_single"), ms_mean(sub, "ustart_single")])]
    pts += [(k, [ms_mean(sub, f"both_m{k}")]) for k in (2, 4, 8)]
    (REP / "data" / "ms_path.dat").write_text("k lpips psnr\n" + "".join(
        f"{k} {sum(m['lpips'] for m in ms) / len(ms):.4f} {sum(m['psnr'] for m in ms) / len(ms):.3f}\n" for k, ms in pts))
    (REP / "data" / "ms_points.csv").write_text("label;lpips;psnr\n" + "".join(
        f"{lab};{S[k]['lpips']:.4f};{S[k]['psnr']:.3f}\n" for k, lab in
        [("unet", "SID U-Net"), ("step", "one-step"), ("stepavg", "one-step average"), ("ens", "U-Net + 8-sample mean")]))

# ensemble control (colab/unet_ensemble.py): a second SID U-Net (seed 1) and the average of the two U-Nets, same 598 pairs
ens = {r["short"]: r["scores"] for r in ms_rows("ens_unet.jsonl")}
if ens and one:
    by_name = {r["short"]: r for r in one}
    common = sorted(set(ens) & set(by_name))
    for key, name in [("UB", "unet_b"), ("UAB", "unet_ab")]:
        for met, tag in zip(MSM, ("PSNR", "SSIM", "MS", "LPIPS")):
            v = sum(ens[k][name][met] for k in common) / len(common)
            N[f"{key}{tag}"] = f(v, 2 if met == "psnr" else 3)

    def scene_gap(a, b, met="psnr", sign=1):
        by = {}
        for k in common:
            by.setdefault(by_name[k]["scene"], []).append(a(k)[met] - b(k)[met])
        d = [sum(v) / len(v) for v in by.values()]
        mu = sum(d) / len(d)
        se = (sum((x - mu) ** 2 for x in d) / (len(d) - 1) / len(d)) ** 0.5
        return mu, se, sum(1 for x in d if sign * x > 0), len(d)

    ua, ub, uab = (lambda k: ens[k]["unet_a"]), (lambda k: ens[k]["unet_b"]), (lambda k: ens[k]["unet_ab"])
    dd = lambda k: by_name[k]["scores"]["naive_r+ustart_r"]  # noqa: E731
    for key, (a, b) in {"ABA": (uab, ua), "BA": (ub, ua), "DDA": (dd, ua), "DDAB": (dd, uab)}.items():
        mu, se, win, n = scene_gap(a, b)
        N[f"G{key}"], N[f"G{key}SE"], N[f"G{key}WIN"] = f(mu, 2), f(se, 2), win
    N["GDDABMS"] = f(scene_gap(dd, uab, "msssim")[0], 3)
    N["GDDABLPIPS"] = f(-scene_gap(dd, uab, "lpips")[0], 3)
    N["GDDABLPIPSWIN"] = scene_gap(dd, uab, "lpips", sign=-1)[2]
    N["GDDABMSWIN"] = scene_gap(dd, uab, "msssim")[2]

# zero-shot transfer to the ELD cameras (colab/crosscam.py): models trained on SID Sony only, full ELD test images
cross = ms_rows("crosscam.jsonl")
if cross:
    CN = {"unet_a": "SID U-Net, seed 0", "unet_b": "SID U-Net, seed 1", "unet_ab": "Average of the two U-Nets",
          "naive_r": "DDBM one-step estimate, naive start", "ustart_r": "DDBM one-step estimate, U-Net start",
          "ddbm_pair": "Average of the two DDBM one-step estimates"}
    cams = ["CanonEOS70D", "CanonEOS700D", "NikonD850"]

    def cmean(rows, n, met):
        return sum(r["scores"][n][met] for r in rows) / len(rows)

    cells = {}
    for n in CN:
        cells[n] = [cmean([r for r in cross if r["camera"] == c], n, "psnr") for c in cams] + [cmean(cross, n, m) for m in MSM]
    best = [(min if j == 6 else max)(cells[n][j] for n in CN) for j in range(7)]
    rows = []
    for n, lab in CN.items():
        txt = [f"{v:.2f}" if j in (0, 1, 2, 3) else f"{v:.3f}" for j, v in enumerate(cells[n])]
        txt = [f"\\textbf{{{t}}}" if cells[n][j] == best[j] else t for j, t in enumerate(txt)]
        rows.append(f"{lab} & " + " & ".join(txt) + " \\\\")
    (REP / "data" / "cross_table.tex").write_text("\n".join(rows).rstrip()[:-2].rstrip() + "%\n")
    scenes = sorted({(r["camera"], r["scene"]) for r in cross})
    N["CROSSN"], N["CROSSSCENES"] = len(cross), len(scenes)

    def xgap(a, b, met="psnr"):
        by = {}
        for r in cross:
            by.setdefault((r["camera"], r["scene"]), []).append(r["scores"][a][met] - r["scores"][b][met])
        d = [sum(v) / len(v) for v in by.values()]
        mu = sum(d) / len(d)
        return mu, (sum((x - mu) ** 2 for x in d) / (len(d) - 1) / len(d)) ** 0.5, sum(1 for x in d if x > 0), len(d)

    for key, (a, b) in {"XPAIRA": ("ddbm_pair", "unet_a"), "XUSA": ("ustart_r", "unet_a"), "XABA": ("unet_ab", "unet_a"),
                        "XNAIVEA": ("naive_r", "unet_a")}.items():
        mu, se, win, n = xgap(a, b)
        N[key], N[key + "SE"], N[key + "WIN"] = f(mu, 2), f(se, 2), win
        N[key + "ABS"] = f(abs(mu), 2)
    N["XUSLPIPS"] = f(-xgap("ustart_r", "unet_a", "lpips")[0], 3)
    N["XNAIVECANONB"] = f(cmean([r for r in cross if r["camera"] == "CanonEOS700D"], "naive_r", "psnr"), 1)

# diagnostic (colab/diag_naive.py): brightness of the naive rendering and of the SID U-Net output against the target, ELD
diag = ms_rows("diag_naive.jsonl")
if diag:
    for tag, cam in (("SEVENTY", "CanonEOS70D"), ("SEVENHUNDRED", "CanonEOS700D"), ("NIKON", "NikonD850")):
        rs = [r for r in diag if r["camera"] == cam]
        N[f"DIAGXT{tag}"] = f(sum(r["xT"]["mean_ratio"] for r in rs) / len(rs), 1)
        N[f"DIAGUN{tag}"] = f(sum(r["unet"]["mean_ratio"] for r in rs) / len(rs), 2)
    rs = [r for r in diag if r["camera"] == "CanonEOS700D"]
    N["DIAGBR"] = f(sum(r["xT"]["rgb_ratio"][2] / r["xT"]["rgb_ratio"][0] for r in rs) / len(rs), 1)

# few-shot adaptation of the U-Net and the DDBM (colab/fewshot_ddbm.py): rows pooled over the three ELD cameras
fs = ms_rows("fewshot_ddbm.jsonl")
if fs:
    def pooled(k):
        return [(r["scene"], row["camera"], r["scores"]) for row in fs if row["k"] == k for r in row["rows"]]
    rd = [json.loads(line) for line in open(RES / "fewshot_readout_lr2e-5.jsonl")] if (RES / "fewshot_readout_lr2e-5.jsonl").exists() else []

    def rpool(k):
        if k == 0:  # nothing is adapted: the zero-shot estimate of the DDBM
            return [(r["scene"], row["camera"], {"R": r["scores"]["D"]}) for row in fs if row["k"] == 0 for r in row["rows"]]
        return [(r["scene"], row["camera"], {"R": r["scores"]["D"]}) for row in rd if row["k"] == k for r in row["rows"]]

    lines = []
    for k in sorted({row["k"] for row in fs}):
        P = pooled(k)
        R = rpool(k)
        cells = {v: (sum(x[2][v]["psnr"] for x in P) / len(P), sum(x[2][v]["lpips"] for x in P) / len(P)) for v in ("U", "D", "DU")}
        cells["R"] = (sum(x[2]["R"]["psnr"] for x in R) / len(R), sum(x[2]["R"]["lpips"] for x in R) / len(R)) if R else (float("nan"), float("nan"))
        order = ("U", "D", "R", "DU")
        bestp, bestl = max(cells[v][0] for v in order if cells[v][0] == cells[v][0]), min(cells[v][1] for v in order if cells[v][1] == cells[v][1])
        row_txt = []
        for v in order:
            p_, l_ = cells[v]
            row_txt.append((f"\\textbf{{{p_:.2f}}}" if p_ == bestp else f"{p_:.2f}") + " / " + (f"\\textbf{{{l_:.3f}}}" if l_ == bestl else f"{l_:.3f}"))
        lines.append(f"{k} & " + " & ".join(row_txt) + " \\\\")
        if R and k:
            W = {1: "ONE", 3: "THREE", 6: "SIX"}[k]
            N[f"FSRP{W}"], N[f"FSRL{W}"] = f(cells["R"][0], 2), f(cells["R"][1], 3)
            N[f"FSUP{W}"], N[f"FSUL{W}"] = f(cells["U"][0], 2), f(cells["U"][1], 3)
    (REP / "data" / "fewshot_ddbm_table.tex").write_text("\n".join(lines).rstrip()[:-2].rstrip() + "%\n")
    for k in (1, 3, 6):
        P = pooled(k)
        by = {}
        for sc, cam, s_ in P:
            by.setdefault((cam, sc), []).append((s_["DU"]["lpips"] - s_["U"]["lpips"], s_["DU"]["psnr"] - s_["U"]["psnr"]))
        dl = [sum(x[0] for x in v) / len(v) for v in by.values()]
        dp = [sum(x[1] for x in v) / len(v) for v in by.values()]
        mu = sum(dl) / len(dl)
        W = {1: "ONE", 3: "THREE", 6: "SIX"}[k]
        N[f"FSDUL{W}"] = f(-mu, 3)
        N[f"FSDULWIN{W}"] = sum(1 for x in dl if x < 0)
        N[f"FSDUP{W}"] = f(sum(dp) / len(dp), 2)
        N["FSSCENES"] = len(by)
    N["FSUPSNRONE"], N["FSUPSNRSIX"] = f(sum(x[2]["U"]["psnr"] for x in pooled(1)) / len(pooled(1)), 2), f(sum(x[2]["U"]["psnr"] for x in pooled(6)) / len(pooled(6)), 2)
    N["FSDPSNRSIX"] = f(sum(x[2]["D"]["psnr"] for x in pooled(6)) / len(pooled(6)), 2)
    N["FSZEROU"] = f(sum(x[2]["U"]["psnr"] for x in pooled(0)) / len(pooled(0)), 2)

# sweep of the DDBM adaptation recipe at K=3 (colab/run_fewshot2.sh): learning rate and steps
if fs:
    import glob as _glob
    cfgs = [("2e-5", 200, [r for row in fs if row["k"] == 3 for r in row["rows"]])]
    for fn in sorted(_glob.glob(str(RES / "fewshot_lr*.jsonl"))):
        lines = [json.loads(line) for line in open(fn)]
        cfgs.append((fn, lines[0]["steps"], [r for row in lines for r in row["rows"]]))
    vals = [sum(r["scores"]["D"]["psnr"] for r in rs) / len(rs) for _, _, rs in cfgs]
    if len(vals) > 1:
        N["SWEEPN"], N["SWEEPDMIN"], N["SWEEPDMAX"] = len(vals), f(min(vals), 1), f(max(vals), 1)
        uvals = [sum(r["scores"]["U"]["psnr"] for r in rs) / len(rs) for _, _, rs in cfgs]
        N["SWEEPUMIN"], N["SWEEPUMAX"] = f(min(uvals), 1), f(max(uvals), 1)

(REP / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(N.items())))
print(open(REP / "numbers.tex").read())
