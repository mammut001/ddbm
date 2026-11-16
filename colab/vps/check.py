# Executed on the Colab VM with `colab exec -f`: one status snapshot (also keeps the session active).
import json
import os

fl = sorted(os.listdir("/content/flags")) if os.path.isdir("/content/flags") else []
print("FLAGS", " ".join(fl))
import subprocess
running = subprocess.run("pgrep -f 'ldr-src/[c]olab/' > /dev/null", shell=True).returncode == 0
print("PIPE", "running" if running else "none")
print("MOUNT", "yes" if os.path.isdir("/content/drive/MyDrive/SID") else "no")
for f in ("/content/logs/prep.log", "/content/runs/sid_s1.out", "/content/logs/ens.log", "/content/logs/prep_eld.log", "/content/logs/cross.log"):
    if os.path.exists(f):
        t = open(f, errors="ignore").read()
        if "Traceback" in t:
            print("ERR", f, t[t.rfind("Traceback"):][-700:].replace("\n", " | "))
f = "/content/runs/sid_s1/log.jsonl"
if os.path.exists(f):
    rows = []
    for x in open(f):
        try:
            rows.append(json.loads(x))
        except json.JSONDecodeError:
            pass
    tr = [r for r in rows if "loss" in r]
    va = [r for r in rows if isinstance(r.get("val"), dict)]
    if tr:
        print("TRAIN", json.dumps({k: tr[-1].get(k) for k in ("step", "loss", "it_s")}))
    if va:
        print("VAL", json.dumps({"step": va[-1].get("step"), "psnr": va[-1]["val"].get("psnr")}))
f = "/content/results/ens_unet.jsonl"
if os.path.exists(f):
    print("ENS", sum(1 for _ in open(f)), "/ 598")
f = "/content/results/crosscam.jsonl"
if os.path.exists(f):
    print("CROSS", sum(1 for _ in open(f)), "/ 72")
print("GPU", os.popen("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null").read().strip())
