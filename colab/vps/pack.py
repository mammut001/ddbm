# Executed on the Colab VM: pack the state worth keeping (checkpoints, logs, flags, results) into state.tgz.
# Written under another name first and renamed, so a cut-off pack never replaces a good one.
import os
import subprocess

os.chdir("/content")
pipe = open("/content/pipe.txt").read() if os.path.exists("/content/pipe.txt") else ""
import glob
names = ["flags", *sorted(glob.glob("results/*.jsonl")), "logs"]
if "crosscam" not in pipe and "diag" not in pipe and "fewshot" not in pipe:  # the cross-camera run only reads its checkpoints, so there is nothing to save from runs/
    names += ["runs/sid_s1", "runs/sid_s1.out"]
parts = [p for p in names if os.path.exists(p)]
if not parts:
    print("nothing to pack")
    raise SystemExit
subprocess.run(["tar", "czf", "/content/state.part.tgz", *parts], check=True)
os.replace("/content/state.part.tgz", "/content/state.tgz")
print("packed", os.path.getsize("/content/state.tgz"))
