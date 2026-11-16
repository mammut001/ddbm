# Executed on the Colab VM after the machine was replaced: unpack code and last state, redo the data steps
# (the SID files and the cache are gone), keep finished training stages, and start the pipeline again.
import os
import subprocess

if subprocess.run("pgrep -f 'ldr-src/[c]olab/' > /dev/null", shell=True).returncode == 0:
    print("already running")
    raise SystemExit
pipe = open("/content/pipe.txt").read().strip() if os.path.exists("/content/pipe.txt") else "colab/run_unet2.sh"
cmd = ("mkdir -p /content/ldr-src /content/runs/sid && tar xzf /content/ldr-src.tgz -C /content/ldr-src && "
       "cp /content/sid_best.pt /content/runs/sid/best.pt && tar xzf /content/state.tgz -C /content && "
       "cd /content && rm -f flags/DONE_restore flags/DONE_prep flags/RUNNING_* flags/FAILED_* && "
       f"nohup bash /content/ldr-src/{pipe} >> /content/pipeline.out 2>&1")
subprocess.Popen(["bash", "-c", cmd], start_new_session=True)
print("started")
