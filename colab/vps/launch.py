# Executed on the Colab VM with `colab exec -f`: unpack the code and start the pipeline in the background.
import os
import subprocess

if subprocess.run("pgrep -f 'ldr-src/[c]olab/' > /dev/null", shell=True).returncode == 0:
    print("already running")
    raise SystemExit
pipe = open("/content/pipe.txt").read().strip() if os.path.exists("/content/pipe.txt") else "colab/run_unet2.sh"
cmd = ("mkdir -p /content/ldr-src /content/flags /content/runs/sid && tar xzf /content/ldr-src.tgz -C /content/ldr-src && "
       "cp /content/sid_best.pt /content/runs/sid/best.pt && cd /content && "
       f"nohup bash /content/ldr-src/{pipe} > /content/pipeline.out 2>&1")
subprocess.Popen(["bash", "-c", cmd], start_new_session=True)
print("started")
