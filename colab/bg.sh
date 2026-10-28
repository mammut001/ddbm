#!/bin/bash
# Run a shell command in the background on a Colab VM, logging to /content/logs/<name>.log
#   colab/bg.sh <session> <a|b> <name> "<command>"
S=$1; ACC=$2; NAME=$3; CMD=$4
if [ "$ACC" = b ]; then export HOME=~/.colab-account-b; fi
python3 - "$NAME" "$CMD" <<'PY' | colab exec -s "$S"
import sys, json
name, cmd = sys.argv[1], sys.argv[2]
print(f"import subprocess, os; os.makedirs('/content/logs', exist_ok=True); "
      f"subprocess.Popen({json.dumps('nohup bash -c ' + json.dumps(cmd) + ' > /content/logs/' + name + '.log 2>&1 &')}, shell=True); print('launched {name}')")
PY
