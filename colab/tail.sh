#!/bin/bash
#   colab/tail.sh <session> <a|b> "<shell command>"   (e.g. "tail -5 /content/logs/x.log")
S=$1; ACC=$2; CMD=$3
if [ "$ACC" = b ]; then export HOME=~/.colab-account-b; fi
python3 -c "import json,sys;print('import subprocess;print(subprocess.run('+json.dumps(sys.argv[1])+',shell=True,capture_output=True,text=True).stdout[-6000:])')" "$CMD" | colab exec -s "$S"
