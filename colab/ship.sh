#!/bin/bash
# Upload the code to a Colab session and unpack it in /content/ldr-src.
#   colab/ship.sh <session> [b]     (b = second account, HOME=~/.colab-account-b)
set -e
S=$1
cd "$(dirname "$0")/.."
tar czf /tmp/ldr-src.tgz main.py ldr colab
if [ "$2" = b ]; then export HOME=~/.colab-account-b; fi
colab upload -s "$S" /tmp/ldr-src.tgz /content/ldr-src.tgz
echo 'import subprocess; print(subprocess.run("mkdir -p /content/ldr-src && tar xzf /content/ldr-src.tgz -C /content/ldr-src && ls /content/ldr-src", shell=True, capture_output=True, text=True).stdout)' | colab exec -s "$S"
