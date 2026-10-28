#!/bin/bash
# Ping a Colab kernel every 5 minutes so background jobs are not reaped as idle.
#   colab/keepalive.sh <session> [b]
S=$1
if [ "$2" = b ]; then export HOME=~/.colab-account-b; fi
while true; do
  out=$(echo 'print("alive")' | colab exec -s "$S" 2>&1 | tail -1)
  echo "$(date +%H:%M) $S $out"
  case "$out" in *"No active"*|*"not found"*) exit 0;; esac
  sleep 300
done
