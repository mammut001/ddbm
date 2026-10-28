#!/bin/bash
# Ping a Colab kernel every 5 minutes so background jobs are not reaped as idle.
# Each ping is killed after 90 s so a hung connection cannot stall the loop.
#   colab/keepalive.sh <session> [b]
S=$1
if [ "$2" = b ]; then export HOME=~/.colab-account-b; fi
while true; do
  echo 'print("alive")' | colab exec -s "$S" > /tmp/ka-$S.out 2>&1 &
  pid=$!
  ( sleep 90; kill $pid 2>/dev/null ) &
  wait $pid
  out=$(tail -1 /tmp/ka-$S.out)
  echo "$(date +%H:%M) $S $out"
  case "$out" in *"No active"*|*"not found"*) exit 0;; esac
  sleep 300
done
