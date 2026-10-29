#!/bin/bash
# Every 20 minutes copy results and training logs (no checkpoints) from the VM into results/.
cd "$(dirname "$0")/.."
mkdir -p results
while true; do
  ssh -F colab/ssh_config_b colabb "cd /content && tar czf - results \$(ls -d runs/*/log.jsonl) logs 2>/dev/null" 2>/dev/null > /tmp/ldr-sync.tgz \
    && [ -s /tmp/ldr-sync.tgz ] && tar xzf /tmp/ldr-sync.tgz -C results && echo "$(date +%H:%M) synced"
  sleep 1200
done
