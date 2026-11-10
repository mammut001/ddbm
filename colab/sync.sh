#!/bin/bash
# Every 10 minutes copy results and training logs (no checkpoints) from the VM into results/;
# exits once the file WATCH contains UNTIL (default: RERUNDONE in the run log).
#   CFG=colab/ssh_config_a HOST=colaba LOG=rerun4 WATCH=results/runs/ddbm_vp/log.jsonl UNTIL='"step": 4000, "val"' colab/sync.sh
# With SESSION set (an account-A session), it exits once that session is gone, because an ssh to a
# missing session silently creates a new CPU runtime.
cd "$(dirname "$0")/.."
CFG=${CFG:-colab/ssh_config_b}; HOST=${HOST:-colabb}; LOG=${LOG:-rerun3}
WATCH=${WATCH:-results/logs/$LOG.log}; UNTIL=${UNTIL:-RERUNDONE}
mkdir -p results
while true; do
  if [ -n "$SESSION" ] && ! colab status 2>/dev/null | grep -q "^\[$SESSION\]"; then
    echo "$(date +%H:%M) session $SESSION is gone"; exit 1
  fi
  ssh -F "$CFG" "$HOST" "cd /content && tar czf - results \$(ls -d runs/*/log.jsonl) logs 2>/dev/null" 2>/dev/null > /tmp/ldr-sync.tgz \
    && [ -s /tmp/ldr-sync.tgz ] && tar xzf /tmp/ldr-sync.tgz -C results \
    && echo "$(date +%H:%M) synced: $(grep -h STAGE results/logs/$LOG.log 2>/dev/null | tail -1)"
  grep -qF "$UNTIL" "$WATCH" 2>/dev/null && exit 0
  sleep 600
done
