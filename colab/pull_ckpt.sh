#!/bin/bash
# Every 30 minutes copy a training checkpoint from the VM to the same path here, so that a lost VM can be
# replaced and the run resumed from it (train-ddbm resumes from last.pt in its run directory).
#   SESSION=ldra colab/pull_ckpt.sh runs/ddbm_vp_unet/last.pt
# With SESSION set (an account-A session), it exits once that session is gone.
cd "$(dirname "$0")/.."
F=$1; CFG=${CFG:-colab/ssh_config_a}; HOST=${HOST:-colaba}
mkdir -p "$(dirname "$F")"
while true; do
  if [ -n "$SESSION" ] && ! colab status 2>/dev/null | grep -q "^\[$SESSION\]"; then
    echo "$(date +%H:%M) session $SESSION is gone"; exit 1
  fi
  ssh -F "$CFG" "$HOST" "[ -f /content/$F ] && cp /content/$F /content/ckpt-copy.pt" 2>/dev/null \
    && scp -q -F "$CFG" "$HOST:/content/ckpt-copy.pt" "$F.tmp" 2>/dev/null && mv "$F.tmp" "$F" \
    && echo "$(date +%H:%M) copied $F"
  sleep 1800
done
