#!/bin/bash
# Runs on the GreenCloud VPS in tmux, so the Mac can sleep. Opens a Colab GPU session, uploads the code and the
# first U-Net's checkpoint, starts colab/run_unet2.sh there, checks it every POLL seconds, pulls the state
# (checkpoints, logs, results) to $D/state.tgz every PULL_EVERY checks, and stops the Colab session when the
# pipeline is done, has failed, has been idle, has no Drive mount, or reaches the deadline (HARD_H hours).
# If the machine is lost it opens a new one and resumes from the last state, but the Drive mount needs the
# user's authorization again (`colab drivemount -s $SES`), so without it the session is stopped after
# MOUNT_WAIT_MIN minutes instead of idling on the GPU. Status: $D/status.txt, $D/driver.log.
# Dry run on a CPU session: GPU_LIST=cpu NEED_MOUNT=0 PIPE=colab/vps/dry_run.sh POLL=60 PULL_EVERY=3 SES=ddbmtest
export PATH=$HOME/.local/bin:$PATH
D=${D:-/root/ddbm-sup}; cd $D || exit 1
SES=${SES:-ddbmu}; HARD_H=${HARD_H:-8}; POLL=${POLL:-300}; PULL_EVERY=${PULL_EVERY:-6}
MOUNT_WAIT_MIN=${MOUNT_WAIT_MIN:-20}; NEED_MOUNT=${NEED_MOUNT:-1}; EXTRA=${EXTRA:-}
GPU_LIST=${GPU_LIST:-"A100 A100 A100 A100"}; PIPE=${PIPE:-colab/run_unet2.sh}; SIDBEST=${SIDBEST:-$D/sid_best.pt}
log() { echo "$(date -u '+%m-%d %H:%M') $*" >> $D/driver.log; }
cx() { timeout 240 colab exec -s $SES -f "$@" 2>&1 | grep -v '^\[colab\]'; }
up() { for t in 1 2 3; do timeout 900 colab upload -s $SES "$1" "$2" 2>&1 | grep -qi uploaded && return 0; log "upload of $(basename $1) failed (try $t)"; sleep 20; done; return 1; }
usage() { colab usage 2>&1 | grep -E 'balance|assignments' | tr '\n' ' ' >> $D/driver.log; echo >> $D/driver.log; }
pull() {  # the pack step must report success, otherwise the old state.tgz on the VM would be downloaded and called fresh
  local out
  out=$(timeout 700 colab exec --timeout 600 -s $SES -f $D/pack.py 2>&1 | grep -v "^\[colab\]" | tail -1)
  echo "$out" >> $D/driver.log
  case "$out" in packed*|"nothing to pack"*) ;; *) log "state pull FAILED: pack step did not report success"; return 1;; esac
  timeout 900 colab download -s $SES /content/state.tgz $D/state.tmp.tgz >/dev/null 2>&1 && gzip -t $D/state.tmp.tgz \
    && mv $D/state.tmp.tgz $D/state.tgz && log "state pulled" && return 0
  log "state pull FAILED (kept the previous one)"; return 1
}
finish() {  # $1 = reason; the last state is pulled first unless the reason says there is nothing to pull
  if [ "$2" != nopull ]; then
    ok=0; for try in 1 2 3 4 5 6; do pull && { ok=1; break; }; sleep 120; done
    # a finished run is only stopped once its results are safely here; otherwise keep the VM and ask for help
    if [ $ok = 0 ] && [ "$1" = "all done" ]; then
      log "NOT STOPPED: the final state could not be pulled, the VM is kept"; echo "NEEDS_PULL all done but the final pull failed; session $SES still running" > $D/status.txt; exit 1
    fi
  fi
  timeout 180 colab stop -s $SES >/dev/null 2>&1; log "STOPPED: $1"; usage
  [ "$1" = "all done" ] && [ -f $D/state.tgz ] && mkdir -p $D/final && tar xzf $D/state.tgz -C $D/final
  echo "STOPPED $1" > $D/status.txt; exit 0
}
open_session() {
  for gpu in $GPU_LIST; do
    flag="--gpu $gpu"; [ "$gpu" = cpu ] && flag=""
    if timeout 300 colab new -s $SES $flag > $D/new.log 2>&1 && ! grep -q Error $D/new.log; then
      log "session on $gpu"; echo $gpu > $D/gpu.txt; t_session=$(date +%s); return 0; fi
    log "no $gpu: $(grep -o 'Service Unavailable\|Error[^:]*' $D/new.log | head -1)"; sleep 90
  done; return 1
}
launched() { sleep 20; cx $D/check.py | grep -q -E '^(PIPE running|FLAGS .+)'; }
upbig() {  # $1 local file, $2 path on the VM; 50 MB pieces, since the upload API refuses big files
  rm -rf $D/xparts; mkdir -p $D/xparts; split -b 50m "$1" $D/xparts/p-
  for f in $D/xparts/*; do up $f /content/$(basename "$2").$(basename $f) || return 1; done
  printf 'import glob, os\nd = "%s"\nos.makedirs(os.path.dirname(d), exist_ok=True)\nps = sorted(glob.glob("/content/%s.p-*"))\nwith open(d + ".tmp", "wb") as o:\n    for p in ps:\n        o.write(open(p, "rb").read()); os.remove(p)\nos.replace(d + ".tmp", d)\nprint("joined", len(ps), os.path.getsize(d))\n' "$2" "$(basename "$2")" > $D/xjoin.py
  for t in 1 2 3; do cx $D/xjoin.py | tail -1 | grep -q joined && return 0; log "join of $(basename $2) failed ($t)"; sleep 15; done; return 1
}
start() {
  echo $PIPE > $D/pipe.txt
  up $D/ldr-src.tgz /content/ldr-src.tgz && up $SIDBEST /content/sid_best.pt && up $D/pipe.txt /content/pipe.txt || return 1
  for pair in $EXTRA; do upbig "${pair%%:*}" "${pair##*:}" || return 1; done  # EXTRA="local:/content/path ..."
  if [ -f $D/state.tgz ]; then  # the upload API refuses big files, so the state goes up in 50 MB pieces
    rm -rf $D/parts; mkdir -p $D/parts; split -b 50m $D/state.tgz $D/parts/state.tgz.part-
    for f in $D/parts/*; do up $f /content/$(basename $f) || return 1; done
    ok=0; for t in 1 2 3; do cx $D/join.py | tail -1 | grep -q joined && { ok=1; break; }; log "join attempt $t failed"; sleep 15; done
    [ $ok = 1 ] || return 1
  fi
  for try in 1 2 3; do  # launch.py and resume.py do nothing if the pipeline already runs, so a retry never starts two
    if [ -f $D/state.tgz ]; then cx $D/resume.py | tail -1 >> $D/driver.log; else cx $D/launch.py | tail -1 >> $D/driver.log; fi
    launched && return 0
    log "launch attempt $try: the pipeline did not show up"
  done
  return 1
}
deadline=$(( $(date +%s) + HARD_H * 3600 )); fails=0; idle=0; n=0; restarts=0
log "driver starts: SES=$SES gpu=[$GPU_LIST] pipe=$PIPE deadline=${HARD_H}h"; usage
open_session || { log "no GPU at all"; echo "NO GPU" > $D/status.txt; exit 1; }
start || finish "start failed" nopull
while :; do
  sleep $POLL; n=$((n + 1))
  out=$(cx $D/check.py)
  if ! echo "$out" | grep -q '^FLAGS'; then
    fails=$((fails + 1)); log "check failed ($fails)"
    if [ $fails -ge 3 ]; then
      [ $restarts -ge 3 ] && { log "too many restarts"; timeout 180 colab stop -s $SES; echo "GAVE UP" > $D/status.txt; exit 1; }
      restarts=$((restarts + 1)); timeout 180 colab stop -s $SES >/dev/null 2>&1; log "VM lost; restart $restarts from the last state"
      echo "NEEDS_MOUNT after a lost VM: run colab drivemount -s $SES within ${MOUNT_WAIT_MIN} min" > $D/status.txt
      open_session && { start || finish "start failed after a lost VM" nopull; }; fails=0
    fi; continue
  fi
  fails=0; { date -u '+%m-%d %H:%M'; cat $D/gpu.txt; echo "$out"; } > $D/status.txt
  echo "$out" | grep -E '^(TRAIN|VAL|ENS)' | tr '\n' ' ' | sed 's/$/\n/' >> $D/progress.log
  flags=$(echo "$out" | grep '^FLAGS')
  echo "$flags" | grep -q ALL_DONE && finish "all done"
  echo "$flags" | grep -q FAILED_ && finish "a stage failed: $flags"
  if [ "$NEED_MOUNT" = 1 ] && ! echo "$out" | grep -q '^MOUNT yes' && [ $(( $(date +%s) - t_session )) -gt $((MOUNT_WAIT_MIN * 60)) ]; then
    finish "no Drive mount within ${MOUNT_WAIT_MIN} min" nopull
  fi
  if echo "$flags" | grep -q RUNNING_ || echo "$out" | grep -q "^PIPE running"; then idle=0; else idle=$((idle + 1)); [ $idle -ge 4 ] && finish "idle"; fi
  [ $(date +%s) -gt $deadline ] && finish "deadline"
  [ $((n % PULL_EVERY)) -eq 0 ] && pull
done
