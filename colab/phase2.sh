#!/bin/bash
# Phase 2: starts once the SID baseline and the DDBM have finished (their final checkpoints exist).
cd /content/ldr-src
R=/content/runs
until [ -f $R/sid/final.pt ] && [ -f $R/ddbm/last.pt ] && ! pgrep -f "train-ddb[m]" > /dev/null; do sleep 60; done
P="python main.py --workers 4"
$P train --model ddbm_feature --run $R/ddbm_feature --iters 40000 > $R/ddbm_feature.out 2>&1 &
$P train --model bridge --no-distill --run $R/bridge_nodistill --iters 40000 --ms-start 20000 > $R/bridge_nodistill.out 2>&1 &
wait
$P train --model bridge --no-film --run $R/bridge_nofilm --iters 40000 --ms-start 20000 > $R/bridge_nofilm.out 2>&1
echo PHASE2DONE
