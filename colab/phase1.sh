#!/bin/bash
# Phase 1 on one A100 80GB: the three SID models that everything else starts from, in parallel.
set -x
cd /content/ldr-src
P="python main.py --workers 4"
R=/content/runs
mkdir -p $R
$P train --model sid    --run $R/sid    --iters ${ITERS:-40000} --ms-start 20000 > $R/sid.out 2>&1 &
$P train --model bridge --run $R/bridge --iters ${ITERS:-40000} --ms-start 20000 > $R/bridge.out 2>&1 &
$P train-ddbm           --run $R/ddbm   --iters ${DITERS:-60000} --val-n 20     > $R/ddbm.out 2>&1 &
wait
echo PHASE1DONE
