#!/bin/bash
# Second half of rerun.sh after restarting the SID U-Net with seed 1 (seed 0 repeated the first run).
set -x
mark() { echo "STAGE $1 $(date +%H:%M)"; }
cd /content/ldr-src
R=/content/runs; E=/content/results
rm -rf $R/sid
python main.py --workers 4 --seed 1 train --model sid --run $R/sid --iters 40000 --ms-start 20000 > $R/sid.out 2>&1 &
until grep -q '"step": 30000, "val"' $R/ddbm_official_s1/log.jsonl 2>/dev/null; do sleep 60; done
mark train_ddbm
P="python main.py --workers 3"
$P eval --ckpt $R/ddbm_official_s1/best.pt --kind ddbm --ddbm-steps 5 --out $E/ddbm_official_s1_test_n5.json & a=$!
$P eval --ckpt $R/ddbm_official_s1/best.pt --kind ddbm --ddbm-steps 10 --out $E/ddbm_official_s1_test_n10.json & b=$!
wait $a $b
mark eval_ddbm
until [ -f $R/sid/final.pt ]; do sleep 60; done
mark train_sid
$P eval --ckpt $R/sid/best.pt --out $E/sid_test.json
mark eval_sid
python colab/figures.py curves
python colab/figures.py crops
python colab/figures.py bench
mark figures
echo RERUNDONE
