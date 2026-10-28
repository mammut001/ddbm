#!/bin/bash
# Full-resolution SID test evaluation of the models from phase 1 (598 images each).
cd /content/ldr-src
R=/content/runs; E=/content/results
mkdir -p $E
until [ -f $R/sid/final.pt ] && [ -f $R/ddbm/last.pt ] && ! pgrep -f "train-ddb[m]" > /dev/null; do sleep 60; done
P="python main.py --workers 2"
$P eval --ckpt $R/sid/best.pt --out $E/sid_test.json
for n in 10 5 20; do $P eval --ckpt $R/ddbm/best.pt --kind ddbm --ddbm-steps $n --out $E/ddbm_test_n$n.json; done
$P eval --ckpt $R/ddbm/best.pt --kind ddbm --ddbm-steps 10 --eta 0 --out $E/ddbm_test_n10_eta0.json
echo EVAL1DONE
