#!/bin/bash
# Third run: SID U-Net (seed 0, for LPIPS), the simplified DDBM and the official-recipe DDBM with
# sigma_max = 1 trained to 60k, then test evaluation with LPIPS and the figures.
set -x
cd /content
mark() { echo "STAGE $1 $(date +%H:%M)"; }
pip install -q rawpy exifread pytorch_msssim lpips
[ -d /content/Sony/short ] || { wget -q https://storage.googleapis.com/isl-datasets/SID/Sony2025.zip -O Sony.zip \
  && unzip -q Sony.zip -d /content && rm Sony.zip; }
mark download
cd /content/ldr-src
P="python main.py --workers 3"
R=/content/runs; E=/content/results
mkdir -p $R $E
[ -f /content/cache/meta.json ] || $P prep-sid > /content/logs/prep.log 2>&1
mark prep
$P train --model sid --run $R/sid --iters 40000 --ms-start 20000 > $R/sid.out 2>&1 & a=$!
$P train-ddbm --run $R/ddbm --iters 60000 --val-n 20 > $R/ddbm.out 2>&1 & b=$!
$P train-ddbm --official --bridge-sigma-max 1 --run $R/ddbm_official_s1 --iters 60000 --val-n 20 > $R/ddbm_official_s1.out 2>&1 & c=$!
wait $a; mark train_sid
$P eval --ckpt $R/sid/best.pt --out $E/sid_test.json; mark eval_sid
wait $b $c; mark train_ddbm
$P eval --ckpt $R/ddbm/best.pt --kind ddbm --ddbm-steps 5 --out $E/ddbm_test_n5.json & a=$!
$P eval --ckpt $R/ddbm_official_s1/best.pt --kind ddbm --ddbm-steps 10 --out $E/ddbm_official_s1_test_n10.json & b=$!
$P eval --ckpt $R/ddbm_official_s1/best.pt --kind ddbm --ddbm-steps 20 --out $E/ddbm_official_s1_test_n20.json & c=$!
wait $a $b $c
$P eval --ckpt $R/ddbm/best.pt --kind ddbm --ddbm-steps 10 --out $E/ddbm_test_n10.json & a=$!
$P eval --ckpt $R/ddbm_official_s1/best.pt --kind ddbm --ddbm-steps 5 --out $E/ddbm_official_s1_test_n5.json & b=$!
wait $a $b
mark eval_ddbm
python colab/figures.py curves
python colab/figures.py crops
mark figures
echo RERUNDONE
