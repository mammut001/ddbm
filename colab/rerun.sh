#!/bin/bash
# Rerun after the first VM was reclaimed: SID U-Net (for qualitative crops and timing) and the
# official-recipe DDBM with sigma_max = 1, then their evaluations and the figures.
set -x
cd /content
mark() { echo "STAGE $1 $(date +%H:%M)"; }
pip install -q rawpy exifread pytorch_msssim
[ -d /content/Sony/short ] || { wget -q https://storage.googleapis.com/isl-datasets/SID/Sony2025.zip -O Sony.zip \
  && unzip -q Sony.zip -d /content && rm Sony.zip; }
ls /content/Sony/short | wc -l; ls /content/Sony/long | wc -l
mark download
cd /content/ldr-src
P="python main.py --workers 4"
R=/content/runs; E=/content/results
mkdir -p $R $E
[ -f /content/cache/meta.json ] || $P prep-sid > /content/logs/prep.log 2>&1
mark prep
$P train --model sid --run $R/sid --iters 40000 --ms-start 20000 > $R/sid.out 2>&1 &
$P train-ddbm --official --bridge-sigma-max 1 --run $R/ddbm_official_s1 --iters 30000 --val-n 20 > $R/ddbm_official_s1.out 2>&1 &
wait
mark train
P="python main.py --workers 3"
$P eval --ckpt $R/sid/best.pt --out $E/sid_test.json &
$P eval --ckpt $R/ddbm_official_s1/best.pt --kind ddbm --ddbm-steps 5 --out $E/ddbm_official_s1_test_n5.json &
$P eval --ckpt $R/ddbm_official_s1/best.pt --kind ddbm --ddbm-steps 10 --out $E/ddbm_official_s1_test_n10.json &
wait
mark eval
python colab/figures.py curves
python colab/figures.py crops
python colab/figures.py bench
mark figures
echo RERUNDONE
