#!/bin/bash
# Fourth run, the full DDBM: the released recipe with its VP bridge (sigma in [1e-4, 1], beta_d 2,
# beta_min 0.1), EMA 0.9999 as in the released training script, trained for TRAIN_HOURS of wall
# clock, then the test set with the released 40-step sampler (and 20 steps), crops and timing.
set -x
cd /content
mark() { echo "STAGE $1 $(date +%H:%M)"; }
pip install -q rawpy exifread pytorch_msssim lpips
[ -d /content/Sony/short ] || { wget -q https://storage.googleapis.com/isl-datasets/SID/Sony2025.zip -O Sony.zip \
  && unzip -q Sony.zip -d /content && rm Sony.zip; }
mark download
cd /content/ldr-src
R=/content/runs; E=/content/results
mkdir -p $R $E
[ -f /content/cache/meta.json ] || python main.py --workers 4 prep-sid > /content/logs/prep.log 2>&1
mark prep
timeout ${TRAIN_HOURS:-4.5}h python main.py --workers 6 train-ddbm --official --bridge-type vp --ema 0.9999 --ema-warmup \
  --run $R/ddbm_vp --iters 400000 --val-n 20 > $R/ddbm_vp.out 2>&1
mark train
P="python main.py --workers 3"
$P eval --ckpt $R/ddbm_vp/best.pt --kind ddbm --ddbm-steps 40 --out $E/ddbm_vp_test_n40.json & a=$!
$P eval --ckpt $R/ddbm_vp/best.pt --kind ddbm --ddbm-steps 20 --out $E/ddbm_vp_test_n20.json & b=$!
wait $a $b
mark eval
python colab/figures.py crops 40
python colab/figures.py bench 20 40
mark figures
echo RERUNDONE
