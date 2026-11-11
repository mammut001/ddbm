#!/bin/bash
# Sixth run: the full DDBM started from the SID U-Net output (x_T = U-Net, --unet-start) instead of the naive
# rendering, everything else as in the fourth run and for the same 302k iterations. A short smoke run of
# training and evaluation goes first. After training, colab/rerun6_eval.sh tests the final checkpoint.
# The SID U-Net checkpoint is uploaded separately into /content/runs/sid.
set -x
cd /content
mark() { echo "STAGE $1 $(date +%H:%M)"; }
pip install -q rawpy exifread pytorch_msssim lpips remotezip
cd /content/ldr-src
[ -d /content/Sony/short ] || python colab/fetch_sid_test.py 012
mark download
R=/content/runs; E=/content/results
mkdir -p $R $E /content/logs
[ -f /content/cache/sid/pairs.json ] || python main.py --workers 6 prep-sid > /content/logs/prep.log 2>&1
mark prep
until [ -f $R/sid/best.pt ]; do sleep 5; done
T="python main.py --workers 6 train-ddbm --official --bridge-type vp --ema 0.9999 --ema-warmup --unet-start $R/sid/best.pt"
$T --run $R/smoke --iters 200 --val-every 100 --val-n 2 > $R/smoke.out 2>&1 || { tail -20 $R/smoke.out; exit 1; }
python main.py eval --ckpt $R/smoke/best.pt --kind ddbm --ddbm-steps 4 --limit 2 --out $E/smoke.json || exit 1
mark smoke
$T --run $R/ddbm_vp_unet --iters 302000 --val-n 20 > $R/ddbm_vp_unet.out 2>&1
exec bash colab/rerun6_eval.sh
