#!/bin/bash
# Fifteenth-to-sixteenth run: the SID-trained U-Net (and the frozen DDBM on its output) on a new Sony sensor, the A7R III of
# the SIED dataset, zero-shot and after fine-tuning on K = 10, 30, 100 images (colab/sied_prep.py, colab/custom_camera.py).
# The two zip files come from a temporary HTTP server on the VPS (host and port in /content/sied_host.txt, uploaded by
# colab/vps/driver.sh through EXTRA), so no Drive mount is needed.
set -x
F=/content/flags
mkdir -p $F /content/logs /content/results
step() { rm -f $F/RUNNING_*; touch $F/RUNNING_$1; }
finish_step() { rm -f $F/RUNNING_$1; touch $F/DONE_$1; }
fail() { rm -f $F/RUNNING_*; touch $F/FAILED_$1; exit 1; }
pip install -q rawpy exifread pytorch_msssim lpips || fail pip
cd /content/ldr-src
until [ -f /content/sied_host.txt ]; do sleep 5; done
HOST=$(cat /content/sied_host.txt)
if [ ! -f $F/DONE_fetch ]; then
  step fetch
  for pair in "Sony_gt.zip:3540159732" "Sony_0.001.zip:11489809753"; do
    f=${pair%%:*}; size=${pair##*:}
    curl -fL -C - --retry 8 --retry-delay 5 "http://$HOST/$f" -o /content/$f >> /content/logs/fetch.log 2>&1 || fail curl
    [ "$(stat -c %s /content/$f)" = "$size" ] || fail size
  done
  finish_step fetch
fi
if [ ! -f $F/DONE_prep ]; then
  step prep
  python colab/sied_prep.py --zips /content --level 0.001 --n-train 100 >> /content/logs/custom.log 2>&1 || fail prep
  rm -f /content/Sony_gt.zip /content/Sony_0.001.zip
  finish_step prep
fi
if [ ! -f $F/DONE_adapt ]; then
  step adapt
  until [ -f /content/runs/sid/best.pt ] && [ -f /content/runs/ddbm_vp_unet/best.pt ]; do sleep 5; done
  python colab/custom_camera.py adapt --name sied --train-pairs pairs_train.json --test-pairs pairs_test.json \
    --ks 0 10 30 100 --steps 200 --ddbm-ckpt /content/runs/ddbm_vp_unet/best.pt >> /content/logs/custom.log 2>&1 || fail adapt
  cp /content/runs/custom_sied/result_ks.json /content/results/custom_sied.json
  finish_step adapt
fi
touch $F/ALL_DONE
echo RERUNDONE
