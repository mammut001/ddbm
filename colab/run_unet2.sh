#!/bin/bash
# Eighth run: a second SID U-Net (seed 1, same recipe as the first: 40k iterations, MS-SSIM from 20k) as the
# ensemble control, then colab/unet_ensemble.py on all test pairs. Started and watched by colab/vps/driver.sh,
# which uploads the code to /content/ldr-src and the first U-Net's checkpoint to /content/runs/sid/best.pt.
# Progress is kept as flag files in /content/flags (DONE_*, RUNNING_*, FAILED_*, ALL_DONE). The SID scenes come
# from the copy in Google Drive: `colab drivemount` needs one browser authorization per VM, and this script
# waits for it. After a lost machine the driver unpacks the last state (runs/sid_s1/last.pt) and train resumes.
set -x
F=/content/flags
mkdir -p $F /content/logs /content/runs /content/results
step() { rm -f $F/RUNNING_*; touch $F/RUNNING_$1; }
finish_step() { rm -f $F/RUNNING_$1; touch $F/DONE_$1; }
fail() { rm -f $F/RUNNING_*; touch $F/FAILED_$1; exit 1; }

pip install -q rawpy exifread pytorch_msssim lpips || fail pip
cd /content/ldr-src

if [ ! -f $F/DONE_restore ]; then
  step restore
  until [ -f /content/drive/MyDrive/SID/Sony_test.tar.zst ]; do sleep 5; done
  [ -d /content/Sony/short ] || bash colab/restore_sid_from_drive.sh || fail restore
  finish_step restore
fi

if [ ! -f $F/DONE_prep ]; then
  step prep
  [ -f /content/cache/sid/pairs.json ] || python main.py --workers 6 prep-sid > /content/logs/prep.log 2>&1 || fail prep
  finish_step prep
fi

if [ ! -f $F/DONE_train ]; then
  step train
  python main.py --seed 1 --workers 6 train --model sid --run /content/runs/sid_s1 --iters 40000 --ms-start 20000 \
    >> /content/runs/sid_s1.out 2>&1 || fail train
  finish_step train
fi

if [ ! -f $F/DONE_ens ]; then
  step ens
  until [ -f /content/runs/sid/best.pt ]; do sleep 5; done
  python colab/unet_ensemble.py >> /content/logs/ens.log 2>&1 || fail ens
  finish_step ens
fi

touch $F/ALL_DONE
echo RERUNDONE
