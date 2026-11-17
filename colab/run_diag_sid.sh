#!/bin/bash
# Fourteenth run (evaluation only): brightness and colour of the naive rendering on the SID Sony test pairs
# (colab/diag_naive_sid.py). The test scenes come from the copy in Google Drive (needs `colab drivemount`, started by
# colab/vps/driver.sh with the default NEED_MOUNT=1). Flags in /content/flags as in colab/run_unet2.sh.
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
  [ -d /content/Sony/short ] || bash colab/restore_sid_from_drive.sh test || fail restore
  finish_step restore
fi
if [ ! -f $F/DONE_prep ]; then
  step prep
  [ -f /content/cache/sid/pairs.json ] || python main.py --workers 6 prep-sid > /content/logs/prep.log 2>&1 || fail prep
  finish_step prep
fi
if [ ! -f $F/DONE_diag ]; then
  step diag
  until [ -f /content/runs/sid/best.pt ]; do sleep 5; done
  python colab/diag_naive_sid.py >> /content/logs/cross.log 2>&1 || fail diag
  finish_step diag
fi
touch $F/ALL_DONE
echo RERUNDONE
