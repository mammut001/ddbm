#!/bin/bash
# Thirteenth run (colab/diag_gain.py): oracle brightness and colour correction of the naive start on the ELD cameras, same ELD steps as run_crosscam.sh.
# ELD is a public file on Google Drive (12.3 GB, fetched with gdown, no mount needed). Started by colab/vps/driver.sh
# with PIPE=colab/run_crosscam.sh NEED_MOUNT=0; the four checkpoints are uploaded into /content/runs by the driver.
# Flags in /content/flags as in colab/run_unet2.sh.
set -x
F=/content/flags
mkdir -p $F /content/logs /content/runs /content/results
step() { rm -f $F/RUNNING_*; touch $F/RUNNING_$1; }
finish_step() { rm -f $F/RUNNING_$1; touch $F/DONE_$1; }
fail() { rm -f $F/RUNNING_*; touch $F/FAILED_$1; exit 1; }

pip install -q rawpy exifread pytorch_msssim lpips gdown || fail pip
cd /content/ldr-src

if [ ! -f $F/DONE_eld ]; then
  step eld
  if [ ! -d /content/ELDraw/ELD ]; then
    [ -f /content/ELD.zip ] || gdown --id 13Ge6-FY9RMPrvGiPvw7O4KS3LNfUXqEX -O /content/ELD.zip || fail gdown
    mkdir -p /content/ELDraw && unzip -q -o /content/ELD.zip -d /content/ELDraw || fail unzip
    rm -f /content/ELD.zip
  fi
  finish_step eld
fi

if [ ! -f $F/DONE_prep ]; then
  step prep
  [ -f /content/cache/eld/meta.json ] || python main.py --workers 6 prep-eld --eld-raw /content/ELDraw/ELD > /content/logs/prep_eld.log 2>&1 || fail prep
  python colab/fix_eld_iso.py >> /content/logs/prep_eld.log 2>&1 || fail iso
  finish_step prep
fi

if [ ! -f $F/DONE_cross ]; then
  step cross
  until [ -f /content/runs/sid/best.pt ] && [ -f /content/runs/ddbm_vp/best.pt ]; do sleep 5; done
  python colab/diag_gain.py >> /content/logs/cross.log 2>&1 || fail cross
  finish_step cross
fi
touch $F/ALL_DONE
echo RERUNDONE
