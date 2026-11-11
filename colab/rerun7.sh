#!/bin/bash
# Seventh run, evaluation only (colab/multisample.py): the one-step readout of the two full DDBMs on all test
# pairs, then means of 4 samples of each DDBM on two pairs per test scene. The test scenes come from the copy
# in Google Drive (mount it with `colab drivemount`); the three checkpoints are uploaded into /content/runs,
# the sixth run's final checkpoint as ddbm_vp_unet/best.pt. A two-image smoke run goes first.
set -x
mark() { echo "STAGE $1 $(date +%H:%M)"; }
pip install -q rawpy exifread pytorch_msssim lpips
cd /content/ldr-src
until [ -d /content/Sony/short ] || [ -f /content/drive/MyDrive/SID/Sony_test.tar.zst ]; do sleep 5; done
[ -d /content/Sony/short ] || bash colab/restore_sid_from_drive.sh test
mark restore
R=/content/runs; E=/content/results
mkdir -p $E /content/logs
[ -f /content/cache/sid/pairs.json ] || python main.py --workers 6 prep-sid > /content/logs/prep.log 2>&1
mark prep
until [ -f $R/sid/best.pt ] && [ -f $R/ddbm_vp/best.pt ] && [ -f $R/ddbm_vp_unet/best.pt ]; do sleep 5; done
python colab/multisample.py mean --per-scene 1 --k 2 --limit 2 --out smoke_mean.jsonl || exit 1
mark smoke
python colab/multisample.py readout
mark readout
python colab/multisample.py mean --per-scene 2 --k 4
mark mean
echo RERUNDONE
