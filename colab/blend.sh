#!/bin/bash
# Blend of the SID U-Net and the full DDBM (colab/blend.py): the weight is chosen on two pairs per
# validation scene, then every weight is scored on all test pairs. Only the val and test scenes are
# fetched. The checkpoints are uploaded separately into /content/runs.
set -x
cd /content
mark() { echo "STAGE $1 $(date +%H:%M)"; }
pip install -q rawpy exifread pytorch_msssim lpips remotezip
cd /content/ldr-src
[ -d /content/Sony/short ] || python colab/fetch_sid_test.py 12
mark download
mkdir -p /content/results /content/logs
[ -f /content/cache/sid/pairs.json ] || python main.py --workers 4 prep-sid > /content/logs/prep.log 2>&1
mark prep
until [ -f /content/runs/sid/best.pt ] && [ -f /content/runs/ddbm_vp/best.pt ]; do sleep 5; done
python colab/blend.py val 2
mark val
python colab/blend.py test
mark test
echo BLENDDONE
