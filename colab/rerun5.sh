#!/bin/bash
# Test evaluation of the full DDBM, again: the fourth run's VM was reclaimed during its evaluation, with
# the checkpoint already copied off. Only the test scenes are fetched. The released 40 steps come first,
# then crops and timing, then 20 steps. Finished rows stay in results/*.partial.jsonl, so a rerun continues.
set -x
cd /content
mark() { echo "STAGE $1 $(date +%H:%M)"; }
pip install -q rawpy exifread pytorch_msssim lpips remotezip
cd /content/ldr-src
[ -d /content/Sony/short ] || python colab/fetch_sid_test.py
mark download
R=/content/runs; E=/content/results
mkdir -p $E
[ -f /content/cache/sid/pairs.json ] || python main.py --workers 4 prep-sid > /content/logs/prep.log 2>&1
mark prep
until [ -f $R/ddbm_vp/best.pt ]; do sleep 5; done
P="python main.py --workers 3 eval --ckpt $R/ddbm_vp/best.pt --kind ddbm"
$P --ddbm-steps 40 --out $E/ddbm_vp_test_n40.json
mark eval40
python colab/figures.py crops 40
python colab/figures.py bench 20 40
mark figures
$P --ddbm-steps 20 --out $E/ddbm_vp_test_n20.json
mark eval20
echo RERUNDONE
