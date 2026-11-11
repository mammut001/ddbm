#!/bin/bash
# Sixth run, after training: tests the final checkpoint (302k). In this run the 10-step validation PSNR is a
# poor guide for the released 40-step sampler: it peaked at 10k, fell until 42k and only then recovered, while
# with 40 steps the 54k model was already better than the 10k one. So best.pt is replaced by last.pt.
# Waits for the training started by rerun6.sh.
set -x
mark() { echo "STAGE $1 $(date +%H:%M)"; }
cd /content/ldr-src
R=/content/runs; E=/content/results
while pgrep -f "train-ddbm.*ddbm_vp_unet" > /dev/null; do sleep 30; done
mark train
cp $R/ddbm_vp_unet/best.pt $R/ddbm_vp_unet/best_val10.pt
cp $R/ddbm_vp_unet/last.pt $R/ddbm_vp_unet/best.pt
P="python main.py --workers 3 eval --ckpt $R/ddbm_vp_unet/best.pt --kind ddbm"
$P --ddbm-steps 40 --out $E/ddbm_vp_unet_test_n40.json
mark eval40
python colab/figures.py crops 40
python colab/figures.py bench 40
mark figures
$P --ddbm-steps 10 --out $E/ddbm_vp_unet_test_n10.json
mark eval10
echo RERUNDONE
