#!/bin/bash
# Phase 3, unattended: everything that depends on the trained Bridge U-Net, then the
# evaluations of the phase-2 models and of the official DDBM as each one finishes.
cd /content/ldr-src
R=/content/runs; E=/content/results
mkdir -p $E
P="python main.py --workers 3"
wait_for() { until [ -f "$1" ]; do sleep 120; done; }

wait_for $R/bridge/final.pt
$P eval --ckpt $R/bridge/best.pt --out $E/bridge_test.json

# ELD: full fine-tuning and decoder+bridge fine-tuning (scenes 1-6), evaluated on scenes 7-10
$P finetune --init $R/bridge/best.pt --run $R/ft_full --iters 8000 --ms-start 2000 --val-every 1000 --val-crop 1024 > $R/ft_full.out 2>&1 &
$P finetune --init $R/bridge/best.pt --run $R/ft_dec --freeze-encoder --iters 8000 --ms-start 2000 --val-every 1000 --val-crop 1024 > $R/ft_dec.out 2>&1 &
$P meta --init $R/bridge/best.pt --run $R/meta --iters 2000 > $R/meta.out 2>&1
wait
for cam in CanonEOS70D CanonEOS700D NikonD850; do
  for m in ft_full ft_dec; do $P eval --ckpt $R/$m/best.pt --data $cam --crop 1024 --out $E/${m}_$cam.json; done
done

# few-shot: the same adaptation from three initialisations (K = 0, 1, 3, 5 scenes per camera)
$P fewshot --init $R/sid/best.pt    --out $E/fewshot_sid_adam.json
$P fewshot --init $R/bridge/best.pt --out $E/fewshot_bridge_adam.json
$P fewshot --init $R/meta/meta.pt   --out $E/fewshot_meta_adam.json
$P fewshot --init $R/bridge/best.pt --mode inner --steps 30 --shots 1 3 5 --out $E/fewshot_bridge_inner.json
$P fewshot --init $R/meta/meta.pt   --mode inner --steps 30 --shots 1 3 5 --out $E/fewshot_meta_inner.json
echo FEWSHOTDONE

# SID test for the phase-2 models, as they finish
wait_for $R/bridge_nodistill/final.pt
$P eval --ckpt $R/bridge_nodistill/best.pt --out $E/bridge_nodistill_test.json
wait_for $R/ddbm_feature/final.pt
$P eval --ckpt $R/ddbm_feature/best.pt --steps 3 --out $E/ddbm_feature_test_n3.json
$P eval --ckpt $R/ddbm_feature/best.pt --steps 0 --out $E/ddbm_feature_test_n0.json
wait_for $R/bridge_nofilm/final.pt
$P eval --ckpt $R/bridge_nofilm/best.pt --out $E/bridge_nofilm_test.json

# official DDBM: 60k steps written as last.pt at the end
until grep -q '"step": 60000, "val"' $R/ddbm_official/log.jsonl 2>/dev/null; do sleep 120; done
for n in 5 10; do $P eval --ckpt $R/ddbm_official/best.pt --kind ddbm --ddbm-steps $n --out $E/ddbm_official_test_n$n.json; done
$P eval --ckpt $R/ddbm_official/best.pt --kind ddbm --ddbm-steps 40 --limit 100 --out $E/ddbm_official_test_n40_first100.json
echo PHASE3DONE
