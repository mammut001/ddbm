#!/bin/bash
# Seventeenth run: a grid of few-shot adaptation recipes (colab/adapt_grid.py) on SIED Sony (A7R III) and the ELD cameras, three
# seeds each, the two datasets in parallel on one GPU. SIED comes from the temporary HTTP server on the VPS (host and port in
# /content/sied_host.txt, uploaded by colab/vps/driver.sh through EXTRA), ELD from the public Drive file (gdown); if the ELD
# download is refused (quota), only SIED runs and a SKIPPED_eld flag is left.
set -x
F=/content/flags
mkdir -p $F /content/logs /content/results
step() { rm -f $F/RUNNING_*; touch $F/RUNNING_$1; }
finish_step() { rm -f $F/RUNNING_$1; touch $F/DONE_$1; }
fail() { rm -f $F/RUNNING_*; touch $F/FAILED_$1; exit 1; }
pip install -q rawpy exifread pytorch_msssim lpips gdown || fail pip
cd /content/ldr-src
until [ -f /content/sied_host.txt ]; do sleep 5; done
HOST=$(cat /content/sied_host.txt)
if [ ! -f $F/DONE_data ]; then
  step data
  for pair in "Sony_gt.zip:3540159732" "Sony_0.001.zip:11489809753"; do
    f=${pair%%:*}; size=${pair##*:}
    curl -fL -C - --retry 8 --retry-delay 5 "http://$HOST/$f" -o /content/$f >> /content/logs/fetch.log 2>&1 || fail curl
    [ "$(stat -c %s /content/$f)" = "$size" ] || fail size
  done
  python colab/sied_prep.py --zips /content --level 0.001 --n-train 100 >> /content/logs/custom.log 2>&1 || fail prep
  rm -f /content/Sony_gt.zip /content/Sony_0.001.zip
  if [ ! -d /content/ELDraw/ELD ]; then
    if gdown --id 13Ge6-FY9RMPrvGiPvw7O4KS3LNfUXqEX -O /content/ELD.zip > /content/logs/gdown.log 2>&1; then
      mkdir -p /content/ELDraw && unzip -q -o /content/ELD.zip -d /content/ELDraw && rm -f /content/ELD.zip
      python main.py --workers 6 prep-eld --eld-raw /content/ELDraw/ELD > /content/logs/prep_eld.log 2>&1 \
        && python colab/fix_eld_iso.py >> /content/logs/prep_eld.log 2>&1 || touch $F/SKIPPED_eld
    else
      touch $F/SKIPPED_eld
    fi
  fi
  finish_step data
fi
if [ ! -f $F/DONE_grid ]; then
  step grid
  until [ -f /content/runs/sid/best.pt ] && [ -f /content/runs/ddbm_vp_unet/best.pt ]; do sleep 5; done
  R="base cos dec dec_long enc wd"
  python colab/adapt_grid.py sied --seeds 0 1 2 --recipes $R >> /content/logs/grid_sied.log 2>&1 &
  P1=$!
  if [ ! -f $F/SKIPPED_eld ]; then
    python colab/adapt_grid.py eld --seeds 0 1 2 --recipes $R --ks 1 6 >> /content/logs/grid_eld.log 2>&1 &
    P2=$!
  fi
  wait $P1 || fail grid_sied
  [ -n "$P2" ] && { wait $P2 || fail grid_eld; }
  finish_step grid
fi
touch $F/ALL_DONE
echo RERUNDONE
