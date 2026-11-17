#!/bin/bash
# Copies the SIED Sony files (public dataset, Jiang et al., ICCV 2025) from a temporary HTTP server on the VPS into the
# Drive that is mounted on this VM, under MyDrive/SIED. The host and port come from /content/sied_host.txt (uploaded
# by colab/vps/driver.sh through EXTRA), so no address is stored in the repository. Needs `colab drivemount`.
set -x
F=/content/flags
mkdir -p $F /content/logs
step() { rm -f $F/RUNNING_*; touch $F/RUNNING_$1; }
finish_step() { rm -f $F/RUNNING_$1; touch $F/DONE_$1; }
fail() { rm -f $F/RUNNING_*; touch $F/FAILED_$1; exit 1; }
until [ -f /content/drive/MyDrive/.shortcut-targets-by-id ] || [ -d /content/drive/MyDrive ]; do sleep 5; done
until [ -f /content/sied_host.txt ]; do sleep 5; done
HOST=$(cat /content/sied_host.txt)
mkdir -p /content/drive/MyDrive/SIED
step fetch
for pair in "Sony_gt.zip:3540159732" "Sony_0.001.zip:11489809753"; do
  f=${pair%%:*}; size=${pair##*:}
  if [ "$(stat -c %s /content/drive/MyDrive/SIED/$f 2>/dev/null)" = "$size" ]; then echo "$f already in Drive"; continue; fi
  curl -fL -C - --retry 8 --retry-delay 5 "http://$HOST/$f" -o /content/$f >> /content/logs/fetch.log 2>&1 || fail curl
  [ "$(stat -c %s /content/$f)" = "$size" ] || fail size
  cp /content/$f /content/drive/MyDrive/SIED/$f.part >> /content/logs/fetch.log 2>&1 || fail cp
  mv /content/drive/MyDrive/SIED/$f.part /content/drive/MyDrive/SIED/$f
  [ "$(stat -c %s /content/drive/MyDrive/SIED/$f)" = "$size" ] || fail drivesize
  rm -f /content/$f
done
ls -la /content/drive/MyDrive/SIED > /content/logs/drive_listing.txt
finish_step fetch
touch $F/ALL_DONE
echo RERUNDONE
