#!/bin/bash
# Smoke test of colab/custom_camera.py on SID Sony ARW files arranged like a camera of your own: scene folders with
# ref.ARW (the long exposure) and short frames. Needs the SID test scenes from Drive (colab drivemount) and the DDBM
# checkpoint uploaded by the driver (EXTRA=".../ddbm_vp_unet.pt:/content/runs/ddbm_vp_unet/best.pt").
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
if [ ! -f $F/DONE_custom ]; then
  step custom
  until [ -f /content/runs/sid/best.pt ] && [ -f /content/runs/ddbm_vp_unet/best.pt ]; do sleep 5; done
  python - <<'PY' || fail layout
import os, glob, re
src = "/content/Sony"
scenes = sorted({re.match(r"(\d{5})_", os.path.basename(f)).group(1) for f in glob.glob(f"{src}/short/*.ARW")})[:6]
for i, sc in enumerate(scenes, 1):
    d = f"/content/testcam/scene-{i:02d}"
    os.makedirs(d, exist_ok=True)
    longs = sorted(glob.glob(f"{src}/long/{sc}_*.ARW"))
    shorts = sorted(glob.glob(f"{src}/short/{sc}_*.ARW"))[:3]
    if os.path.lexists(f"{d}/ref.ARW"): continue
    os.symlink(longs[0], f"{d}/ref.ARW")
    for j, s in enumerate(shorts, 1):
        os.symlink(s, f"{d}/short_{j:02d}.ARW")
print("layout", scenes)
PY
  P="python colab/custom_camera.py"
  $P prep --root /content/testcam --name sonytest --workers 4 >> /content/logs/custom.log 2>&1 || fail prep
  $P adapt --name sonytest --k 3 --steps 60 --ddbm-ckpt /content/runs/ddbm_vp_unet/best.pt >> /content/logs/custom.log 2>&1 || fail adapt
  cp /content/runs/custom_sonytest/result.json /content/results/custom_smoke.json
  S=$(ls /content/testcam/scene-05/short_0*.ARW | head -2)
  $P enhance --ckpt /content/runs/custom_sonytest/adapted.pt --inputs $S --ratio 100 --out /content/results/custom_out \
    --ddbm-ckpt /content/runs/ddbm_vp_unet/best.pt >> /content/logs/custom.log 2>&1 || fail enhance
  finish_step custom
fi
touch $F/ALL_DONE
echo RERUNDONE
