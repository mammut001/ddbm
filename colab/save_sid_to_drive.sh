#!/bin/bash
# Save the SID Sony raw files of this VM into Google Drive (mounted at /content/drive by `colab drivemount`),
# one zstd-compressed tar per split, so that later VMs restore them from Drive instead of downloading the
# dataset again: the SID authors ask for one download per user, because their storage bills grew a lot.
#   bash colab/save_sid_to_drive.sh
set -e
D=/content/drive/MyDrive/SID
mkdir -p $D
command -v zstd > /dev/null || apt-get -qq install -y zstd > /dev/null
cd /content
for s in train:0 val:2 test:1; do
  name=${s%%:*}; d=${s##*:}
  [ -f $D/Sony_$name.tar.zst ] && continue
  ls Sony/short/$d*.ARW Sony/long/$d*.ARW | nice -n 19 tar cf - -T - | nice -n 19 zstd -q -T4 -3 -o $D/Sony_$name.tar.zst.part
  mv $D/Sony_$name.tar.zst.part $D/Sony_$name.tar.zst
  echo "saved $name $(du -h $D/Sony_$name.tar.zst | cut -f1)"
done
echo DRIVEDONE
