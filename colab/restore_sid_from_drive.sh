#!/bin/bash
# Restore the SID Sony raw files from Google Drive (saved there by colab/save_sid_to_drive.sh) into
# /content/Sony, instead of downloading the dataset again. Mount Drive first with
# `colab drivemount -s <session>`, which needs one browser authorization per VM.
#   bash colab/restore_sid_from_drive.sh [train] [val] [test]      (default: all three)
set -e
D=/content/drive/MyDrive/SID
command -v zstd > /dev/null || apt-get -qq install -y zstd > /dev/null
for name in ${@:-train val test}; do
  zstd -dc $D/Sony_$name.tar.zst | tar xf - -C /content
  echo "restored $name"
done
