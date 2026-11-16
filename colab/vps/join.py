"""Run on the VM by colab/vps/driver.sh: joins the state.tgz.part-* pieces uploaded in /content into /content/state.tgz."""
import glob
import os

parts = sorted(glob.glob("/content/state.tgz.part-*"))
if parts:  # a rerun after a cut-off call finds no pieces and keeps the joined file
    with open("/content/state.tgz.tmp", "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                while chunk := f.read(1 << 24):
                    out.write(chunk)
    os.replace("/content/state.tgz.tmp", "/content/state.tgz")
    for p in parts:
        os.remove(p)
print("joined", len(parts), os.path.getsize("/content/state.tgz"))
