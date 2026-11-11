"""Fetch only some SID Sony scenes into /content/Sony with HTTP range requests.

Scene ids start with 0 (train), 1 (test) or 2 (val). A rerun of an evaluation does not need the
training scenes, so this reads a fraction of the dataset zip instead of downloading it whole.
Prefer colab/restore_sid_from_drive.sh: the SID authors ask not to download the data again per run,
and a copy of all three splits is in Google Drive.

    python colab/fetch_sid_test.py [first digits of the scene ids, default 1] [max_files]
"""

import re
import sys
from concurrent.futures import ThreadPoolExecutor

from remotezip import RemoteZip

URL = "https://storage.googleapis.com/isl-datasets/SID/Sony2025.zip"


def fetch(names):
    with RemoteZip(URL) as z:
        for n in names:
            z.extract(n, "/content")


digits = sys.argv[1] if len(sys.argv) > 1 else "1"
with RemoteZip(URL) as z:
    files = [i for i in z.infolist() if re.fullmatch(rf"Sony/(short|long)/[{digits}]\d{{4}}_.*\.ARW", i.filename)]
print(f"{len(files)} files of scenes [{digits}]xxxx, {sum(i.compress_size for i in files) / 1e9:.1f} GB", flush=True)
names = [i.filename for i in files][:int(sys.argv[2]) if len(sys.argv) > 2 else None]
with ThreadPoolExecutor(8) as ex:
    list(ex.map(fetch, [names[i::8] for i in range(8)]))
print(f"fetched {len(names)}")
