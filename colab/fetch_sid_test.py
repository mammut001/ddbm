"""Fetch only the SID Sony test scenes (ids 1xxxx) into /content/Sony with HTTP range requests.

A rerun of the test evaluation does not need the training scenes, so this reads about a fifth of
the dataset zip instead of downloading it whole.

    python colab/fetch_sid_test.py [max_files]
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


with RemoteZip(URL) as z:
    files = [i for i in z.infolist() if re.fullmatch(r"Sony/(short|long)/1\d{4}_.*\.ARW", i.filename)]
print(f"{len(files)} test files, {sum(i.compress_size for i in files) / 1e9:.1f} GB", flush=True)
names = [i.filename for i in files][:int(sys.argv[1]) if len(sys.argv) > 1 else None]
with ThreadPoolExecutor(8) as ex:
    list(ex.map(fetch, [names[i::8] for i in range(8)]))
print(f"fetched {len(names)}")
