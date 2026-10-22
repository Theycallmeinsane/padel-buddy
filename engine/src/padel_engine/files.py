"""Writing outputs so a reader never sees half a file.

While a match is analysed the app reads results as they are published, so
each file is written beside its final name and then moved over it in one
step. On Windows the move fails while another process has the old file
open; it is retried for a moment, then written in place as a last resort.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path


def write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        for attempt in range(20):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))
        path.write_bytes(data)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_text(path: str | Path, text: str) -> None:
    write_bytes(path, text.encode("utf-8"))
