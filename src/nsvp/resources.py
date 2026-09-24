from __future__ import annotations

import hashlib
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .execution import check_cancelled


@contextmanager
def device_lock(device: str | None, *, directory: Path | None = None) -> Iterator[None]:
    if device is None or device == "cpu":
        yield
        return
    # ROCm uses CUDA-style device names in the external torch runners.
    identity = device.replace("rocm", "cuda")
    if ":" not in identity:
        identity += ":0"
    root = directory or Path(tempfile.gettempdir()) / "nsvp-device-locks"
    root.mkdir(parents=True, exist_ok=True)
    path = root / (hashlib.sha256(identity.encode()).hexdigest() + ".lock")
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        acquired = False
        try:
            while not acquired:
                check_cancelled()
                stream.seek(0)
                try:
                    if sys.platform == "win32":
                        import msvcrt
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError:
                    time.sleep(0.1)
                except OSError as exc:
                    if sys.platform != "win32" or exc.errno not in (13, 11, 36):
                        raise
                    time.sleep(0.1)
            yield
        finally:
            if acquired:
                stream.seek(0)
                if sys.platform == "win32":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
