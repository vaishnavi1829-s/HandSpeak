"""Download the RealSign ISL Dataset.zip (Git LFS media endpoint).

The GitHub repository stores ``Dataset.zip`` as a Git LFS pointer.  The real
binary lives at the ``media.githubusercontent.com`` endpoint.  This script
streams it to ``dataset/Dataset.zip`` with progress output and supports
resuming a partial download via HTTP Range requests (re-run it to continue).

Run from the project root::

    python scripts/download_realsign.py
"""

from __future__ import annotations

import os
import sys
import time

import urllib.request

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

LFS_URL = (
    "https://media.githubusercontent.com/media/"
    "RealSign62/RealSign-Indian-Sign-Language-Dataset/main/Dataset.zip"
)
DEST = os.path.join(PROJECT_ROOT, "dataset", "Dataset.zip")
EXPECTED_SIZE = 656_689_688  # from the LFS pointer

CHUNK = 1024 * 1024
_MAX_RETRIES = 50


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def main() -> None:
    os.makedirs(os.path.dirname(DEST), exist_ok=True)

    start = time.time()
    already = os.path.getsize(DEST) if os.path.exists(DEST) else 0

    if already == EXPECTED_SIZE:
        print(f"[REALSIGN] Already downloaded (size matches): {DEST}")
        return
    if already > EXPECTED_SIZE:
        print(f"[REALSIGN] Existing file too large ({already}); removing and restarting.")
        os.remove(DEST)
        already = 0

    print(f"[REALSIGN] Downloading {LFS_URL}")
    print(f"[REALSIGN] Destination: {DEST}")
    print(f"[REALSIGN] Resuming from {_human(already)}")

    retries = 0
    while already < EXPECTED_SIZE and retries < _MAX_RETRIES:
        headers = {"User-Agent": "Mozilla/5.0"}
        if already > 0:
            headers["Range"] = f"bytes={already}-"

        try:
            req = urllib.request.Request(LFS_URL, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                if already > 0 and resp.status != 206:
                    # Server ignored the Range header; restart from scratch.
                    print("[REALSIGN] Server ignored Range; restarting file.")
                    os.remove(DEST)
                    already = 0
                    continue

                mode = "ab" if already > 0 else "wb"
                with open(DEST, mode) as fh:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
                        already += len(chunk)
                        if already % (10 * CHUNK) < CHUNK:
                            pct = 100.0 * already / EXPECTED_SIZE
                            print(f"  {_human(already)} / {_human(EXPECTED_SIZE)} ({pct:.1f}%)")
        except Exception as exc:  # pragma: no cover - network resilience
            print(f"[REALSIGN] Connection interrupted: {exc}")
            print(f"[REALSIGN] Resuming from {_human(already)}...")
            retries += 1
            time.sleep(2)

    elapsed = time.time() - start
    actual = os.path.getsize(DEST)
    print(f"[REALSIGN] Done in {elapsed:.1f}s -> {_human(actual)}")

    if actual != EXPECTED_SIZE:
        print(f"[REALSIGN] WARNING: expected {EXPECTED_SIZE} bytes, got {actual}.")
        print("[REALSIGN] Re-run this script to resume the remaining bytes.")
        sys.exit(1)


if __name__ == "__main__":
    main()
