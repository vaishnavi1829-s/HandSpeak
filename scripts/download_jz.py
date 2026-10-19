"""Download J and Z ASL alphabet images from the Kaggle ASL Alphabet dataset.

Source dataset (public, GPL 2): https://www.kaggle.com/datasets/grassknoted/asl-alphabet
This is the classic 87,000-image ASL Alphabet dataset (29 classes x 3000 images).

Strategy:
  1. Download the dataset archive (streamed to a temporary file on disk).
  2. Inspect the ZIP's actual internal path structure.
  3. Extract ONLY the J and Z training images into ``dataset/jz/J`` and
     ``dataset/jz/Z``, matching class folders flexibly.
  4. Delete the temporary archive.

This script does NOT touch the existing ``dataset/raw`` Sign Language MNIST CSV
files or any other existing data.
"""

import os
import sys
import tempfile
import zipfile
from collections import Counter

import requests

DATASET_REF = "grassknoted/asl-alphabet"
DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_REF}"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JZ_ROOT = os.path.join(PROJECT_ROOT, "dataset", "jz")
OUT_DIRS = {
    "J": os.path.join(JZ_ROOT, "J"),
    "Z": os.path.join(JZ_ROOT, "Z"),
}

# The classic dataset has a top-level "asl_alphabet_train" folder and separate
# test folders. Training images live under a per-class directory named after
# the letter (A..Z, space, del, nothing). We only care about J and Z.
TARGET_LETTERS = {"J", "Z"}


def download_archive(dest_path):
    """Stream the dataset archive to ``dest_path`` and return the byte count."""
    print(f"Downloading archive from {DOWNLOAD_URL}")
    with requests.get(DOWNLOAD_URL, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0)) or None
        done = 0
        with open(dest_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100.0 / total
                    print(f"\r  {done / (1024 * 1024):.1f} / "
                          f"{total / (1024 * 1024):.1f} MB ({pct:.1f}%)",
                          end="", flush=True)
        print()
    return done


def extract_jz(archive_path):
    """Extract J and Z training images, matching class folders flexibly."""
    print("Inspecting ZIP structure ...")
    with zipfile.ZipFile(archive_path) as zf:
        all_names = [n for n in zf.namelist() if not n.endswith("/")]
        print(f"  total entries: {len(all_names)}")
        print("  first 15 entries:")
        for n in all_names[:15]:
            print(f"    {n}")

        # Identify J/Z images: basename's parent directory must be exactly
        # 'J' or 'Z', and the path must be under the train split (not test).
        jz_members = []
        for n in all_names:
            parts = n.rstrip("/").split("/")
            if len(parts) < 2:
                continue
            parent = parts[-2]
            if parent not in TARGET_LETTERS:
                continue
            # The classic training path is "asl_alphabet_train/<class>/<file>".
            # Avoid the test folder "asl_alphabet_test/<class>_test.jpg".
            if "train" not in n.lower():
                continue
            jz_members.append(n)

        print(f"  matched J/Z training entries: {len(jz_members)}")
        if not jz_members:
            # Fall back: any J/Z class folder regardless of train/test keyword.
            jz_members = []
            for n in all_names:
                parts = n.rstrip("/").split("/")
                if len(parts) >= 2 and parts[-2] in TARGET_LETTERS:
                    jz_members.append(n)
            print(f"  fallback matched J/Z entries: {len(jz_members)}")

        extracted = Counter()
        for name in jz_members:
            parts = name.rstrip("/").split("/")
            letter = parts[-2]
            dest_dir = OUT_DIRS[letter]
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, os.path.basename(name))
            with zf.open(name) as src, open(dest_path, "wb") as dst:
                dst.write(src.read())
            extracted[letter] += 1
        return extracted


def main():
    fd, archive_path = tempfile.mkstemp(suffix=".zip", prefix="asl_alphabet_")
    os.close(fd)
    try:
        download_archive(archive_path)
        extracted = extract_jz(archive_path)
    finally:
        try:
            os.remove(archive_path)
            print(f"Removed temporary archive: {archive_path}")
        except OSError:
            pass

    print("\nExtraction complete:")
    ok = True
    for letter in ("J", "Z"):
        n = extracted.get(letter, 0)
        print(f"  {letter}: {n} images -> {OUT_DIRS[letter]}")
        if n < 1000:
            print(f"[FAIL] {letter} has fewer than 1000 images ({n}).",
                  file=sys.stderr)
            ok = False

    if ok:
        print("OK: J and Z both have at least 1000 images.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
