"""Verify the downloaded J/Z ASL alphabet images.

Reports, for each folder:
  - number of files
  - number of readable images
  - file format distribution (by extension and by PIL detected format)
  - image dimensions distribution
  - channel mode distribution (grayscale vs RGB)
"""

import os
from collections import Counter

from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FOLDERS = {
    "J": os.path.join(PROJECT_ROOT, "dataset", "jz", "J"),
    "Z": os.path.join(PROJECT_ROOT, "dataset", "jz", "Z"),
}


def verify_folder(letter):
    folder = FOLDERS[letter]
    print(f"\n===== {letter} ({folder}) =====")
    if not os.path.isdir(folder):
        print("  MISSING FOLDER")
        return

    all_files = os.listdir(folder)
    print(f"  Total files on disk : {len(all_files)}")

    ext_counter = Counter()
    fmt_counter = Counter()
    mode_counter = Counter()
    size_counter = Counter()
    readable = 0
    errors = []

    for name in sorted(all_files):
        path = os.path.join(folder, name)
        ext_counter[os.path.splitext(name)[1].lower()] += 1
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                fmt_counter[img.format] += 1
                mode_counter[img.mode] += 1
                size_counter[img.size] += 1
            readable += 1
        except Exception as exc:  # noqa: BLE001
            errors.append((name, str(exc)))

    print(f"  Readable images    : {readable}")
    print(f"  Extensions         : {dict(ext_counter)}")
    print(f"  Formats (PIL)      : {dict(fmt_counter)}")
    print(f"  Modes (channels)   : {dict(mode_counter)}")
    print(f"  Dimensions (W,H)   : {dict(size_counter)}")
    if errors:
        print(f"  Unreadable ({len(errors)}):")
        for name, err in errors[:10]:
            print(f"    - {name}: {err}")


def main():
    for letter in ("J", "Z"):
        verify_folder(letter)
    print("\nDone.")


if __name__ == "__main__":
    main()
