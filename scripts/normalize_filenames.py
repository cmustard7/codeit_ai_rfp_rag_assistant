"""Normalize file names in data/files to NFC to avoid macOS/Windows mismatch."""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES_DIR = ROOT / "data" / "files"


def main() -> None:
    if not FILES_DIR.exists():
        print(f"Not found: {FILES_DIR}")
        return

    renamed = 0
    skipped = 0
    for name in os.listdir(FILES_DIR):
        nfc = unicodedata.normalize("NFC", name)
        if name == nfc:
            continue
        src = FILES_DIR / name
        dst = FILES_DIR / nfc
        if dst.exists():
            print(f"skip (exists): {name} -> {nfc}")
            skipped += 1
            continue
        print(f"rename: {name} -> {nfc}")
        os.rename(src, dst)
        renamed += 1

    print(f"renamed: {renamed}, skipped: {skipped}")


if __name__ == "__main__":
    main()
