"""Trim leading/trailing spaces from filenames in data/files."""

from __future__ import annotations

import os
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
        base, ext = os.path.splitext(name)
        new_name = base.strip() + ext  # 앞뒤 공백 제거
        if new_name == name:
            continue
        src = FILES_DIR / name
        dst = FILES_DIR / new_name
        if dst.exists():
            print(f"skip (exists): {name!r} -> {new_name!r}")
            skipped += 1
            continue
        print(f"rename: {name!r} -> {new_name!r}")
        os.rename(src, dst)
        renamed += 1

    print(f"renamed: {renamed}, skipped: {skipped}")


if __name__ == "__main__":
    main()
