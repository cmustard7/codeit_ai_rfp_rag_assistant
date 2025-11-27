"""Rename files in data/files to match file_name entries from data_list.csv.

주로 앞뒤 공백을 제거한 뒤 파일명이 안 맞게 된 경우,
data_list.csv에 적힌 정확한 이름으로 다시 맞춰준다.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES_DIR = ROOT / "data" / "files"
DATA_LIST = ROOT / "data" / "data_list.csv"


def main() -> None:
    if not FILES_DIR.exists():
        print(f"Not found: {FILES_DIR}")
        return
    if not DATA_LIST.exists():
        print(f"Not found: {DATA_LIST}")
        return

    existing = list(FILES_DIR.iterdir())
    existing_names = {p.name for p in existing}

    renamed = 0
    resolved = 0
    skipped = 0

    with DATA_LIST.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target = row.get("file_name")
            if not target:
                continue
            target_path = FILES_DIR / target
            if target_path.exists():
                continue  # already present

            # 시도 1: 앞뒤 공백 제거된 이름이 존재하면 원래 이름으로 복구
            stripped = target.strip()
            candidate = None
            if stripped != target:
                for name in existing_names:
                    if name.strip() == stripped:
                        candidate = name
                        break

            if candidate:
                src = FILES_DIR / candidate
                print(f"rename: {candidate!r} -> {target!r}")
                src.rename(target_path)
                existing_names.discard(candidate)
                existing_names.add(target)
                renamed += 1
                continue

            # 시도 2: 동일한 stripped 이름이 없으면 스킵
            skipped += 1

    total = renamed + resolved + skipped
    print(f"processed: {total}, renamed: {renamed}, skipped: {skipped}")


if __name__ == "__main__":
    main()
