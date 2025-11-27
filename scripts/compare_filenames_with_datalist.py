"""Compare file names in data/files with file_name entries in data_list.csv.

- CSV에만 있고 실제 폴더에 없는 이름
- 폴더에만 있고 CSV에 없는 이름
을 각각 집계해 보여준다.
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

    # CSV 기준 이름 목록
    csv_names = set()
    with DATA_LIST.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        field = None
        if reader.fieldnames:
            for col in reader.fieldnames:
                if col.lower().strip() in {"file_name", "filename", "파일명", "파일 이름"}:
                    field = col
                    break
            if field is None:
                print(f"file_name 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {reader.fieldnames}")
                return
        else:
            print("CSV에 헤더가 없습니다. file_name 컬럼을 지정할 수 없습니다.")
            return
        for row in reader:
            fname = row.get(field)
            if fname:
                csv_names.add(fname.strip())

    # 실제 파일 목록
    file_names = {p.name for p in FILES_DIR.iterdir() if p.is_file()}

    only_in_csv = sorted(csv_names - file_names)
    only_in_files = sorted(file_names - csv_names)
    common = csv_names & file_names

    print(f"CSV 총 파일명: {len(csv_names)}")
    print(f"폴더 총 파일명: {len(file_names)}")
    print(f"공통: {len(common)}")
    print(f"CSV에만 있고 폴더에 없는 것: {len(only_in_csv)}")
    if only_in_csv:
        print("예시 10개 (csv-only):")
        for name in only_in_csv[:10]:
            print(" -", name)
    print(f"폴더에만 있고 CSV에 없는 것: {len(only_in_files)}")
    if only_in_files:
        print("예시 10개 (files-only):")
        for name in only_in_files[:10]:
            print(" -", name)


if __name__ == "__main__":
    main()
