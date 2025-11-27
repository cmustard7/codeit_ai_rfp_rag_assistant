"""Check which file_names from data_list are missing in data/files."""

from __future__ import annotations

from pathlib import Path

# Add project root to sys.path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import document_parser as dp  # noqa: E402
from src.data_loader import load_project_entries  # noqa: E402


def main() -> None:
    root = dp.FILES_ROOT.resolve()
    names = []
    missing = []

    for e in load_project_entries():
        fname = e.get("file_name")
        if not fname:
            continue
        names.append(fname)
        path = (root / fname).resolve()
        if not path.exists():
            missing.append(fname)

    unique = len(set(names))
    print(f"총 data_list 파일명: {unique}")
    print(f"실제 존재: {unique - len(missing)}, 누락: {len(missing)}")
    if missing:
        print("누락 예시 20개:")
        for x in missing[:20]:
            print(" -", x)


if __name__ == "__main__":
    main()
