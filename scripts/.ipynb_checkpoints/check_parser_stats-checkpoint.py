"""Quick helper to inspect parser path usage (HWPX/PDF/etc)."""

from __future__ import annotations

from collections import Counter
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import document_parser as dp  # noqa: E402
from src.data_loader import load_project_entries  # noqa: E402


def main() -> None:
    # 캐시/카운터 초기화
    try:
        dp.extract_text.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass
    dp.STATS.clear()
    missing = 0
    for entry in load_project_entries():
        fname = entry.get("file_name")
        if fname:
            _ = dp.extract_text(fname)
            path = dp._resolve_file_path(fname)  # type: ignore[attr-defined]
            if path is None or not path.exists():
                missing += 1

    stats: Counter = dp.STATS
    print("=== document_parser stats ===")
    total = sum(stats.values())
    for key, value in stats.most_common():
        print(f"{key:12s}: {value}")
    print(f"total counted : {total}")
    if missing:
        print(f"missing files : {missing} (data/files 경로에서 찾을 수 없음)")


if __name__ == "__main__":
    main()
