"""HWP/HWPX -> PDF converter wrapper for document_parser.py.

Usage:
    python scripts/hwp2pdf_simple.py input.hwp output.pdf

환경변수:
    HWP2PDF_METHOD: auto | office | standalone (default: auto)
        - auto: .hwp는 office, .hwpx는 standalone 시도
        - office: 한컴오피스 엔진 사용 (Windows + 한컴 설치 필요)
        - standalone: 한컴 미설치 환경용 (주로 .hwpx에 권장)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python hwp2pdf_simple.py <input.hwp|hwpx> <output.pdf>", file=sys.stderr)
        return 1

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    method = os.getenv("HWP2PDF_METHOD", "auto")

    try:
        from simple_hwp2pdf import convert
    except Exception as exc:  # broaden to show root cause
        print(
            "simple-hwp2pdf를 불러오지 못했습니다. "
            "pip install \"simple-hwp2pdf[office]\" 로 재설치하거나, 아래 에러를 확인하세요.",
            file=sys.stderr,
        )
        print(f"import error: {exc!r}", file=sys.stderr)
        print("sys.executable:", sys.executable, file=sys.stderr)
        print("sys.path:", sys.path, file=sys.stderr)
        return 1

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        convert(str(src), str(dst), method=method)
    except Exception as exc:  # pragma: no cover
        print(f"convert failed: {exc}", file=sys.stderr)
        return 1

    if not dst.exists() or dst.stat().st_size == 0:
        print(f"PDF not created or empty: {dst}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
