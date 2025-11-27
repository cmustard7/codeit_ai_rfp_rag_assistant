"""data/files 아래 문서를 가능한 선에서 텍스트로 변환하는 모듈."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import List

import zlib

try:  # pragma: no cover
    import pdfplumber  # type: ignore
except ImportError:  # pragma: no cover
    pdfplumber = None

try:  # pragma: no cover
    from docx import Document  # type: ignore
except ImportError:  # pragma: no cover
    Document = None

try:  # pragma: no cover
    import olefile  # type: ignore
except ImportError:  # pragma: no cover
    olefile = None

FILES_ROOT = Path("data/files")
TEXT_ENCODINGS = ("utf-8", "cp949", "euc-kr")

logger = logging.getLogger(__name__)


def _read_text_file(path: Path) -> str:
    for enc in TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    if pdfplumber is None:
        logger.info("pdfplumber 미설치로 PDF 파싱을 건너뜁니다: %s", path)
        return ""
    with pdfplumber.open(path) as pdf:
        texts: List[str] = []
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
        return "\n".join(texts)


def _read_docx(path: Path) -> str:
    if Document is None:
        logger.info("python-docx 미설치로 DOCX 파싱을 건너뜁니다: %s", path)
        return ""
    document = Document(path)
    return "\n".join(para.text for para in document.paragraphs)


def _read_hwp(path: Path) -> str:
    if olefile is None:
        logger.info("olefile 미설치로 HWP 파싱을 건너뜁니다: %s", path)
        return ""
    texts: List[str] = []
    try:
        with olefile.OleFileIO(path) as ole:  # type: ignore[attr-defined]
            for entry in ole.listdir():
                if entry and entry[0] == "BodyText":
                    stream = ole.openstream(entry)
                    data = stream.read()
                    if data[:4] == b"\xfe\xff\xff\xff":
                        data = data[4:]
                    try:
                        decompressed = zlib.decompress(data, -15)
                    except zlib.error:
                        decompressed = data
                    try:
                        texts.append(decompressed.decode("utf-16"))
                    except UnicodeDecodeError:
                        texts.append(decompressed.decode("utf-8", errors="ignore"))
    except Exception as exc:  # pragma: no cover
        logger.warning("HWP 파싱 실패 (%s): %s", path, exc)
        return ""
    return "\n".join(texts)


@lru_cache(maxsize=256)
def extract_text(file_name: str) -> str:
    """data/files 안의 파일명을 받아 텍스트를 읽고 캐시한다."""
    if not file_name:
        return ""
    path = (FILES_ROOT / file_name).resolve()
    if not path.exists():
        logger.warning("문서 파일을 찾을 수 없습니다: %s", path)
        return ""

    suffix = path.suffix.lower()
    try:
        if suffix in {".txt", ".md", ".text", ".csv"}:
            return _read_text_file(path)
        if suffix == ".pdf":
            return _read_pdf(path)
        if suffix in {".docx", ".doc"}:
            return _read_docx(path)
        if suffix == ".hwp":
            return _read_hwp(path)
        # 기타 포맷은 현재 미지원
        logger.info("지원되지 않는 확장자(%s)이므로 data_list 텍스트를 사용합니다.", suffix)
        return ""
    except Exception as exc:  # pragma: no cover
        logger.warning("문서 파싱 실패 (%s): %s", path, exc)
        return ""
