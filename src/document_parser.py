"""data/files 아래 문서를 가능한 선에서 텍스트로 변환하는 모듈."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import List

import zlib
import re
from dotenv import load_dotenv

try:  # pragma: no cover
    import pdfplumber  # type: ignore
except ImportError:  # pragma: no cover
    pdfplumber = None

try:  # pragma: no cover
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None

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

# 확장 전처리 옵션
load_dotenv(override=True)
ENABLE_HWPX = os.getenv("ENABLE_HWPX", "1").lower() not in {"0", "false", "no"}
ENABLE_PDF_HTML = os.getenv("ENABLE_PDF_HTML", "1").lower() not in {"0", "false", "no"}
ENABLE_HWP_PDF = os.getenv("ENABLE_HWP_PDF", "0").lower() not in {"0", "false", "no"}
ENABLE_PYMUPDF = os.getenv("ENABLE_PYMUPDF", "1").lower() not in {"0", "false", "no"}
ENABLE_OCR = os.getenv("ENABLE_OCR", "0").lower() not in {"0", "false", "no"}
ENABLE_CAMELOT = os.getenv("ENABLE_CAMELOT", "0").lower() not in {"0", "false", "no"}
OCR_LANG = os.getenv("OCR_LANG", "ko")
HWP2HWPX_BIN = os.getenv("HWP2HWPX_BIN", "hwp2hwpx")
HWP2PDF_BIN = os.getenv("HWP2PDF_BIN", "hwp2pdf")  # LibreOffice/unoconv 등을 지정해도 됨
PDFTOHTML_BIN = os.getenv("PDFTOHTML_BIN", "pdftohtml")

logger = logging.getLogger(__name__)
STATS: Counter = Counter()


def _normalize_filename(name: str) -> str:
    """Remove whitespace for loose filename matching."""
    return re.sub(r"\s+", "", name)


def _resolve_file_path(file_name: str) -> Path | None:
    """Locate file under FILES_ROOT, allowing whitespace-insensitive match."""
    direct = (FILES_ROOT / file_name).resolve()
    if direct.exists():
        return direct

    target_norm = _normalize_filename(file_name)
    for cand in FILES_ROOT.iterdir():
        if _normalize_filename(cand.name) == target_norm:
            logger.info("공백 무시 매칭으로 파일을 찾았습니다: %s -> %s", file_name, cand.name)
            return cand.resolve()
    return None


def _read_text_file(path: Path) -> str:
    for enc in TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_xml_text(xml_content: str) -> str:
    try:
        root = ET.fromstring(xml_content)
    except Exception:
        return ""
    texts: List[str] = []
    for elem in root.iter():
        if elem.text and elem.text.strip():
            texts.append(elem.text.strip())
    return "\n".join(texts)


def _read_hwp_via_hwpx(path: Path) -> str:
    if not ENABLE_HWPX:
        return ""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / (path.stem + ".hwpx")
        try:
            proc = subprocess.run(
                [HWP2HWPX_BIN, str(path), str(out_path)],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                STATS["hwp_hwpx_fail"] += 1
                logger.warning(
                    "hwp2hwpx 변환 실패 (%s): rc=%s, stderr=%s",
                    path,
                    proc.returncode,
                    proc.stderr.strip() if proc.stderr else "",
                )
                return ""
            if not out_path.exists() or out_path.stat().st_size == 0:
                logger.warning("HWPX 변환 결과가 비어 있습니다: %s", out_path)
                return ""
            xml_text = out_path.read_text(encoding="utf-8", errors="ignore")
            parsed = _extract_xml_text(xml_text)
            if parsed:
                STATS["hwp_hwpx"] += 1
                return parsed
        except FileNotFoundError:
            logger.info("hwp2hwpx 변환기를 찾을 수 없어 HWPX 변환을 건너뜁니다: %s", HWP2HWPX_BIN)
        except subprocess.CalledProcessError as exc:
            logger.warning("hwp2hwpx 변환 실패 (%s): %s", path, exc)
        except Exception as exc:  # pragma: no cover
            logger.warning("HWXPX 파싱 실패 (%s): %s", path, exc)
    return ""


def _read_pdf(path: Path) -> str:
    parts: List[str] = []
    # PyMuPDF: 레이아웃 기반 텍스트 추출 보강
    if ENABLE_PYMUPDF:
        if fitz is None:
            logger.info("PyMuPDF 미설치로 pymupdf 파싱을 건너뜁니다: %s", path)
        else:
            try:
                with fitz.open(path) as doc:
                    texts = []
                    for page in doc:
                        texts.append(page.get_text("text") or "")
                    if texts:
                        STATS["pdf_pymupdf"] += 1
                        parts.append("\n".join(texts))
            except Exception as exc:  # pragma: no cover
                logger.warning("PyMuPDF 파싱 실패 (%s): %s", path, exc)

    if pdfplumber is None:
        logger.info("pdfplumber 미설치로 PDF 파싱을 건너뜁니다: %s", path)
    else:
        with pdfplumber.open(path) as pdf:
            texts: List[str] = []
            for page in pdf.pages:
                texts.append(page.extract_text() or "")
            if texts:
                STATS["pdf_plumber"] += 1
                parts.append("\n".join(texts))

    if ENABLE_PDF_HTML:
        try:
            proc = subprocess.run(
                [PDFTOHTML_BIN, "-xml", "-nodrm", "-i", "-stdout", str(path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            xml_text = proc.stdout.decode("utf-8", errors="ignore")
            parsed = _extract_xml_text(xml_text)
            if parsed:
                STATS["pdf_html"] += 1
                parts.append(parsed)
        except FileNotFoundError:
            logger.info("pdftohtml 변환기를 찾을 수 없어 PDF HTML 파싱을 건너뜁니다: %s", PDFTOHTML_BIN)
        except subprocess.CalledProcessError as exc:
            logger.warning("pdftohtml 변환 실패 (%s): %s", path, exc)
        except Exception as exc:  # pragma: no cover
            logger.warning("pdftohtml 파싱 실패 (%s): %s", path, exc)

    # Camelot: 표 추출 텍스트 보강
    if ENABLE_CAMELOT:
        try:
            import camelot  # type: ignore

            tables = camelot.read_pdf(str(path), flavor="lattice", pages="all")
            if tables:
                table_texts = []
                for tb in tables:
                    df = tb.df
                    table_texts.append("\n".join(" ".join(row) for row in df.values.tolist()))
                if table_texts:
                    STATS["pdf_tables"] += 1
                    parts.append("\n".join(table_texts))
        except ImportError:
            logger.info("Camelot 미설치로 표 추출을 건너뜁니다: %s", path)
        except Exception as exc:  # pragma: no cover
            logger.warning("Camelot 표 추출 실패 (%s): %s", path, exc)

    # OCR: PaddleOCR + pdf2image 경로
    if ENABLE_OCR:
        try:
            from paddleocr import PaddleOCR  # type: ignore
            from pdf2image import convert_from_path  # type: ignore
        except ImportError:
            logger.info("PaddleOCR/pdf2image 미설치로 OCR을 건너뜁니다: %s", path)
        else:
            try:
                ocr = PaddleOCR(lang=OCR_LANG, use_angle_cls=True, show_log=False)
                images = convert_from_path(str(path))
                ocr_texts = []
                for img in images:
                    res = ocr.ocr(img)
                    for line in res:
                        for (_, text, _) in line:
                            ocr_texts.append(text)
                if ocr_texts:
                    STATS["pdf_ocr"] += 1
                    parts.append("\n".join(ocr_texts))
            except Exception as exc:  # pragma: no cover
                logger.warning("OCR 추출 실패 (%s): %s", path, exc)

    return "\n".join(parts)


def _convert_hwp_to_pdf(path: Path) -> str:
    """외부 변환기로 HWP를 PDF로 변환한 뒤 PDF 파이프라인으로 처리한다."""
    if not ENABLE_HWP_PDF:
        return ""
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / (path.stem + ".pdf")
        try:
            subprocess.run(
                [HWP2PDF_BIN, str(path), str(pdf_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if not pdf_path.exists() or pdf_path.stat().st_size == 0:
                logger.warning("HWP→PDF 변환 결과가 비어 있습니다: %s", pdf_path)
                return ""
            STATS["hwp_pdf"] += 1
            return _read_pdf(pdf_path)
        except FileNotFoundError:
            logger.info("HWP→PDF 변환기를 찾을 수 없어 건너뜁니다: %s", HWP2PDF_BIN)
        except subprocess.CalledProcessError as exc:
            logger.warning("HWP→PDF 변환 실패 (%s): %s", path, exc)
        except Exception as exc:  # pragma: no cover
            logger.warning("HWP→PDF 변환 중 오류 (%s): %s", path, exc)
    return ""


def _read_docx(path: Path) -> str:
    if Document is None:
        logger.info("python-docx 미설치로 DOCX 파싱을 건너뜁니다: %s", path)
        return ""
    document = Document(path)
    STATS["docx"] += 1
    return "\n".join(para.text for para in document.paragraphs)


def _read_hwp(path: Path) -> str:
    parts: List[str] = []

    # 우선 HWPX 변환 시도 (성공해도 PDF 병행 추출 가능)
    if ENABLE_HWPX:
        parsed = _read_hwp_via_hwpx(path)
        if parsed:
            parts.append(parsed)

    # HWP→PDF 변환 후 PDF 파이프라인 시도 (표/이미지 보강 목적)
    pdf_parsed = _convert_hwp_to_pdf(path)
    if pdf_parsed:
        parts.append(pdf_parsed)

    # 이미 추출된 내용이 있다면 그대로 반환
    if parts:
        return "\n".join(parts)

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
        if texts:
            STATS["hwp_ole"] += 1
    except Exception as exc:  # pragma: no cover
        logger.warning("HWP 파싱 실패 (%s): %s", path, exc)
        return ""
    return "\n".join(texts)


@lru_cache(maxsize=256)
def extract_text(file_name: str) -> str:
    """data/files 안의 파일명을 받아 텍스트를 읽고 캐시한다."""
    if not file_name:
        return ""
    path = _resolve_file_path(file_name)
    if path is None or not path.exists():
        logger.warning("문서 파일을 찾을 수 없습니다: %s", file_name)
        return ""

    suffix = path.suffix.lower()
    try:
        if suffix in {".txt", ".md", ".text", ".csv"}:
            STATS["text"] += 1
            return _read_text_file(path)
        if suffix == ".pdf":
            return _read_pdf(path)
        if suffix in {".docx", ".doc"}:
            return _read_docx(path)
        if suffix == ".hwp":
            return _read_hwp(path)
        # 기타 포맷은 현재 미지원
        logger.info("지원되지 않는 확장자(%s)이므로 data_list 텍스트를 사용합니다.", suffix)
        STATS["unsupported"] += 1
        return ""
    except Exception as exc:  # pragma: no cover
        logger.warning("문서 파싱 실패 (%s): %s", path, exc)
        STATS["error"] += 1
        return ""


def extract_text_from_path(path: Path) -> str:
    """파일 시스템 경로를 직접 받아 텍스트를 읽는다."""
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
        logger.info("지원되지 않는 확장자(%s)이므로 건너뜁니다.", suffix)
        return ""
    except Exception as exc:  # pragma: no cover
        logger.warning("문서 파싱 실패 (%s): %s", path, exc)
        return ""
