"""
문서 로딩 모듈 (PDF, HWP)
"""
import os
from pathlib import Path
from typing import List, Dict, Optional
import logging

# PDF 로딩
try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

# HWP 로딩
import olefile
import zlib
import struct

logger = logging.getLogger(__name__)


class DocumentLoader:
    """문서 로더 클래스"""

    @staticmethod
    def load_pdf(file_path: str) -> str:
        """
        PDF 파일을 로드하여 텍스트 추출

        Args:
            file_path: PDF 파일 경로

        Returns:
            추출된 텍스트
        """
        try:
            reader = PdfReader(file_path)
            text = ""

            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"페이지 {page_num} 추출 실패: {e}")

            return text.strip()

        except Exception as e:
            logger.error(f"PDF 로딩 실패 ({file_path}): {e}")
            return ""

    @staticmethod
    def load_hwp(file_path: str) -> str:
        """
        HWP 파일을 로드하여 텍스트 추출
        olefile을 사용한 기본적인 텍스트 추출

        Args:
            file_path: HWP 파일 경로

        Returns:
            추출된 텍스트
        """
        try:
            f = olefile.OleFileIO(file_path)

            # HWP 파일의 텍스트는 BodyText 섹션에 저장됨
            dirs = f.listdir()

            text = ""
            for dir_name in dirs:
                if dir_name[0] == "BodyText":
                    stream_name = "/".join(dir_name)
                    try:
                        data = f.openstream(stream_name).read()

                        # HWP 5.0 포맷 파싱
                        # 간단한 텍스트 추출 (완벽하지 않을 수 있음)
                        unpacked = zlib.decompress(data, -15)
                        text += unpacked.decode('utf-16le', errors='ignore')
                    except Exception as e:
                        logger.warning(f"스트림 {stream_name} 추출 실패: {e}")

            f.close()
            return text.strip()

        except Exception as e:
            logger.error(f"HWP 로딩 실패 ({file_path}): {e}")
            # 대안: LibreOffice/한컴오피스 변환 필요할 수 있음
            logger.info("HWP 파일은 완전한 추출을 위해 변환이 필요할 수 있습니다.")
            return ""

    @staticmethod
    def load_document(file_path: str) -> Dict[str, str]:
        """
        문서 파일을 로드 (PDF 또는 HWP 자동 감지)

        Args:
            file_path: 문서 파일 경로

        Returns:
            문서 정보 딕셔너리 (text, file_name, file_type)
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        extension = file_path.suffix.lower()

        if extension == '.pdf':
            text = DocumentLoader.load_pdf(str(file_path))
            file_type = 'pdf'
        elif extension == '.hwp':
            text = DocumentLoader.load_hwp(str(file_path))
            file_type = 'hwp'
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {extension}")

        return {
            'text': text,
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_type': file_type,
            'char_count': len(text)
        }

    @staticmethod
    def load_documents_from_directory(
        directory: str,
        file_extensions: List[str] = ['.pdf', '.hwp']
    ) -> List[Dict[str, str]]:
        """
        디렉토리 내 모든 문서 로드

        Args:
            directory: 문서가 있는 디렉토리
            file_extensions: 로드할 파일 확장자 목록

        Returns:
            문서 정보 딕셔너리 리스트
        """
        directory = Path(directory)
        documents = []

        for ext in file_extensions:
            files = list(directory.glob(f"*{ext}"))

            for file_path in files:
                try:
                    logger.info(f"로딩 중: {file_path.name}")
                    doc = DocumentLoader.load_document(str(file_path))
                    documents.append(doc)
                    logger.info(f"  → {doc['char_count']} 글자 추출")
                except Exception as e:
                    logger.error(f"문서 로딩 실패 ({file_path.name}): {e}")

        logger.info(f"\n총 {len(documents)}개 문서 로드 완료")
        return documents


# 유틸리티 함수
def load_single_document(file_path: str) -> Dict[str, str]:
    """단일 문서 로드 (편의 함수)"""
    return DocumentLoader.load_document(file_path)


def load_all_documents(directory: str) -> List[Dict[str, str]]:
    """디렉토리 내 모든 문서 로드 (편의 함수)"""
    return DocumentLoader.load_documents_from_directory(directory)
