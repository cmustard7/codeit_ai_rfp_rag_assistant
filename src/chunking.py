"""
문서 청킹 모듈
"""
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

from . import config

logger = logging.getLogger(__name__)


class DocumentChunker:
    """문서 청킹 클래스"""

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP
    ):
        """
        Args:
            chunk_size: 청크 크기
            chunk_overlap: 청크 간 중첩 크기
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        logger.info(f"DocumentChunker 초기화: chunk_size={chunk_size}, overlap={chunk_overlap}")

    def chunk_text(self, text: str) -> List[str]:
        """
        텍스트를 청크로 분할

        Args:
            text: 분할할 텍스트

        Returns:
            청크 리스트
        """
        chunks = self.text_splitter.split_text(text)
        return chunks

    def chunk_document(
        self,
        document: Dict[str, str],
        add_metadata: bool = True
    ) -> List[Dict[str, str]]:
        """
        문서를 청크로 분할하고 메타데이터 추가

        Args:
            document: 문서 딕셔너리 (text, file_name 등)
            add_metadata: 메타데이터 추가 여부

        Returns:
            청크 딕셔너리 리스트
        """
        text = document.get('text', '')
        chunks = self.chunk_text(text)

        chunk_dicts = []
        for i, chunk in enumerate(chunks):
            chunk_dict = {
                'text': chunk,
                'chunk_id': i,
                'chunk_size': len(chunk)
            }

            if add_metadata:
                # 원본 문서 메타데이터 추가
                chunk_dict['file_name'] = document.get('file_name', '')
                chunk_dict['file_type'] = document.get('file_type', '')
                chunk_dict['file_path'] = document.get('file_path', '')

                # 추가 메타데이터가 있으면 포함
                for key, value in document.items():
                    if key not in ['text', 'char_count']:
                        chunk_dict[key] = value

            chunk_dicts.append(chunk_dict)

        return chunk_dicts

    def chunk_documents(
        self,
        documents: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        여러 문서를 청크로 분할

        Args:
            documents: 문서 딕셔너리 리스트

        Returns:
            모든 청크 딕셔너리 리스트
        """
        all_chunks = []

        for doc in documents:
            try:
                chunks = self.chunk_document(doc)
                all_chunks.extend(chunks)
                logger.info(f"{doc.get('file_name', 'Unknown')}: {len(chunks)}개 청크 생성")
            except Exception as e:
                logger.error(f"청킹 실패 ({doc.get('file_name', 'Unknown')}): {e}")

        logger.info(f"총 {len(all_chunks)}개 청크 생성 완료")
        return all_chunks


# 편의 함수
def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP) -> List[str]:
    """텍스트를 청크로 분할하는 간단한 함수"""
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_text(text)


def chunk_documents(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """문서들을 청크로 분할하는 간단한 함수"""
    chunker = DocumentChunker()
    return chunker.chunk_documents(documents)
