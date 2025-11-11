"""
ChromaDB Vector Store 관리 모듈
"""
import os
from typing import List, Dict, Optional
import logging

import chromadb
from packaging import version
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from . import config

logger = logging.getLogger(__name__)


class ChromaDBManager:
    """ChromaDB 관리 클래스"""

    def __init__(
        self,
        collection_name: str = config.COLLECTION_NAME,
        persist_directory: str = None,
        api_key: str = None
    ):
        """
        Args:
            collection_name: 컬렉션 이름
            persist_directory: 데이터 저장 디렉토리
            api_key: OpenAI API Key
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or str(config.CHROMA_DB_DIR)
        self.api_key = api_key or config.OPENAI_API_KEY

        # 디렉토리 생성
        os.makedirs(self.persist_directory, exist_ok=True)

        # 임베딩 모델
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=self.api_key
        )

        # ChromaDB 클라이언트
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        logger.info(f"ChromaDB 초기화: {self.persist_directory}")
        logger.info(f"컬렉션: {self.collection_name}")
        self._chroma_version = version.parse(getattr(chromadb, "__version__", "0.0.0"))

    def create_vectorstore(
        self,
        chunks: List[Dict[str, str]],
        reset: bool = False
    ):
        """
        청크로부터 Vector Store 생성

        Args:
            chunks: 청크 딕셔너리 리스트
            reset: 기존 컬렉션 삭제 후 재생성 여부
        """
        logger.info(f"{len(chunks)}개 청크로 Vector Store 생성 중...")

        # 기존 컬렉션 삭제
        if reset:
            try:
                self.client.delete_collection(name=self.collection_name)
                logger.info(f"기존 컬렉션 '{self.collection_name}' 삭제")
            except:
                pass

        # 텍스트와 메타데이터 분리
        texts = [chunk['text'] for chunk in chunks]
        metadatas = []

        for chunk in chunks:
            metadata = {k: v for k, v in chunk.items() if k != 'text'}
            # ChromaDB는 숫자 타입도 지원
            metadatas.append(metadata)

        # LangChain Chroma 사용
        self.vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )

        # 0.4 버전 이전 Chroma는 수동 persist 호출이 필요
        if self._chroma_version < version.parse("0.4.0") and hasattr(self.vectorstore, "persist"):
            self.vectorstore.persist()

        logger.info(f"✅ Vector Store 생성 완료: {len(texts)}개 문서")

        return self.vectorstore

    def load_vectorstore(self):
        """
        기존 Vector Store 로드

        Returns:
            Chroma vectorstore
        """
        try:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )

            logger.info(f"✅ Vector Store 로드 완료: {self.collection_name}")
            return self.vectorstore

        except Exception as e:
            logger.error(f"Vector Store 로드 실패: {e}")
            return None

    def similarity_search(
        self,
        query: str,
        k: int = config.TOP_K,
        filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        유사도 검색

        Args:
            query: 검색 쿼리
            k: 반환할 상위 k개 문서
            filter: 메타데이터 필터

        Returns:
            검색 결과 리스트
        """
        if not hasattr(self, 'vectorstore'):
            logger.error("Vector Store가 로드되지 않았습니다.")
            return []

        try:
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter
            )

            return results

        except Exception as e:
            logger.error(f"검색 실패: {e}")
            return []

    def similarity_search_with_score(
        self,
        query: str,
        k: int = config.TOP_K,
        filter: Optional[Dict] = None
    ) -> List[tuple]:
        """
        유사도 점수와 함께 검색

        Args:
            query: 검색 쿼리
            k: 반환할 상위 k개 문서
            filter: 메타데이터 필터

        Returns:
            (문서, 점수) 튜플 리스트
        """
        if not hasattr(self, 'vectorstore'):
            logger.error("Vector Store가 로드되지 않았습니다.")
            return []

        try:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )

            return results

        except Exception as e:
            logger.error(f"검색 실패: {e}")
            return []

    def get_collection_info(self) -> Dict:
        """
        컬렉션 정보 조회

        Returns:
            컬렉션 정보 딕셔너리
        """
        try:
            collection = self.client.get_collection(name=self.collection_name)

            info = {
                'name': collection.name,
                'count': collection.count(),
                'metadata': collection.metadata
            }

            return info

        except Exception as e:
            logger.error(f"컬렉션 정보 조회 실패: {e}")
            return {}


# 편의 함수
def create_vectorstore(chunks: List[Dict[str, str]], api_key: str = None):
    """Vector Store 생성 편의 함수"""
    manager = ChromaDBManager(api_key=api_key)
    return manager.create_vectorstore(chunks)


def load_vectorstore(api_key: str = None):
    """Vector Store 로드 편의 함수"""
    manager = ChromaDBManager(api_key=api_key)
    return manager.load_vectorstore()
