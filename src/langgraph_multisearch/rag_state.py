from typing import TypedDict, Dict, List, Any
from langchain_core.documents import Document

class RAGState(TypedDict):
    vector_store: Any
    retriever: Any
    question: str

    # Retrieval 결과
    docs: list
    metadata: dict
    context: str

    # LLM 결과
    answer: str
    prompt: str

    # 평가
    score: float
    evaluate_reason: str

    # RAG 제어용
    retry: int
    refined_query: str

    # 🔥 새로 추가
    is_compare: bool           # 비교 질문인지 여부
    compare_keys: List[str]    # 비교 대상 키워드 (기관명/사업명 등)
    rerank_scores: list
    full_vs: Any               # 전체 vector_store cache
    source_index: Dict[str, List[Document]]
    distilled_context: str