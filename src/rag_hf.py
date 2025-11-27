# src/rag_hf.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

from .vector_store import (
    load_vectorstore,
    search_vectorstore,
    search_chroma,
)
from .hf_client import hf_chat

# JSON 벡터스토어 기본 경로 (이미 vector_store.py에서 쓰는 것과 동일하게 맞춰줘도 OK)
DEFAULT_VECTORSTORE_PATH = Path("data/vectorstore.json")


def load_store(path: Path = DEFAULT_VECTORSTORE_PATH) -> dict:
    """JSON 벡터스토어를 읽어오는 헬퍼."""
    return load_vectorstore(path)


def retrieve_contexts(question: str, store: dict, use_chroma: bool = False, top_k: int = 3) -> List[Dict]:
    """
    질문에 대해 상위 top_k 개의 관련 청크를 검색.
    - use_chroma=False: JSON 벡터스토어(search_vectorstore)
    - use_chroma=True : Chroma(search_chroma) 사용
    """
    if use_chroma:
        return search_chroma(question, top_k=top_k)
    else:
        return search_vectorstore(question, store, top_k=top_k)


def build_prompt(question: str, contexts: List[Dict]) -> List[Dict[str, str]]:
    """
    HF/OpenAI 호환 chat-completions 형식에 맞게 messages 리스트 생성.
    """
    # 컨텍스트를 하나의 큰 문자열로 합치기
    context_blocks = []
    for idx, c in enumerate(contexts, start=1):
        meta = c.get("metadata", {})
        header = f"[{idx}] 기관: {meta.get('agency','')}, 사업명: {meta.get('project','')}, 파일: {meta.get('file_name','')}"
        body = c.get("text", "")
        context_blocks.append(f"{header}\n{body}")

    context_text = "\n\n".join(context_blocks) if context_blocks else "관련 문서를 찾지 못했습니다."

    system_msg = (
        "당신은 한국어 RFP 전문 어시스턴트입니다. "
        "아래 '검색된 컨텍스트' 안에서 최대한 근거를 찾아 사용자의 질문에 답변하세요.\n"
        "- 모르면 모른다고 말합니다.\n"
        "- 추측으로 지어내지 않습니다.\n"
        "- 반드시 한국어로 답변합니다."
    )

    user_msg = (
        f"다음은 검색된 컨텍스트입니다:\n"
        f"---\n{context_text}\n---\n\n"
        f"사용자 질문: {question}\n\n"
        f"위 컨텍스트를 기반으로, 질문에 대해 충실하고 간결하게 답변해 주세요."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    return messages


def answer_with_rag(
    question: str,
    store: dict | None = None,
    use_chroma: bool = False,
    top_k: int = 3,
    max_tokens: int = 512,
    temperature: float = 0.3,
) -> Dict:
    """
    시나리오 A: 벡터스토어 + HF/TGI(Qwen) 기반 RAG 한 방에 실행.

    return 형식 예:
    {
        "answer": "...",
        "contexts": [...],  # 사용한 청크 정보
    }
    """
    load_dotenv(override=True)

    # 벡터스토어 로드 (필요시)
    if store is None and not use_chroma:
        store = load_store()

    # 1) 검색
    contexts = retrieve_contexts(question, store, use_chroma=use_chroma, top_k=top_k)

    # 2) 프롬프트 구성
    messages = build_prompt(question, contexts)

    # 3) HF/TGI LLM 호출
    answer_text = hf_chat(messages, max_tokens=max_tokens, temperature=temperature)

    return {
        "answer": answer_text,
        "contexts": contexts,
    }
