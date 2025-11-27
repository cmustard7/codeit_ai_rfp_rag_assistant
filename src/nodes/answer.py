"""LLM answer node that consumes the assembled context."""

from __future__ import annotations

import os
from typing import Dict

from dotenv import dotenv_values, load_dotenv
from langchain_openai import ChatOpenAI

from ..graph_state import GraphState

SYSTEM_PROMPT = """당신은 RAG Assistant입니다. 주어진 컨텍스트만 근거로 한국어로 대답하세요."""
MODEL_NAME = "gpt-5-mini"
MODEL_TEMPERATURE = 0
DOTENV_PATH = os.environ.get("LANGGRAPH_DOTENV", ".env")


def _resolve_api_key() -> str:
    load_dotenv(dotenv_path=DOTENV_PATH, override=True)
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    values = dotenv_values(DOTENV_PATH)
    key = values.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    return key


<<<<<<< Updated upstream
_llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=MODEL_TEMPERATURE,
    api_key=_resolve_api_key(),
)


def answer_question(state: GraphState) -> Dict[str, str]:
    """Generate an answer using the global ChatOpenAI client."""
    context = state.get("context", "")
    question = state.get("current_question", "")
    prompt = f"SYSTEM: {SYSTEM_PROMPT}\nCONTEXT:\n{context}\n\nQUESTION: {question}"
    answer = _llm.invoke(prompt).content
    return {"last_answer": answer, "context": context}
=======
_llm = get_chat_client()
_small_llm = None
_large_llm = None
if DISTILLATION:
    try:
        _small_llm = ChatOpenAI(model=SMALL_LLM_MODEL, temperature=MODEL_TEMPERATURE)
        _large_llm = ChatOpenAI(model=LARGE_LLM_MODEL, temperature=MODEL_TEMPERATURE)
    except Exception:
        _small_llm = None
        _large_llm = None
        
# 최대 프롬프트 길이 (문자 단위) – 안전하게 3500자로 제한
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", "3500"))

def _extract_answer(response) -> str:
    """HF(str) / ChatOpenAI(message) 등 다양한 타입을 안전하게 문자열로 변환."""
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        content = response.content
        return content if isinstance(content, str) else str(content)
    return str(response)


def answer_question(state: GraphState) -> Dict[str, str]:
    """Generate an answer using the global LLM client (OpenAI or HF)."""
    context = state.get("context", "") or ""
    question = state.get("current_question", "") or ""

    # ===== 1) 프롬프트 길이 제한: context를 잘라서 4096 토큰 제한 회피 =====
    prefix = f"SYSTEM: {SYSTEM_PROMPT}\nCONTEXT:\n"
    suffix = f"\n\nQUESTION: {question}"

    # 전체 프롬프트 최대 길이 안에서 context에 쓸 수 있는 최대 길이 계산
    max_context_len = MAX_PROMPT_CHARS - len(prefix) - len(suffix)
    if max_context_len < 0:
        max_context_len = 0

    if len(context) > max_context_len:
        trimmed_context = context[:max_context_len]
    else:
        trimmed_context = context

    prompt = f"{prefix}{trimmed_context}{suffix}"

    client = _llm
    response = None

    # ===== 2) 지식 증류(distillation) 로직 (HF/OpenAI 모두 대응) =====
    if DISTILLATION and _small_llm and _large_llm:
        # 1차: 작은 모델
        response = _small_llm.invoke(prompt)
        answer_tmp = _extract_answer(response)

        # 간단한 휴리스틱: 너무 짧거나 '모르' 포함 시 상위 모델로 재시도
        if (len(answer_tmp) < 50) or ("모르" in answer_tmp):
            response = _large_llm.invoke(prompt)
    else:
        response = client.invoke(prompt)

    usage = {}
    if hasattr(response, "response_metadata"):
        usage = response.response_metadata.get("token_usage", {}) or {}

    # ===== 3) 응답 타입 공통 처리 =====
    answer = _extract_answer(response)

    # 너무 길면 잘라서 저장 (평가용이라 400자 제한)
    if len(answer) > 400:
        answer = answer[:400]

    # context는 "실제로 사용한" trimmed_context를 넣는 게 디버깅/로그에 더 유용
    return {"last_answer": answer, "context": trimmed_context, "usage": usage}
>>>>>>> Stashed changes
