"""LLM answer node that consumes the assembled context."""

from __future__ import annotations

import os
from typing import Dict

from dotenv import dotenv_values, load_dotenv
from langchain_openai import ChatOpenAI

from ..graph_state import GraphState
from ..providers import get_chat_client

SYSTEM_PROMPT = (
    "당신은 RAG Assistant입니다. 주어진 컨텍스트만 근거로 한국어로 간결히 답하세요. "
    "불확실하면 모른다고 답하고, 불릿 3~5개 이내, 전체 400자 이내로 요약해 주세요."
)
MODEL_NAME = os.environ.get("LANGGRAPH_LLM_MODEL", "gpt-5-mini")
MODEL_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0"))
DOTENV_PATH = os.environ.get("LANGGRAPH_DOTENV", ".env")
DISTILLATION = os.getenv("DISTILLATION", "0").lower() not in {"0", "false", "no"}
SMALL_LLM_MODEL = os.getenv("SMALL_LLM_MODEL", "gpt-5-mini")
LARGE_LLM_MODEL = os.getenv("LARGE_LLM_MODEL", os.getenv("LANGGRAPH_LLM_MODEL", "gpt-5-nano"))


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


def answer_question(state: GraphState) -> Dict[str, str]:
    """Generate an answer using the global ChatOpenAI client."""
    context = state.get("context", "")
    question = state.get("current_question", "")
    prompt = f"SYSTEM: {SYSTEM_PROMPT}\nCONTEXT:\n{context}\n\nQUESTION: {question}"
    client = _llm
    if DISTILLATION and _small_llm and _large_llm:
        response = _small_llm.invoke(prompt)
        answer_tmp = response.content if isinstance(response.content, str) else str(response.content)
        # 간단한 휴리스틱: 너무 짧거나 '모르' 포함 시 상위 모델로 재시도
        if (len(answer_tmp) < 50) or ("모르" in answer_tmp):
            response = _large_llm.invoke(prompt)
    else:
        response = _llm.invoke(prompt)
    usage = {}
    if hasattr(response, "response_metadata"):
        usage = response.response_metadata.get("token_usage", {}) or {}
    answer = response.content if isinstance(response.content, str) else str(response.content)
    if len(answer) > 400:
        answer = answer[:400]
    return {"last_answer": answer, "context": context, "usage": usage}
