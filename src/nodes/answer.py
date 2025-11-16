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
