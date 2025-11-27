"""Utility functions to select LLM/embedding providers (OpenAI or HuggingFace)."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 우선 최신 패키지(langchain-huggingface)를 시도하고, 없으면 community로 폴백
try:  # pragma: no cover
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
except ImportError:  # pragma: no cover
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
    HuggingFaceEndpoint = None  # type: ignore


def _env_flag(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() not in {"0", "false", "no"}


def get_embedding_client() -> Any:
    """Return embedding client based on env.

    Default: OpenAI (text-embedding-3-small). If `EMBED_PROVIDER=hf`, use HuggingFace
    sentence-transformers (model name via `HF_EMBED_MODEL`, default all-MiniLM-L6-v2).
    """

    load_dotenv(override=True)
    provider = os.getenv("EMBED_PROVIDER", "openai").lower()
    if provider == "hf":
        # 다국어 대응을 위해 기본값을 multilingual-e5-base로 설정
        model = os.getenv("HF_EMBED_MODEL", "intfloat/multilingual-e5-base")
        return HuggingFaceEmbeddings(model_name=model)
    model = os.getenv("LANGGRAPH_EMBED_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model)


def get_chat_client() -> Any:
    """Return chat client based on env.

    Default: OpenAI gpt-5-mini. If `LLM_PROVIDER=hf`, use HuggingFaceHub-compatible
    chat model (via `HF_LLM_MODEL`, default "mistralai/Mistral-7B-Instruct-v0.1").
    """

    load_dotenv(override=True)
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "hf":
        # HuggingFace Hub/Endpoint LLM. HF 토큰 필요: HUGGINGFACEHUB_API_TOKEN
        model = os.getenv("HF_LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
        task = os.getenv("HF_LLM_TASK", "text-generation")
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        endpoint_url = os.getenv("HF_ENDPOINT_URL")
        if HuggingFaceEndpoint is None:
            raise ImportError("langchain-huggingface 또는 langchain-community의 HuggingFaceEndpoint가 필요합니다. pip install -U langchain-huggingface langchain-community")
        if endpoint_url:
            return HuggingFaceEndpoint(
                endpoint_url=endpoint_url,
                huggingfacehub_api_token=token,
                task=task,
                temperature=temperature,
            )
        return HuggingFaceEndpoint(
            repo_id=model,
            huggingfacehub_api_token=token,
            task=task,
            temperature=temperature,
        )
    model = os.getenv("LANGGRAPH_LLM_MODEL", "gpt-5-mini")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    return ChatOpenAI(model=model, temperature=temperature)


def is_scenario_a() -> bool:
    """Convenience flag: scenario A(on-prem/HF) if provider hints are set to hf."""

    return _env_flag("SCENARIO_A", False) or os.getenv("LLM_PROVIDER", "").lower() == "hf" or os.getenv("EMBED_PROVIDER", "").lower() == "hf"
