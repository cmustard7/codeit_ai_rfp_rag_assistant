from __future__ import annotations

import os
import requests
from typing import List, Dict, Any

from dotenv import load_dotenv

# .env 로부터 환경변수 로드
load_dotenv()

HF_ENDPOINT_URL = os.getenv("HF_ENDPOINT_URL", "http://127.0.0.1:8080")
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")


def _build_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    # 필요하면 인증 헤더 추가 (TGI가 토큰 요구하는 경우)
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    return headers


def hf_chat(
    messages: List[Dict[str, Any]],
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    """
    Hugging Face TGI(OpenAI 호환 /v1/chat/completions)에 요청 보내는 헬퍼 함수.
    messages 형식은 OpenAI Chat API와 동일하게 사용:
    [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      ...
    ]
    """
    base_url = HF_ENDPOINT_URL.rstrip("/")
    url = f"{base_url}/v1/chat/completions"

    payload = {
        "model": HF_LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    resp = requests.post(url, headers=_build_headers(), json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # OpenAI 스타일 응답에서 content만 뽑기
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Unexpected response format from HF endpoint: {data}") from e
