"""GPT-5 Judge 기반 자동 평가 스크립트."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

DEFAULT_RESULTS = Path("data/results.json")
DEFAULT_OUTPUT = Path("data/eval/judge_scores.json")
DEFAULT_MODEL = "gpt-5-mini"

PROMPT_TEMPLATE = """당신은 RAG 시스템 답변을 채점하는 심사위원입니다.
질문과 답변을 읽고, 1~5 정수 점수와 이유를 JSON 하나로만 반환하세요.

이 질문이 문서로 답 가능한가? answerable={answerable}
- answerable=true: 문서 근거로 정확히 답하면 5, 모호/부분이면 3, 틀리거나 “모르겠다”는 1.
- answerable=false: 문서에 없다고 밝히거나 추측을 피하면 4~5, 지어내면 1.

출력은 예시처럼 딱 하나의 JSON만:
{{"score": 4, "reason": "핵심 요구를 충족했으나 예산 세부 설명이 부족"}}
어떤 설명도 덧붙이지 마세요.

질문: {question}
모델 답변: {answer}
"""


def load_dataset(results_path: Path, limit: int | None) -> List[Dict[str, Any]]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    rows = data.get("results", [])
    if limit:
        rows = rows[:limit]
    return rows

def parse_ai_response(content: Any) -> Dict[str, Any]:
    """LLM 응답에서 JSON 부분만 잘라서 dict로 파싱 (실패 시 안전하게 기본값 반환)."""
    if isinstance(content, list):
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    else:
        text = str(content)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except Exception:
        # 파싱 실패 시 기본값 반환
        return {"score": 0, "reason": f"parse_failed: {text[:200]}"}

def get_judge_llm_and_runner(model: str):
    """
    .env 의 LLM_BACKEND 에 따라 Judge용 LLM을 생성하고,
    프롬프트를 실행하는 작은 래퍼(run_fn)를 함께 반환한다.

    - backend = "openai" : ChatOpenAI 사용 (기존 방식)
    - backend = "hf"     : HuggingFaceEndpoint / TGI 등 사용
    """
    load_dotenv()
    backend = os.environ.get("LLM_BACKEND", "openai").lower()

    if backend == "openai":
        # OPENAI_API_KEY 는 .env 에서 load_dotenv 로 로딩됨
        llm = ChatOpenAI(model=model)

        def run_fn(prompt: str) -> Any:
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content

        return llm, run_fn

    elif backend == "hf":
        # HF LLM 설정 (예: Gemma, Mistral 등)
        # 예시 .env:
        #   LLM_BACKEND=hf
        #   HF_LLM_MODEL=google/gemma-2-2b-it
        #   HF_ENDPOINT_URL=http://127.0.0.1:8080
        #   HF_API_TOKEN=hf_...
        from langchain_community.llms import HuggingFaceEndpoint

        endpoint_url = os.environ.get("HF_ENDPOINT_URL")
        api_token = os.environ.get("HF_API_TOKEN")
        hf_model = model or os.environ.get("HF_LLM_MODEL")

        if not endpoint_url:
            raise RuntimeError("HF 백엔드 사용 시 HF_ENDPOINT_URL 환경변수가 필요합니다.")
        if not api_token:
            raise RuntimeError("HF 백엔드 사용 시 HF_API_TOKEN 환경변수가 필요합니다.")

        llm = HuggingFaceEndpoint(
            endpoint_url=endpoint_url,
            huggingfacehub_api_token=api_token,
            task="text-generation",
            model_kwargs={
                "max_new_tokens": 512,
                "temperature": 0.2,
                "repetition_penalty": 1.05,
            },
        )

        def run_fn(prompt: str) -> Any:
            # HF 쪽은 보통 단일 문자열 프롬프트를 사용
            return llm.invoke(prompt)

        return llm, run_fn

    else:
        raise ValueError(f"알 수 없는 LLM_BACKEND 값입니다: {backend}")

def judge_entries(
    entries: List[Dict[str, Any]],
    model: str,
) -> List[Dict[str, Any]]:
    llm, run_llm = get_judge_llm_and_runner(model)
    evaluations: List[Dict[str, Any]] = []

    for entry in entries:
        question = entry.get("question", "").strip()
        answer = entry.get("answer", "").strip()
        if not question or not answer:
            continue

        # answerable/is_fake 플래그를 사용해 judge가 “모른다” 응답을 적절히 평가하도록 힌트 제공
        is_fake = bool(entry.get("is_fake") or entry.get("fake") or entry.get("answerable") is False)
        answerable = "false" if is_fake else "true"

        prompt = PROMPT_TEMPLATE.format(question=question, answer=answer, answerable=answerable)
        content = run_llm(prompt)
        result = parse_ai_response(content)

        evaluations.append(
            {
                "id": entry.get("id"),
                "question": question,
                "answer": answer,
                "answerable": answerable,
                "score": int(result.get("score", 0)),
                "reason": result.get("reason", ""),
            }
        )

    return evaluations

def main() -> None:
    # .env 먼저 로드 (DEFAULT_MODEL 을 env에서 덮어쓰고 싶을 수도 있으니)
    load_dotenv()

    parser = argparse.ArgumentParser(description="LLM Judge 기반 평가")
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help="run_eval 결과 JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="평가 결과 저장 경로",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="평가할 샘플 수 (기본 전체)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("JUDGE_MODEL", DEFAULT_MODEL),
        help="Judge로 사용할 모델명 (기본: JUDGE_MODEL env 또는 gpt-5-mini)",
    )
    args = parser.parse_args()

    entries = load_dataset(args.results, args.limit)
    if not entries:
        raise ValueError("평가할 데이터가 없습니다.")

    evaluations = judge_entries(entries, args.model)
    avg_score = mean(item["score"] for item in evaluations) if evaluations else 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"average": avg_score, "results": evaluations}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"평균 점수: {avg_score:.2f}")
    print(f"저장 완료: {args.output}")


if __name__ == "__main__":
    main()
