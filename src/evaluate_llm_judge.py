"""GPT-5 Judge 기반 자동 평가 스크립트."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

DEFAULT_RESULTS = Path("data/results.json")
DEFAULT_OUTPUT = Path("data/eval/judge_scores.json")
DEFAULT_MODEL = "gpt-5-mini"

PROMPT_TEMPLATE = """당신은 RAG 시스템의 성능을 평가하는 심사위원입니다.
주어진 질문과 모델의 답변을 읽고, 1에서 5까지의 정수 점수로만 평가하십시오.

- 5점: 질문 요구를 정확하게 충족하고, 정보가 명확하며 모순이 없음
- 3점: 부분적으로 맞지만 중요한 세부 정보가 빠지거나 흐릿함
- 1점: 질문에 어긋나거나 사실상 잘못됨

반드시 JSON 형식으로만 응답하십시오.
예: {{"score": 4, "reason": "핵심 요구를 충족했으나 예산 세부 설명이 부족"}}

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
    if isinstance(content, list):
        text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    else:
        text = str(content)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def judge_entries(entries: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    load_dotenv()
    llm = ChatOpenAI(model=model)
    evaluations: List[Dict[str, Any]] = []
    for entry in entries:
        question = entry.get("question", "").strip()
        answer = entry.get("answer", "").strip()
        if not question or not answer:
            continue
        prompt = PROMPT_TEMPLATE.format(question=question, answer=answer)
        response = llm.invoke([HumanMessage(content=prompt)])
        result = parse_ai_response(response.content)
        evaluations.append(
            {
                "id": entry.get("id"),
                "question": question,
                "answer": answer,
                "score": int(result.get("score", 0)),
                "reason": result.get("reason", ""),
            }
        )
    return evaluations


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Judge 기반 평가")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="run_eval 결과 JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="평가 결과 저장 경로")
    parser.add_argument("--limit", type=int, default=None, help="평가할 샘플 수 (기본 전체)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Judge로 사용할 GPT-5 계열 모델명")
    args = parser.parse_args()

    entries = load_dataset(args.results, args.limit)
    if not entries:
        raise ValueError("평가할 데이터가 없습니다.")

    evaluations = judge_entries(entries, args.model)
    avg_score = mean(item["score"] for item in evaluations) if evaluations else 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"average": avg_score, "results": evaluations}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"평균 점수: {avg_score:.2f}")
    print(f"저장 완료: {args.output}")


if __name__ == "__main__":
    main()
