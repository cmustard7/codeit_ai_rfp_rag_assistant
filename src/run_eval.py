"""질문 집합에 LangGraph 워크플로를 적용해 일괄 평가하는 스크립트."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .graph_state import GraphState
from .workflow import build_workflow

DEFAULT_QUESTIONS_PATH = Path("data/questions.json")
DEFAULT_OUTPUT_PATH = Path("data/results.json")
LOG_KEYS = [
    "EMBED_PROVIDER",
    "LANGGRAPH_EMBED_MODEL",
    "HF_EMBED_MODEL",
    "ENABLE_MMR",
    "MMR_LAMBDA",
    "ENABLE_BM25",
    "BM25_TOP_K",
    "ENABLE_PARAPHRASE",
    "PARAPHRASE_N",
    "ENABLE_RERANK",
    "RERANK_MODEL",
    "RERANK_SCORE_FLOOR",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "RETRIEVAL_TOP_K",
]


def load_questions(path: Path):
    """질문 JSON 파일에서 질문 목록을 읽어 반환한다."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("questions", [])


def parse_arguments() -> argparse.Namespace:
    """CLI 인자를 파싱해 외부에서 재사용하기 쉽게 반환한다."""
    parser = argparse.ArgumentParser(description="LangGraph RAG 평가")
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="질문 JSON 경로 (기본: data/questions.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="결과 JSON 경로 (기본: data/results.json)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases 프로젝트명 (미지정 시 비활성화)",
    )
    parser.add_argument(
        "--wandb-run",
        type=str,
        default=None,
        help="Weights & Biases 런 이름",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    questions_path = args.questions or DEFAULT_QUESTIONS_PATH
    output_path = args.output or DEFAULT_OUTPUT_PATH

    load_dotenv()  # .env 파일에서 OPENAI_API_KEY 등 환경 변수 로드

    questions = load_questions(questions_path)
    workflow = build_workflow()
    state: GraphState = {
        "agency": None,
        "project": None,
        "filters": {},
        "history_summary": "",
        "last_answer": "",
        "context": "",
        "current_question": "",
        "usage": {},
        "retrieval_config": {},
    }
    results = []
    latencies = []

    wandb_run: Optional[object] = None
    if args.wandb_project:
        try:
            import wandb

            config_env = {k: os.getenv(k) for k in LOG_KEYS if os.getenv(k) is not None}
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run,
                config={
                    "question_count": len(questions),
                    **config_env,
                },
            )
        except Exception as exc:  # pragma: no cover
            print(f"wandb 초기화 실패: {exc}")

    for entry in questions:
        question = entry.get("question", "").strip()
        if not question:
            continue
        state["current_question"] = question
        start = time.perf_counter()
        state = workflow.invoke(state)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        usage = state.get("usage", {}) or {}
        retrieval_cfg = state.get("retrieval_config", {}) or {}
        results.append({
            "id": entry.get("id"),
            "question": question,
            "answer": state.get("last_answer", ""),
            "context": state.get("context", ""),
            "retrieved_docs": state.get("retrieved_docs", []),
            "latency_ms": latency_ms,
            "token_usage": usage,
            "retrieval_config": retrieval_cfg,
        })
        if wandb_run:
            wandb_run.log(
                {
                    "question_id": entry.get("id"),
                    "latency_ms": latency_ms,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "vector_hits": retrieval_cfg.get("vector_results", 0),
                    "metadata_hits": retrieval_cfg.get("metadata_results", 0),
                }
            )

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"저장 완료: {output_path}")

    if wandb_run:
        import wandb

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        wandb_run.summary.update({
            "avg_latency_ms": avg_latency,
            "total_questions": len(results),
        })
        wandb_run.finish()


if __name__ == "__main__":
    main()
