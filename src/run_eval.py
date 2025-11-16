"""Batch evaluator that runs the LangGraph workflow over a question set."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from .graph_state import GraphState
from .workflow import build_workflow

DEFAULT_QUESTIONS_PATH = Path("data/questions.json")
DEFAULT_OUTPUT_PATH = Path("data/results.json")


def load_questions(path: Path):
    """Load question entries from the given JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("questions", [])


def parse_arguments() -> argparse.Namespace:
    """Return parsed CLI arguments so the script can be imported easily."""
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
    }
    results = []

    for entry in questions:
        question = entry.get("question", "").strip()
        if not question:
            continue
        state["current_question"] = question
        state = workflow.invoke(state)
        results.append({
            "id": entry.get("id"),
            "question": question,
            "answer": state.get("last_answer", ""),
        })

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


if __name__ == "__main__":
    main()
