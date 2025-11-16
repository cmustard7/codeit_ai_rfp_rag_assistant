"""Interactive CLI shell for LangGraph RAG."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from .graph_state import GraphState
from .workflow import build_workflow


def init_state() -> GraphState:
    """Create a clean state so multiple chat sessions behave consistently."""
    return {
        "agency": None,
        "project": None,
        "filters": {},
        "history_summary": "",
        "last_answer": "",
        "context": "",
        "current_question": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph 인터랙티브 질의")
    parser.add_argument("--reset", action="store_true", help="첫 질문 전 초기 상태를 새로 생성")
    args = parser.parse_args()

    load_dotenv()
    workflow = build_workflow()
    state = init_state()

    print("LangGraph RAG 인터랙티브 모드입니다. 종료하려면 'exit' 또는 'quit' 입력.")
    while True:
        try:
            question = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("종료합니다.")
            break
        if args.reset:
            state = init_state()

        state["current_question"] = question
        state = workflow.invoke(state)
        answer = state.get("last_answer", "(응답 없음)")
        print(f"A> {answer}\n")


if __name__ == "__main__":
    main()
