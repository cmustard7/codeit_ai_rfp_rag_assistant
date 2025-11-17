"""data_list 메타데이터를 기반으로 평가용 질문을 생성하는 스크립트."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from .data_loader import DEFAULT_CSV, DEFAULT_XLSX, TARGET_COLUMNS, load_dataframe, pick_value


def build_question_entries(row: pd.Series, base_idx: int, include_followup: bool) -> List[dict]:
    """단일 행에서 기본 질문과 선택적 후속 질문을 만들어 반환한다."""
    agency = pick_value(row, TARGET_COLUMNS["agency"]) or "해당 기관"
    project = pick_value(row, TARGET_COLUMNS["project"]) or "해당 사업"
    summary = pick_value(row, TARGET_COLUMNS["summary"])
    tags = [f"agency:{agency}"] if agency and agency != "해당 기관" else []

    questions = []
    main_question = f"{agency}이(가) 발주한 '{project}' 사업의 요구사항을 요약해 줘."
    questions.append(
        {
            "id": f"Q{base_idx:02d}",
            "question": main_question,
            "tags": tags,
            "context": summary,
        }
    )

    if include_followup:
        follow_up = f"'{project}' 사업에서 콘텐츠 개발·관리 요구 사항을 자세히 알려줘."
        questions.append(
            {
                "id": f"Q{base_idx+1:02d}",
                "question": follow_up,
                "tags": tags,
                "context": summary,
            }
        )
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description="data_list 기반 질문 생성기")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--output", type=Path, default=Path("data/questions.json"))
    parser.add_argument("--limit", type=int, default=100, help="참고할 사업 행 수")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="행을 무작위로 섞은 뒤 limit 만큼 추출",
    )
    parser.add_argument(
        "--follow-up",
        action="store_true",
        help="각 사업에 대해 후속 질문도 생성",
    )
    args = parser.parse_args()

    df = load_dataframe(args.csv, args.xlsx)
    if df.empty:
        raise ValueError("데이터가 비어 있습니다.")

    if args.shuffle:
        df = df.sample(frac=1, random_state=None).reset_index(drop=True)
    questions: List[dict] = []
    base_id = 1
    for _, row in df.head(args.limit).iterrows():
        new_entries = build_question_entries(row, base_id, args.follow_up)
        questions.extend(new_entries)
        base_id += len(new_entries)

    payload = {"questions": questions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"{args.output}에 {len(questions)}개 질문을 저장했습니다.")


if __name__ == "__main__":
    main()
