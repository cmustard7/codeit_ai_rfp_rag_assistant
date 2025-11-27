"""data_list 메타데이터를 기반으로 평가용 질문을 생성하는 스크립트."""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .data_loader import DEFAULT_CSV, DEFAULT_XLSX, TARGET_COLUMNS, load_dataframe, pick_value

FOLLOW_UP_TEMPLATES = [
    {
        "keywords": ["콘텐츠", "학습", "교육"],
        "template": "'{project}' 사업에서 콘텐츠/학습 관리 요구 사항을 구체적으로 알려 줘.",
    },
    {
        "keywords": ["ai", "인공지능"],
        "template": "'{project}' 사업에서 AI 기능이나 예측 모듈 요구 사항을 정리해 줘.",
    },
    {
        "keywords": ["홍수", "감시", "관제"],
        "template": "'{project}' 사업에서 모니터링·관제 기능 요구 사항을 자세히 설명해 줘.",
    },
    {
        "keywords": ["예산", "비용"],
        "template": "'{project}' 사업의 예산·사업 기간·투입 계획에 대해 자세히 알려 줘.",
    },
]
DEFAULT_FOLLOWUP = "'{project}' 사업에서 추가로 확인해야 할 핵심 요구 사항을 알려 줘."

FAKE_TEMPLATES = [
    "이 사업이 자율주행차 플랫폼과 연동되는 요구 사항을 포함하고 있는지 확인해 줘.",
    "해당 사업에 우주 탐사 모듈이나 위성 통신 요구 사항이 있는지 찾아 줘.",
    "'{project}' 사업이 바이오센서 하드웨어 제조에 대한 구체 요구를 담고 있는가?",
    "이 사업에 양자컴퓨팅이나 양자암호 관련 요구 사항이 있는지 알려 줘.",
]


def detect_followup(summary: str, project: str) -> str:
    text = (summary or "").lower()
    for template in FOLLOW_UP_TEMPLATES:
        if any(keyword in text for keyword in template["keywords"]):
            return template["template"].format(project=project)
    return DEFAULT_FOLLOWUP.format(project=project)


def build_question_entries(
    row: pd.Series,
    base_idx: int,
    include_followup: bool,
    fake_rate: float,
) -> List[dict]:
    """단일 행에서 기본 질문과 선택적 후속/가짜 질문을 만들어 반환한다."""
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
            "tags": tags + ["type:primary"],
            "context": summary,
            "answerable": True,
            "is_fake": False,
        }
    )

    idx_increment = 1
    if include_followup:
        follow_up = detect_followup(summary or "", project)
        questions.append(
            {
                "id": f"Q{base_idx+idx_increment:02d}",
                "question": follow_up,
                "tags": tags + ["type:followup"],
                "context": summary,
                "answerable": True,
                "is_fake": False,
            }
        )
        idx_increment += 1

    if fake_rate > 0 and random.random() < fake_rate:
        fake_template = random.choice(FAKE_TEMPLATES)
        fake_question = fake_template.format(project=project, agency=agency)
        questions.append(
            {
                "id": f"Q{base_idx+idx_increment:02d}",
                "question": fake_question,
                "tags": tags + ["type:fake"],
                "context": summary,
                "answerable": False,
                "is_fake": True,
            }
        )
        idx_increment += 1

    return questions, idx_increment


def build_compare_question(row_a: pd.Series, row_b: pd.Series, qid: str) -> dict:
    agency_a = pick_value(row_a, TARGET_COLUMNS["agency"]) or "A기관"
    project_a = pick_value(row_a, TARGET_COLUMNS["project"]) or "A사업"
    agency_b = pick_value(row_b, TARGET_COLUMNS["agency"]) or "B기관"
    project_b = pick_value(row_b, TARGET_COLUMNS["project"]) or "B사업"
    summary = f"{project_a} vs {project_b}"
    question = (
        f"{agency_a}의 '{project_a}' 사업과 {agency_b}의 '{project_b}' 사업을 비교해서 "
        f"요구사항과 차이점을 설명해 줘."
    )
    return {
        "id": qid,
        "question": question,
        "tags": [f"agency:{agency_a}", f"agency:{agency_b}", "type:compare"],
        "context": summary,
        "answerable": True,
        "is_fake": False,
    }


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
    parser.add_argument(
        "--fake-rate",
        type=float,
        default=0.0,
        help="가짜(존재하지 않을 가능성이 높은) 후속 질문을 추가할 확률",
    )
    parser.add_argument(
        "--compare-rate",
        type=float,
        default=0.0,
        help="추가로 생성할 비교 질문 비율 (0~1)",
    )
    args = parser.parse_args()

    df = load_dataframe(args.csv, args.xlsx)
    if df.empty:
        raise ValueError("데이터가 비어 있습니다.")

    if args.shuffle:
        df = df.sample(frac=1, random_state=None).reset_index(drop=True)
    selected_dicts = df.head(args.limit).to_dict(orient="records")
    selected_series: List[pd.Series] = [pd.Series(row_dict) for row_dict in selected_dicts]

    questions: List[dict] = []
    base_id = 1
    for row in selected_series:
        new_entries, increment = build_question_entries(
            row,
            base_idx=base_id,
            include_followup=args.follow_up,
            fake_rate=max(0.0, min(args.fake_rate, 1.0)),
        )
        questions.extend(new_entries)
        base_id += increment

    compare_count = 0
    if args.compare_rate > 0 and len(selected_series) >= 2:
        pair_total = max(1, int(len(selected_series) * min(args.compare_rate, 1.0)))
        for _ in range(pair_total):
            row_a, row_b = random.sample(selected_series, 2)
            questions.append(
                build_compare_question(row_a, row_b, qid=f"Q{base_id:02d}")
            )
            base_id += 1
            compare_count += 1

    payload = {"questions": questions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"{args.output}에 {len(questions)}개 질문을 저장했습니다.")
    if compare_count:
        print(f"비교 질문 {compare_count}개 추가 생성")


if __name__ == "__main__":
    main()
