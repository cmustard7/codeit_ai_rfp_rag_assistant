"""Retrieval 결과를 정량 평가하는 스크립트."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Set


def load_results(results_path: Path) -> List[dict]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    return data.get("results", [])


def load_gold(gold_path: Path) -> Dict[str, dict]:
    data = json.loads(gold_path.read_text(encoding="utf-8-sig"))
    entries = data.get("gold") or data
    mapping: Dict[str, dict] = {}
    for entry in entries:
        qid = entry.get("id")
        if not qid:
            continue
        mapping[qid] = {
            "expected_files": set(entry.get("expected_files", [])),
            "keywords": entry.get("keywords", []),
        }
    return mapping


def evaluate(results: List[dict], gold: Dict[str, dict]):
    per_question = []
    for row in results:
        qid = row.get("id")
        gold_entry = gold.get(qid)
        if not gold_entry:
            continue
        retrieved_docs = row.get("retrieved_docs") or []
        retrieved_files: Set[str] = {
            doc.get("file_name", "").strip() for doc in retrieved_docs if doc.get("file_name")
        }
        expected_files: Set[str] = set(gold_entry["expected_files"])
        true_positive = len(retrieved_files & expected_files)
        precision = true_positive / len(retrieved_files) if retrieved_files else 0.0
        recall = true_positive / len(expected_files) if expected_files else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        keywords = gold_entry.get("keywords", [])
        context = row.get("context", "").lower()
        keyword_hits = (
            sum(1 for kw in keywords if kw.lower() in context) / len(keywords)
            if keywords
            else None
        )

        per_question.append(
            {
                "id": qid,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "keyword_hit_ratio": keyword_hits,
                "retrieved_files": list(retrieved_files),
                "expected_files": list(expected_files),
            }
        )
    if not per_question:
        raise ValueError("평가 가능한 질문이 없습니다. gold 매핑을 확인하세요.")
    summary = {
        "avg_precision": mean(item["precision"] for item in per_question),
        "avg_recall": mean(item["recall"] for item in per_question),
        "avg_f1": mean(item["f1"] for item in per_question),
    }
    keyword_ratios = [item["keyword_hit_ratio"] for item in per_question if item["keyword_hit_ratio"] is not None]
    if keyword_ratios:
        summary["avg_keyword_hit_ratio"] = mean(keyword_ratios)
    return summary, per_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval 정량 평가")
    parser.add_argument("--results", type=Path, default=Path("data/results.json"), help="run_eval 결과 JSON")
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("data/gold_targets.json"),
        help="정답 파일 매핑 JSON (기본: data/gold_targets.json)",
    )
    parser.add_argument("--output", type=Path, default=Path("data/eval/retrieval_scores.json"), help="평가 결과 저장 경로")
    args = parser.parse_args()

    results = load_results(args.results)
    gold = load_gold(args.gold)
    summary, details = evaluate(results, gold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "details": details}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("=== Retrieval Scores ===")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")
    print(f"저장 완료: {args.output}")


if __name__ == "__main__":
    main()
