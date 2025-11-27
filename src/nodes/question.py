"""Question analysis node for LangGraph pipeline."""

from __future__ import annotations

import re
from typing import Dict

from ..graph_state import GraphState

AGENCY_PATTERN = re.compile(r"(국민연금공단|기초과학연구원|한국[가-힣]+|[가-힣]{2,}대학교|[가-힣]{2,}도청|[가-힣]{2,}시청)")
PROJECT_PATTERN = re.compile(r"[\"'‘’“”]?([가-힣A-Za-z0-9 \\-\\(\\)\\[\\]]+사업)[\"'‘’“”]?")


def analyze_question(state: GraphState) -> Dict[str, object]:
    """질문에서 기관/사업 키워드를 추출하고 상태를 업데이트."""
    question = state.get("current_question", "")
    agency = None
    if match := AGENCY_PATTERN.search(question):
        agency = match.group(1)

    filters = dict(state.get("filters") or {})
    if agency:
        filters["agency"] = agency
    else:
        filters.pop("agency", None)

    project = state.get("project")
    if pm := PROJECT_PATTERN.search(question):
        project = pm.group(1).strip()

    return {
        "agency": agency or state.get("agency"),
        "filters": filters,
        "project": project,
        "current_question": question,
    }
