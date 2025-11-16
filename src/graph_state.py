from __future__ import annotations

from typing import Dict, Optional, TypedDict


class GraphState(TypedDict, total=False):
    """LangGraph 노드 간에 전달되는 상태 딕셔너리."""

    agency: Optional[str]
    project: Optional[str]
    filters: Dict[str, str]
    history_summary: str
    last_answer: str
    context: str
    current_question: str
