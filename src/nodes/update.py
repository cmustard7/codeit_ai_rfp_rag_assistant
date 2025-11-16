"""History summary node that keeps conversation context concise."""

from __future__ import annotations

from typing import Dict

from ..graph_state import GraphState


def update_state(state: GraphState) -> Dict[str, str]:
    """Trim the last answer so subsequent retrieval can reuse it."""
    summary = (state.get("last_answer", "") or "").strip().replace("\n", " ")
    if len(summary) > 300:
        summary = summary[:300]
    return {"history_summary": summary}
