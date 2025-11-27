"""History summary node that keeps conversation context concise."""

from __future__ import annotations

from typing import Dict

from ..graph_state import GraphState


def update_state(state: GraphState) -> Dict[str, str]:
    """Trim the last answer so subsequent retrieval can reuse it."""
    answer = (state.get("last_answer", "") or "").strip()
    context = (state.get("context", "") or "").strip()
    if context:
        combined = f"문맥:{context} 답변:{answer}"
    else:
        combined = answer
    combined = combined.replace("\n", " ")
    if len(combined) > 500:
        combined = combined[:500]
    return {"history_summary": combined}
