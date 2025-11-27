"""베이스라인 RAG 파이프라인을 LangGraph DAG로 구성한다."""

from __future__ import annotations

from langgraph.graph import StateGraph
from langgraph.pregel import Pregel

from .graph_state import GraphState
from .nodes import analyze_question, retrieve_context, answer_question, update_state


def build_workflow() -> Pregel:
    """StateGraph를 구성하고 컴파일한 Pregel 객체를 반환한다."""
    graph = StateGraph(GraphState)
    graph.add_node("question", analyze_question)
    graph.add_node("retrieve", retrieve_context)
    graph.add_node("answer", answer_question)
    graph.add_node("update", update_state)

    graph.set_entry_point("question")
    graph.add_edge("question", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "update")

    return graph.compile()
