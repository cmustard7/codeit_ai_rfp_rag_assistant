"""LangGraph DAG wiring for the baseline RAG pipeline."""

from __future__ import annotations

from langgraph.graph import StateGraph
from langgraph.pregel import Pregel

from .graph_state import GraphState
from .nodes import analyze_question, retrieve_context, answer_question, update_state


def build_workflow() -> Pregel:
    """Create and compile the StateGraph."""
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
