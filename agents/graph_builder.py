"""
agents/graph_builder.py
=======================

Compose the LangGraph workflow:

    START
    → parse_jd
    → extract_requirements
    → search_resumes
    → rank_candidates
    → generate_match_report
    → human_feedback   (interrupt for user input)
       ├── approve  → final_recommendation → END
       └── refine   → rerank → generate_match_report → human_feedback …

`build_graph()` returns a compiled `StateGraph` ready for invocation.
"""

from __future__ import annotations

from typing import Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.nodes import (
    extract_requirements_node,
    final_recommendation_node,
    generate_match_report_node,
    human_feedback_node,
    parse_jd_node,
    rerank_node,
    search_resumes_node,
    rank_candidates_node,
)
from agents.state import AgentState
from config import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------
def _after_feedback(state: AgentState) -> str:
    """Route after the human-feedback node based on what the user said."""
    stage = state.get("stage")
    if stage == "final_recommendation":
        return "final_recommendation"
    return "rerank"


def _after_search(state: AgentState) -> str:
    """Skip ranking if search produced nothing (error / empty index)."""
    if not state.get("candidates"):
        return END
    return "rank_candidates"


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------
def build_graph(
    checkpointer: Optional[MemorySaver] = None,
    *,
    interrupt_for_feedback: bool = True,
):
    """
    Build and compile the LangGraph.

    Parameters
    ----------
    checkpointer:
        Optional `MemorySaver` (or other LangGraph checkpointer).  Pass the
        one from `ConversationMemory` to enable multi-turn memory.
    interrupt_for_feedback:
        When True (default), the graph pauses BEFORE the `human_feedback`
        node so the orchestrator can collect a real user message.  Set to
        False for autonomous test runs that pre-fill the feedback in state.
    """
    log.info("Building LangGraph (interrupt=%s)", interrupt_for_feedback)
    graph = StateGraph(AgentState)

    graph.add_node("parse_jd", parse_jd_node)
    graph.add_node("extract_requirements", extract_requirements_node)
    graph.add_node("search_resumes", search_resumes_node)
    graph.add_node("rank_candidates", rank_candidates_node)
    graph.add_node("generate_match_report", generate_match_report_node)
    graph.add_node("human_feedback", human_feedback_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("final_recommendation", final_recommendation_node)

    # Linear backbone
    graph.add_edge(START, "parse_jd")
    graph.add_edge("parse_jd", "extract_requirements")
    graph.add_edge("extract_requirements", "search_resumes")

    # Conditional: empty search → END
    graph.add_conditional_edges(
        "search_resumes",
        _after_search,
        {"rank_candidates": "rank_candidates", END: END},
    )
    graph.add_edge("rank_candidates", "generate_match_report")
    graph.add_edge("generate_match_report", "human_feedback")

    # After human_feedback → either final_recommendation or rerank
    graph.add_conditional_edges(
        "human_feedback",
        _after_feedback,
        {
            "final_recommendation": "final_recommendation",
            "rerank": "rerank",
        },
    )
    # Re-rank loops back to a fresh match report → human feedback
    graph.add_edge("rerank", "generate_match_report")
    graph.add_edge("final_recommendation", END)

    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_for_feedback:
        compile_kwargs["interrupt_before"] = ["human_feedback"]

    return graph.compile(**compile_kwargs)
