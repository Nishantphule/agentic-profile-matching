"""
agents/nodes.py
===============

Pure functions that implement each LangGraph node.

Each node receives the current `AgentState` and returns a *partial* state
dictionary; LangGraph merges those updates back into the global state.

Nodes:
  • parse_jd_node                — summarise the JD
  • extract_requirements_node    — structured requirement extraction
  • search_resumes_node          — RAG retrieval over the resume index
  • rank_candidates_node         — weighted scoring + explanations
  • generate_match_report_node   — markdown shortlist report
  • human_feedback_node          — interrupts the graph to ask the user
  • rerank_node                  — apply feedback, re-score, re-sort
  • final_recommendation_node    — final hire/no-hire summary
"""

from __future__ import annotations

import json
from typing import Dict, List

from langchain_core.messages import AIMessage

from agents.prompts import final_recommendation_prompt, match_report_prompt
from agents.state import (
    AgentState,
    CandidateMatch,
    HumanFeedback,
    Requirements,
)
from config import get_logger
from llm_client import get_chat_llm
from tools.candidate_comparator import comparison_table
from tools.rag_search import search_candidates
from tools.ranking_engine import filter_candidates, rank_candidates
from tools.requirement_extractor import (
    _content,
    extract_requirements,
    summarise_jd,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1) Parse JD
# ---------------------------------------------------------------------------
def parse_jd_node(state: AgentState) -> Dict:
    jd = (state.get("current_jd") or "").strip()
    if not jd:
        return {
            "stage": "parse_jd",
            "error": "No job description provided.",
            "messages": [AIMessage(content="❌ Please provide a job description first.")],
        }
    log.info("[node] parse_jd")
    try:
        summary = summarise_jd(jd)
    except Exception as exc:  # noqa: BLE001
        log.exception("parse_jd failed")
        return {"stage": "parse_jd", "error": f"parse_jd failed: {exc}"}
    return {
        "jd_summary": summary,
        "stage": "extract_requirements",
        "messages": [AIMessage(content=f"**Parsed JD**\n\n{summary}")],
    }


# ---------------------------------------------------------------------------
# 2) Extract Requirements
# ---------------------------------------------------------------------------
def extract_requirements_node(state: AgentState) -> Dict:
    log.info("[node] extract_requirements")
    jd = state.get("current_jd", "")
    try:
        reqs: Requirements = extract_requirements(jd)
    except Exception as exc:  # noqa: BLE001
        log.exception("extract_requirements failed")
        return {"stage": "extract_requirements", "error": f"extract failed: {exc}"}

    pretty = reqs.summary()
    return {
        "requirements": reqs,
        "stage": "search_resumes",
        "messages": [AIMessage(content=f"**Requirements**\n```\n{pretty}\n```")],
    }


# ---------------------------------------------------------------------------
# 3) Search Resumes (RAG)
# ---------------------------------------------------------------------------
def search_resumes_node(state: AgentState) -> Dict:
    log.info("[node] search_resumes (round %s)", state.get("round", 1))
    reqs: Requirements = state.get("requirements")  # type: ignore[assignment]
    if not reqs:
        return {"stage": "search_resumes", "error": "Missing requirements."}

    top_k = int(state.get("top_k", 10))
    try:
        candidates = search_candidates(reqs, top_k=top_k)
    except FileNotFoundError as exc:
        return {
            "stage": "search_resumes",
            "error": str(exc),
            "messages": [AIMessage(content=f"❌ {exc}")],
        }

    if not candidates:
        return {
            "stage": "search_resumes",
            "candidates": [],
            "all_candidates": [],
            "messages": [AIMessage(content="No candidates found.  Add resumes to `data/resumes/` and re-ingest.")],
        }

    return {
        "candidates": candidates,
        "all_candidates": candidates,
        "stage": "rank_candidates",
        "messages": [AIMessage(content=f"🔎 Retrieved {len(candidates)} candidate(s).")],
    }


# ---------------------------------------------------------------------------
# 4) Rank Candidates
# ---------------------------------------------------------------------------
def rank_candidates_node(state: AgentState) -> Dict:
    log.info("[node] rank_candidates")
    candidates: List[CandidateMatch] = state.get("candidates", []) or []
    reqs: Requirements = state.get("requirements")  # type: ignore[assignment]
    feedback = state.get("human_feedback")

    candidates = filter_candidates(candidates, feedback)
    candidates = rank_candidates(candidates, reqs, feedback=feedback, explain=True)

    # Multi-round screening: round 1 = top 10 by default, round 2 = top 5,
    # round 3 = top 3.  Stamp the round each candidate has reached.
    rnd = int(state.get("round", 1))
    keep = {1: 10, 2: 5, 3: 3}.get(rnd, 10)
    surviving = candidates[:keep]
    for c in candidates:
        c.round_reached = rnd
        c.advanced = c in surviving

    explanations = {c.name: c.explanation for c in surviving}
    return {
        "candidates": surviving,
        "ranking_explanations": explanations,
        "stage": "match_report",
    }


# ---------------------------------------------------------------------------
# 5) Match Report
# ---------------------------------------------------------------------------
def generate_match_report_node(state: AgentState) -> Dict:
    log.info("[node] match_report")
    reqs: Requirements = state.get("requirements")  # type: ignore[assignment]
    candidates: List[CandidateMatch] = state.get("candidates", []) or []
    rnd = int(state.get("round", 1))

    shortlist_payload = [
        {
            "name": c.name,
            "score": c.score,
            "years_experience": c.years_experience,
            "must_have_hits": c.must_have_hits,
            "must_have_misses": c.must_have_misses,
            "nice_to_have_hits": c.nice_to_have_hits,
            "strengths": c.strengths,
            "weaknesses": c.weaknesses,
        }
        for c in candidates
    ]

    llm = get_chat_llm(temperature=0.3, max_tokens=800)
    msgs = match_report_prompt.format_messages(
        requirements=reqs.summary() if reqs else "—",
        shortlist=json.dumps(shortlist_payload, indent=2),
        round=rnd,
    )
    try:
        resp = llm.invoke(msgs)
        report = _content(resp).strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("Match report LLM failed: %s", exc)
        report = _fallback_report(candidates)

    table = comparison_table(candidates, reqs) if reqs else None
    msg = report
    if table is not None and not table.empty:
        msg += "\n\n**Shortlist Table**\n\n" + _df_to_markdown(table)

    return {
        "match_report": report,
        "stage": "human_feedback",
        "awaiting_feedback": True,
        "messages": [AIMessage(content=msg)],
    }


def _df_to_markdown(df) -> str:
    """Render a pandas DataFrame as a GitHub-flavoured markdown table.

    We hand-build the table instead of calling `df.to_markdown(...)` so the
    app keeps working without the optional `tabulate` dependency.
    """
    if df is None or df.empty:
        return ""
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            s = "" if v is None else str(v)
            # Escape pipe and newlines so the table doesn't break
            s = s.replace("|", "\\|").replace("\n", " ")
            cells.append(s)
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def _fallback_report(candidates: List[CandidateMatch]) -> str:
    if not candidates:
        return "_No shortlist._"
    lines = ["## Top Candidates"]
    for i, c in enumerate(candidates[:5], 1):
        lines.append(f"{i}. **{c.name}** — score {c.score:.1f}; "
                     + (c.strengths[0] if c.strengths else "matches several requirements"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6) Human feedback (no-op node — the graph interrupts BEFORE this node).
# ---------------------------------------------------------------------------
def human_feedback_node(state: AgentState) -> Dict:
    """
    The graph is configured with `interrupt_before=["human_feedback"]` so the
    runner pauses here for user input.  Once the orchestrator resumes the
    graph, this node simply ushers the state forward based on the feedback
    that the orchestrator has placed into `state["human_feedback"]`.
    """
    log.info("[node] human_feedback")
    feedback: HumanFeedback | None = state.get("human_feedback")
    if feedback is None:
        # Resumed without explicit feedback → treat as approval.
        return {"stage": "final_recommendation", "awaiting_feedback": False}

    history = list(state.get("feedback_history") or [])
    history.append(
        {
            "round": int(state.get("round", 1)),
            "instruction": feedback.instruction,
            "boost_skills": feedback.boost_skills,
            "penalize_skills": feedback.penalize_skills,
            "min_years_override": feedback.min_years_override,
            "approve": feedback.approve,
        }
    )

    if feedback.approve:
        return {
            "feedback_history": history,
            "stage": "final_recommendation",
            "awaiting_feedback": False,
        }

    return {
        "feedback_history": history,
        "stage": "rerank",
        "awaiting_feedback": False,
    }


# ---------------------------------------------------------------------------
# 7) Re-rank (after feedback)
# ---------------------------------------------------------------------------
def rerank_node(state: AgentState) -> Dict:
    log.info("[node] rerank")
    reqs: Requirements = state.get("requirements")  # type: ignore[assignment]
    feedback = state.get("human_feedback")

    # Start from the *full* pool again so feedback can resurrect candidates
    # that were trimmed previously.
    pool = list(state.get("all_candidates") or state.get("candidates") or [])

    pool = filter_candidates(pool, feedback)
    pool = rank_candidates(pool, reqs, feedback=feedback, explain=True)

    new_round = int(state.get("round", 1)) + 1
    keep = {2: 5, 3: 3}.get(new_round, 3)
    surviving = pool[:keep]

    for c in pool:
        c.round_reached = new_round
        c.advanced = c in surviving

    return {
        "round": new_round,
        "candidates": surviving,
        "ranking_explanations": {c.name: c.explanation for c in surviving},
        "stage": "match_report",
        # Drop the consumed feedback so the next pass starts clean
        "human_feedback": None,
        "messages": [AIMessage(content=f"♻️ Re-ranked for round {new_round} ({len(surviving)} candidate(s)).")],
    }


# ---------------------------------------------------------------------------
# 8) Final Recommendation
# ---------------------------------------------------------------------------
def final_recommendation_node(state: AgentState) -> Dict:
    log.info("[node] final_recommendation")
    reqs: Requirements = state.get("requirements")  # type: ignore[assignment]
    candidates: List[CandidateMatch] = state.get("candidates", []) or []
    history = state.get("feedback_history") or []

    shortlist_payload = [
        {
            "name": c.name,
            "score": c.score,
            "years_experience": c.years_experience,
            "strengths": c.strengths,
            "weaknesses": c.weaknesses,
            "explanation": c.explanation,
        }
        for c in candidates[:5]
    ]
    llm = get_chat_llm(temperature=0.2, max_tokens=900)
    msgs = final_recommendation_prompt.format_messages(
        requirements=reqs.summary() if reqs else "—",
        shortlist=json.dumps(shortlist_payload, indent=2),
        feedback_history=json.dumps(history, indent=2),
    )
    try:
        resp = llm.invoke(msgs)
        text = _content(resp).strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("Final recommendation LLM failed: %s", exc)
        text = _fallback_final(candidates)

    return {
        "final_recommendation": text,
        "stage": "done",
        "messages": [AIMessage(content=text)],
    }


def _fallback_final(candidates: List[CandidateMatch]) -> str:
    if not candidates:
        return "### Final Recommendation\n_No suitable candidates._"
    top = candidates[0]
    backups = candidates[1:3]
    return (
        "### Final Recommendation\n"
        f"**Hire:** {top.name} — strongest must-have coverage (score {top.score:.1f}).\n"
        f"**Strong backups:** {', '.join(b.name for b in backups) or 'None'}\n\n"
        "### Reasoning\n"
        f"{top.explanation or 'Best overall fit on the criteria provided.'}\n\n"
        "### Next Steps\n- Schedule on-site interview\n- Reference checks"
    )
