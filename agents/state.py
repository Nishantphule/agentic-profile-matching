"""
agents/state.py
===============

Typed agent state used by every LangGraph node.

We use a `TypedDict` (with `total=False`) so LangGraph can partially update
the state, and a few Pydantic models for the strongly-typed payloads that
flow through it (requirements, candidates, rankings, etc.).
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Domain payloads
# ---------------------------------------------------------------------------
class Requirements(BaseModel):
    """Structured requirements extracted from a job description."""

    must_have_skills: List[str] = Field(default_factory=list)
    nice_to_have_skills: List[str] = Field(default_factory=list)
    min_years_experience: Optional[float] = None
    education: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    role_title: Optional[str] = None
    seniority: Optional[str] = None

    def summary(self) -> str:
        parts = [f"Role: {self.role_title or 'N/A'}"]
        if self.seniority:
            parts.append(f"Seniority: {self.seniority}")
        if self.min_years_experience is not None:
            parts.append(f"Min experience: {self.min_years_experience} yrs")
        parts.append(f"Must-have: {', '.join(self.must_have_skills) or '—'}")
        parts.append(f"Nice-to-have: {', '.join(self.nice_to_have_skills) or '—'}")
        if self.certifications:
            parts.append(f"Certifications: {', '.join(self.certifications)}")
        return "\n".join(parts)


class CandidateMatch(BaseModel):
    """One scored candidate after RAG retrieval + ranking."""

    candidate_id: str
    name: str
    file_path: Optional[str] = None
    score: float = 0.0
    must_have_hits: List[str] = Field(default_factory=list)
    must_have_misses: List[str] = Field(default_factory=list)
    nice_to_have_hits: List[str] = Field(default_factory=list)
    years_experience: Optional[float] = None
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    explanation: str = ""
    resume_excerpt: str = ""

    # Round-by-round bookkeeping (multi-round screening)
    round_reached: int = 1
    advanced: bool = True


class HumanFeedback(BaseModel):
    """User feedback that can re-shape the ranking."""

    instruction: str
    boost_skills: List[str] = Field(default_factory=list)
    penalize_skills: List[str] = Field(default_factory=list)
    min_years_override: Optional[float] = None
    approve: bool = False  # True → skip back to finalization


ScreeningStage = Literal[
    "idle",
    "parse_jd",
    "extract_requirements",
    "search_resumes",
    "rank_candidates",
    "match_report",
    "human_feedback",
    "rerank",
    "final_recommendation",
    "done",
]


# ---------------------------------------------------------------------------
# Top-level LangGraph state
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    """
    Mutable state that flows between LangGraph nodes.

    `messages` uses LangGraph's built-in `add_messages` reducer so each node
    can simply *return* the new messages it produced and they will be
    appended.  All other keys use the default "replace" reducer.
    """

    # --- conversation ---
    messages: Annotated[List[BaseMessage], add_messages]

    # --- inputs / parsed JD ---
    current_jd: str
    jd_summary: str
    requirements: Requirements

    # --- candidates ---
    candidates: List[CandidateMatch]   # current shortlist (post-ranking)
    all_candidates: List[CandidateMatch]  # full pool before filtering

    # --- explanations / report ---
    match_report: str
    ranking_explanations: Dict[str, str]

    # --- human-in-the-loop ---
    human_feedback: Optional[HumanFeedback]
    feedback_history: List[Dict[str, Any]]
    awaiting_feedback: bool

    # --- workflow control ---
    stage: ScreeningStage
    round: int                     # 1, 2, or 3
    top_k: int                     # how many to retrieve from RAG
    final_recommendation: str
    error: Optional[str]
