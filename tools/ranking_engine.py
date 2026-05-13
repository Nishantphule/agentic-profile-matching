"""
tools/ranking_engine.py
=======================

Weighted, explainable scoring of candidates.

Final score = w1 * must-have coverage
            + w2 * nice-to-have coverage
            + w3 * experience fit
            + w4 * semantic similarity (from FAISS)
            + boost / penalty adjustments from human feedback.

We also call the LLM once per candidate to produce a 3-5 sentence
human-readable explanation that's stored on the `CandidateMatch`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from agents.prompts import explain_candidate_prompt
from agents.state import CandidateMatch, HumanFeedback, Requirements
from config import get_logger
from llm_client import get_chat_llm
from tools.requirement_extractor import _content  # reuse safe content extractor

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
@dataclass
class Weights:
    must_have: float = 0.45
    nice_to_have: float = 0.15
    experience: float = 0.20
    semantic: float = 0.20
    feedback_boost: float = 0.10   # per matched boost skill
    feedback_penalty: float = 0.15  # per matched penalty skill


DEFAULT_WEIGHTS = Weights()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def rank_candidates(
    candidates: List[CandidateMatch],
    requirements: Requirements,
    feedback: Optional[HumanFeedback] = None,
    weights: Weights = DEFAULT_WEIGHTS,
    explain: bool = True,
) -> List[CandidateMatch]:
    """Score, sort, and (optionally) generate explanations for candidates."""
    if not candidates:
        return []

    semantic_min, semantic_max = _semantic_range(candidates)

    for c in candidates:
        c.score = _composite_score(c, requirements, feedback, weights,
                                   semantic_min, semantic_max)
        c.strengths, c.weaknesses = _derive_strengths_weaknesses(c, requirements)

    candidates.sort(key=lambda x: x.score, reverse=True)

    if explain:
        _attach_explanations(candidates, requirements)

    return candidates


def filter_candidates(
    candidates: List[CandidateMatch],
    feedback: Optional[HumanFeedback],
) -> List[CandidateMatch]:
    """Apply hard filters from feedback (e.g. min years override)."""
    if not feedback:
        return candidates
    out = candidates
    if feedback.min_years_override is not None:
        out = [
            c for c in out
            if (c.years_experience or 0) >= feedback.min_years_override
        ]
    return out


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def _semantic_range(candidates: List[CandidateMatch]) -> tuple[float, float]:
    """Min/max raw semantic score for normalisation."""
    scores = [c.score for c in candidates]
    if not scores:
        return 0.0, 1.0
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return lo, lo + 1e-6
    return lo, hi


def _coverage(hits: list, total: list) -> float:
    if not total:
        return 1.0  # no must-haves → no penalty
    return len(hits) / max(1, len(total))


def _experience_score(years: Optional[float], min_required: Optional[float]) -> float:
    if min_required is None or min_required <= 0:
        return 0.7 if years is None else min(1.0, years / 10.0)
    if years is None:
        return 0.3
    ratio = years / min_required
    if ratio >= 1.0:
        return min(1.0, 0.8 + 0.04 * (years - min_required))
    return max(0.0, ratio * 0.8)


def _composite_score(
    c: CandidateMatch,
    reqs: Requirements,
    feedback: Optional[HumanFeedback],
    w: Weights,
    sem_min: float,
    sem_max: float,
) -> float:
    must_cov = _coverage(c.must_have_hits, reqs.must_have_skills)
    nice_cov = _coverage(c.nice_to_have_hits, reqs.nice_to_have_skills)
    exp = _experience_score(c.years_experience, reqs.min_years_experience)
    sem_norm = (c.score - sem_min) / (sem_max - sem_min)

    score = (
        w.must_have * must_cov
        + w.nice_to_have * nice_cov
        + w.experience * exp
        + w.semantic * sem_norm
    )

    # Apply feedback shaping
    if feedback:
        excerpt_l = c.resume_excerpt.lower()
        for skill in feedback.boost_skills:
            s = skill.lower()
            if s in excerpt_l or s in (h.lower() for h in c.must_have_hits + c.nice_to_have_hits):
                score += w.feedback_boost
        for skill in feedback.penalize_skills:
            s = skill.lower()
            if s in excerpt_l:
                score -= w.feedback_penalty

    return round(max(0.0, min(1.0, score)) * 100, 2)  # 0-100 scale


def _derive_strengths_weaknesses(
    c: CandidateMatch, reqs: Requirements
) -> tuple[list, list]:
    strengths = []
    weaknesses = []

    if c.must_have_hits:
        strengths.append(
            f"Matches {len(c.must_have_hits)}/{len(reqs.must_have_skills) or len(c.must_have_hits)} must-have skills"
            + (": " + ", ".join(c.must_have_hits) if c.must_have_hits else "")
        )
    if c.nice_to_have_hits:
        strengths.append("Nice-to-have: " + ", ".join(c.nice_to_have_hits))
    if c.years_experience is not None and reqs.min_years_experience:
        if c.years_experience >= reqs.min_years_experience:
            strengths.append(f"{c.years_experience:g} yrs experience meets the {reqs.min_years_experience:g}+ bar")
        else:
            weaknesses.append(
                f"Only {c.years_experience:g} yrs vs {reqs.min_years_experience:g}+ required"
            )

    if c.must_have_misses:
        weaknesses.append("Missing must-have: " + ", ".join(c.must_have_misses))
    return strengths, weaknesses


# ---------------------------------------------------------------------------
# LLM-powered explanations
# ---------------------------------------------------------------------------
def _attach_explanations(
    candidates: List[CandidateMatch], reqs: Requirements
) -> None:
    """Generate a short narrative explanation for each candidate."""
    llm = get_chat_llm(temperature=0.3, max_tokens=400)
    summary = reqs.summary()

    for c in candidates:
        try:
            msgs = explain_candidate_prompt.format_messages(
                requirements=summary,
                resume=c.resume_excerpt[:1500],
                must_have_hits=", ".join(c.must_have_hits) or "—",
                must_have_misses=", ".join(c.must_have_misses) or "—",
                nice_to_have_hits=", ".join(c.nice_to_have_hits) or "—",
                score=c.score,
            )
            resp = llm.invoke(msgs)
            c.explanation = _content(resp).strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM explanation failed for %s: %s", c.name, exc)
            c.explanation = _fallback_explanation(c)


def _fallback_explanation(c: CandidateMatch) -> str:
    bits = [
        f"{c.name} scored {c.score:.1f}/100.",
        ("Strengths: " + "; ".join(c.strengths)) if c.strengths else "",
        ("Gaps: " + "; ".join(c.weaknesses)) if c.weaknesses else "",
    ]
    return " ".join(b for b in bits if b)
