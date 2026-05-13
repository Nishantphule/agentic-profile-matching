"""
tools/interview_generator.py
============================

Generate technical, behavioural, and project-specific interview questions
for a candidate, tailored to the current `Requirements`.
"""

from __future__ import annotations

from typing import List, Optional

from agents.prompts import interview_questions_prompt
from agents.state import CandidateMatch, Requirements
from config import get_logger
from llm_client import get_chat_llm
from tools.requirement_extractor import _content

log = get_logger(__name__)


def generate_interview_questions(
    candidate: CandidateMatch, requirements: Requirements
) -> str:
    """Return a markdown block with grouped interview questions."""
    llm = get_chat_llm(temperature=0.4, max_tokens=900)
    msgs = interview_questions_prompt.format_messages(
        requirements=requirements.summary(),
        resume=candidate.resume_excerpt[:2000] or candidate.name,
    )
    try:
        resp = llm.invoke(msgs)
        return _content(resp).strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("Interview question generation failed: %s", exc)
        return _fallback(candidate, requirements)


def find_candidate(candidates: List[CandidateMatch], target: str) -> Optional[CandidateMatch]:
    target = (target or "").strip().lower()
    if not target:
        return None
    for c in candidates:
        if target in (c.candidate_id.lower(), c.name.lower()) or target in c.name.lower():
            return c
    return None


# ---------------------------------------------------------------------------
# Deterministic fallback (used only if the LLM call fails)
# ---------------------------------------------------------------------------
def _fallback(c: CandidateMatch, reqs: Requirements) -> str:
    must = ", ".join(reqs.must_have_skills) or "core stack"
    lines = [
        f"### Technical",
        f"1. Walk me through your most complex project using {must}.",
        f"2. Design a scalable system that uses {reqs.must_have_skills[0] if reqs.must_have_skills else 'your primary stack'}.",
        "3. Debug story: a production incident you led the resolution for.",
        "4. Trade-offs between two technologies you've used in production.",
        "5. Code review: what do you look for first?",
        "",
        "### Behavioural",
        "1. Tell me about a time you disagreed with a teammate technically.",
        "2. Describe a project that slipped — what did you learn?",
        "3. How do you balance speed and quality under deadline pressure?",
        "",
        "### Project-Specific",
        f"1. From your resume: dive deeper into the {c.name}-led project most relevant to this role.",
        "2. What metric defined success on that project and how did you measure it?",
        "3. If you re-did it today, what would you change?",
    ]
    return "\n".join(lines)
