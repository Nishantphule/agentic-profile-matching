"""
tools/candidate_comparator.py
=============================

Side-by-side candidate comparison.

Given a list of candidate IDs (or names) and the current `Requirements`, this
tool produces:

  • A deterministic comparison table (pandas DataFrame) for the UI.
  • A markdown comparison from the LLM with strengths, weaknesses, and a
    final recommendation.
"""

from __future__ import annotations

import json
from typing import Iterable, List

import pandas as pd

from agents.prompts import compare_candidates_prompt
from agents.state import CandidateMatch, Requirements
from config import get_logger
from llm_client import get_chat_llm
from tools.requirement_extractor import _content

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def select_candidates(
    candidates: List[CandidateMatch], targets: Iterable[str]
) -> List[CandidateMatch]:
    """Resolve a list of names/ids/'top N' targets to actual candidates."""
    if not candidates:
        return []
    targets = list(targets)

    # Special pseudo-targets
    if not targets:
        return candidates[:3]
    if len(targets) == 1:
        t = targets[0].lower().replace("the ", "").strip()
        if t.startswith("top"):
            try:
                n = int(t.replace("top", "").strip() or "3")
                return candidates[: max(1, n)]
            except ValueError:
                pass

    chosen: List[CandidateMatch] = []
    seen = set()
    for t in targets:
        tl = t.strip().lower()
        for c in candidates:
            if c.candidate_id in seen:
                continue
            if (
                tl == c.candidate_id.lower()
                or tl == c.name.lower()
                or tl in c.name.lower()
            ):
                chosen.append(c)
                seen.add(c.candidate_id)
                break
    return chosen or candidates[:3]


def comparison_table(
    candidates: List[CandidateMatch], requirements: Requirements
) -> pd.DataFrame:
    """Build a clean pandas table for the UI."""
    rows = []
    for c in candidates:
        rows.append(
            {
                "Candidate": c.name,
                "Score": c.score,
                "Years": c.years_experience if c.years_experience is not None else "—",
                "Must-Have Hits": ", ".join(c.must_have_hits) or "—",
                "Must-Have Gaps": ", ".join(c.must_have_misses) or "—",
                "Nice-to-Have": ", ".join(c.nice_to_have_hits) or "—",
                "Round": c.round_reached,
            }
        )
    return pd.DataFrame(rows)


def compare_candidates(
    candidates: List[CandidateMatch], requirements: Requirements
) -> str:
    """LLM-driven narrative comparison (returns markdown)."""
    if not candidates:
        return "_No candidates selected for comparison._"

    payload = [
        {
            "name": c.name,
            "score": c.score,
            "years_experience": c.years_experience,
            "must_have_hits": c.must_have_hits,
            "must_have_misses": c.must_have_misses,
            "nice_to_have_hits": c.nice_to_have_hits,
            "excerpt": c.resume_excerpt[:600],
        }
        for c in candidates
    ]

    llm = get_chat_llm(temperature=0.3, max_tokens=900)
    msgs = compare_candidates_prompt.format_messages(
        requirements=requirements.summary(),
        candidates=json.dumps(payload, indent=2),
    )
    try:
        resp = llm.invoke(msgs)
        return _content(resp).strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM comparison failed: %s", exc)
        return _fallback_comparison(candidates)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fallback_comparison(candidates: List[CandidateMatch]) -> str:
    lines = ["| Candidate | Score | Years | Must-Have Hits | Gaps |",
             "|---|---|---|---|---|"]
    for c in candidates:
        lines.append(
            f"| {c.name} | {c.score:.1f} | "
            f"{c.years_experience if c.years_experience is not None else '—'} | "
            f"{', '.join(c.must_have_hits) or '—'} | "
            f"{', '.join(c.must_have_misses) or '—'} |"
        )
    best = max(candidates, key=lambda c: c.score)
    lines.append("")
    lines.append(f"### Recommendation\n**{best.name}** has the strongest profile based on must-have coverage and overall score.")
    return "\n".join(lines)
