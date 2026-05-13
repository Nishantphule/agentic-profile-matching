"""
tools/requirement_extractor.py
==============================

Extract structured requirements from a raw job description using the LLM.

Returns a typed `Requirements` Pydantic object.  The LLM is asked for strict
JSON; we add belt-and-braces parsing + a deterministic fallback so the rest
of the pipeline always receives a valid object.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from agents.prompts import extract_requirements_prompt, parse_jd_prompt
from agents.state import Requirements
from config import get_logger
from llm_client import get_chat_llm

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def summarise_jd(jd: str) -> str:
    """Return a 3-5 sentence plain-text summary of the JD."""
    llm = get_chat_llm(temperature=0.2)
    messages = parse_jd_prompt.format_messages(jd=jd.strip())
    resp = llm.invoke(messages)
    return _content(resp).strip()


def extract_requirements(jd: str) -> Requirements:
    """LLM-driven structured extraction of JD requirements."""
    llm = get_chat_llm(temperature=0.0, max_tokens=1200)
    messages = extract_requirements_prompt.format_messages(jd=jd.strip())
    resp = llm.invoke(messages)
    raw = _content(resp)
    data = _parse_json_blob(raw)
    if not data:
        log.warning("Requirement extraction returned no JSON; using fallback.")
        return _fallback_requirements(jd)
    return _coerce_requirements(data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _content(resp: Any) -> str:
    """Extract text content from a LangChain message regardless of type."""
    if resp is None:
        return ""
    content = getattr(resp, "content", resp)
    if isinstance(content, list):
        # Some providers return content as a list of parts
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _parse_json_blob(raw: str) -> Dict[str, Any]:
    """Best-effort JSON parsing.  Strips markdown fences and stray prose."""
    if not raw:
        return {}
    raw = raw.strip()
    # Strip ```json ... ``` fences
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    # If there's surrounding prose, grab the first balanced {...} blob
    if not raw.startswith("{"):
        m = _JSON_BLOCK_RE.search(raw)
        if m:
            raw = m.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning("Failed to parse requirement JSON: %s", exc)
        return {}


def _coerce_requirements(data: Dict[str, Any]) -> Requirements:
    """Defensively coerce loose LLM output into a `Requirements` instance."""

    def _as_list(v: Any) -> list:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip().lower() for x in v if str(x).strip()]
        return [str(v).strip().lower()]

    def _as_float(v: Any):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    return Requirements(
        role_title=(data.get("role_title") or None),
        seniority=(data.get("seniority") or None),
        must_have_skills=_as_list(data.get("must_have_skills")),
        nice_to_have_skills=_as_list(data.get("nice_to_have_skills")),
        min_years_experience=_as_float(data.get("min_years_experience")),
        education=_as_list(data.get("education")),
        certifications=_as_list(data.get("certifications")),
        responsibilities=[
            str(r).strip()
            for r in (data.get("responsibilities") or [])
            if str(r).strip()
        ],
    )


def _fallback_requirements(jd: str) -> Requirements:
    """Very small keyword-based fallback if the LLM call fails."""
    from vectorstore.ingest import CANONICAL_SKILLS  # local import to avoid cycle

    lowered = jd.lower()
    hits = [s for s in CANONICAL_SKILLS if s in lowered]
    return Requirements(
        role_title=None,
        seniority=None,
        must_have_skills=hits[: max(1, len(hits) // 2)],
        nice_to_have_skills=hits[len(hits) // 2 :],
        min_years_experience=None,
        education=[],
        certifications=[],
        responsibilities=[],
    )
