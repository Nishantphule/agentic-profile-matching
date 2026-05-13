"""
tools/rag_search.py
===================

Thin wrapper around the FAISS-backed `ResumeIndex` that builds a query string
from a `Requirements` object and returns ranked `CandidateMatch` objects.
"""

from __future__ import annotations

from typing import List, Optional

from agents.state import CandidateMatch, Requirements
from config import get_logger
from vectorstore.ingest import ResumeIndex, ResumeRecord

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lazy-loaded singleton index
# ---------------------------------------------------------------------------
_INDEX: Optional[ResumeIndex] = None


def get_index() -> ResumeIndex:
    """Load (and cache) the persisted FAISS index from disk."""
    global _INDEX
    if _INDEX is None:
        _INDEX = ResumeIndex.load()
    return _INDEX


def reset_index() -> None:
    """Force the next call to `get_index()` to reload from disk."""
    global _INDEX
    _INDEX = None


# ---------------------------------------------------------------------------
# Query construction + search
# ---------------------------------------------------------------------------
def build_query(reqs: Requirements) -> str:
    """Turn a `Requirements` object into a single retrieval query string."""
    pieces = []
    if reqs.role_title:
        pieces.append(reqs.role_title)
    if reqs.seniority:
        pieces.append(f"{reqs.seniority} engineer")
    if reqs.must_have_skills:
        pieces.append("required skills: " + ", ".join(reqs.must_have_skills))
    if reqs.nice_to_have_skills:
        pieces.append("preferred skills: " + ", ".join(reqs.nice_to_have_skills))
    if reqs.min_years_experience:
        pieces.append(f"{reqs.min_years_experience}+ years experience")
    if reqs.responsibilities:
        pieces.append(" ; ".join(reqs.responsibilities[:5]))
    return ". ".join(pieces) or "experienced software engineer"


def search_candidates(
    reqs: Requirements,
    top_k: int = 10,
    index: Optional[ResumeIndex] = None,
) -> List[CandidateMatch]:
    """
    Semantic search over resumes.

    The returned `CandidateMatch` objects only have basic fields populated
    (score, hits, misses, excerpt).  `ranking_engine.rank_candidates` will
    refine the scores using deterministic skill matching + LLM explanations.
    """
    idx = index or get_index()
    if not idx.records:
        log.warning("Resume index is empty.")
        return []

    query = build_query(reqs)
    log.info("RAG query: %s", query)
    hits = idx.search(query, top_k=top_k)
    matches: List[CandidateMatch] = []
    for record, score in hits:
        matches.append(_to_match(record, reqs, semantic_score=score))
    return matches


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_match(
    record: ResumeRecord, reqs: Requirements, semantic_score: float
) -> CandidateMatch:
    must = [s.lower() for s in reqs.must_have_skills]
    nice = [s.lower() for s in reqs.nice_to_have_skills]
    skills = set(s.lower() for s in record.skills)

    must_hits = [s for s in must if s in skills]
    must_misses = [s for s in must if s not in skills]
    nice_hits = [s for s in nice if s in skills]

    excerpt = record.text[:600].replace("\n", " ").strip()
    return CandidateMatch(
        candidate_id=record.candidate_id,
        name=record.name,
        file_path=record.file_path,
        score=float(semantic_score),  # raw similarity, refined later
        must_have_hits=must_hits,
        must_have_misses=must_misses,
        nice_to_have_hits=nice_hits,
        years_experience=record.years_experience,
        resume_excerpt=excerpt,
    )
