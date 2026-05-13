"""
test_flow_1.py — Basic hiring query
====================================

Scenario
--------
A hiring manager pastes a Senior React Engineer JD and asks the agent to
shortlist candidates from the resume library.

Expected behaviour
------------------
1. Agent runs: parse_jd → extract_requirements → search_resumes → rank_candidates
   → generate_match_report.
2. The workflow pauses at `human_feedback` with `awaiting_feedback=True`.
3. The match report mentions at least one top candidate from the
   React-savvy resumes (Alice / Carla / Gloria / Brian).

Run as:
    pytest -q tests/test_flow_1.py        (skipped unless RUN_LIVE_TESTS=1)
    python tests/test_flow_1.py            (interactive console run)
"""

from __future__ import annotations

from tests.conftest import live_only, load_jd


@live_only()
def test_flow_1_basic_hiring_query(agent):
    jd = load_jd("senior_react_engineer.txt")
    reply = agent.start(jd, top_k=8)

    state = agent.state() or {}
    assert state.get("stage") == "human_feedback"
    assert state.get("awaiting_feedback") is True
    assert reply.candidates, "No candidates returned in round 1."
    names = [c.name.lower() for c in reply.candidates]
    assert any(n in " ".join(names) for n in ("alice", "carla", "gloria", "brian")), \
        f"Expected at least one React-savvy candidate in the shortlist: {names}"


def _demo() -> None:  # pragma: no cover
    """Interactive walkthrough.  Run with: `python tests/test_flow_1.py`."""
    from matching_agent import ResumeMatchingAgent
    from vectorstore.ingest import ResumeIndex
    from config import settings

    print(">> Building FAISS index from data/resumes ...")
    idx = ResumeIndex()
    idx.build_from_dir(settings.resumes_dir)
    idx.save()

    print(">> Sending JD ...")
    agent = ResumeMatchingAgent()
    reply = agent.start(load_jd("senior_react_engineer.txt"), top_k=8)
    print("\n=== USER ===\n(pasted Senior React Engineer JD)")
    print("\n=== AGENT ===\n")
    print(reply.text)


if __name__ == "__main__":  # pragma: no cover
    _demo()
