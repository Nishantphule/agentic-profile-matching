"""
test_flow_2.py — Requirement refinement
========================================

Scenario
--------
After seeing the first shortlist for the Senior React Engineer role, the
hiring manager refines the requirement:

    "Only show candidates with 5+ years of React experience."

Expected behaviour
------------------
1. Agent runs the initial flow as in test 1.
2. The user message is interpreted as feedback with
   `min_years_override=5`.
3. The graph resumes through `rerank` and `match_report`.
4. The new shortlist contains ONLY candidates with >=5 years of experience.

Run as:
    pytest -q tests/test_flow_2.py        (skipped unless RUN_LIVE_TESTS=1)
    python tests/test_flow_2.py            (interactive console run)
"""

from __future__ import annotations

from tests.conftest import live_only, load_jd


@live_only()
def test_flow_2_requirement_refinement(agent):
    agent.start(load_jd("senior_react_engineer.txt"), top_k=10)

    reply = agent.send("Only show candidates with 5+ years of experience.")
    state = agent.state() or {}

    assert state.get("round", 1) >= 2, "Expected round to advance after rerank."
    for c in reply.candidates:
        assert (c.years_experience or 0) >= 5, \
            f"{c.name} has {c.years_experience}y but should be filtered out."


def _demo() -> None:  # pragma: no cover
    from matching_agent import ResumeMatchingAgent
    from vectorstore.ingest import ResumeIndex
    from config import settings

    idx = ResumeIndex()
    idx.build_from_dir(settings.resumes_dir)
    idx.save()

    agent = ResumeMatchingAgent()
    print(">> Round 1:")
    print(agent.start(load_jd("senior_react_engineer.txt"), top_k=10).text)

    print("\n>> USER: Only show candidates with 5+ years of experience.")
    reply = agent.send("Only show candidates with 5+ years of experience.")
    print("\n>> AGENT:")
    print(reply.text)


if __name__ == "__main__":  # pragma: no cover
    _demo()
