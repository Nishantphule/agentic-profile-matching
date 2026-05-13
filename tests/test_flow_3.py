"""
test_flow_3.py — Candidate comparison
======================================

Scenario
--------
After the initial shortlist for the Senior React Engineer role, the hiring
manager asks the agent to compare the top 3 candidates side-by-side.

Expected behaviour
------------------
1. Agent classifies the request as `intent="compare"`.
2. A comparison table is built deterministically (pandas) and embedded in
   the reply.
3. The LLM produces a recommendation calling out the strongest candidate.

Run as:
    pytest -q tests/test_flow_3.py        (skipped unless RUN_LIVE_TESTS=1)
    python tests/test_flow_3.py            (interactive console run)
"""

from __future__ import annotations

from tests.conftest import live_only, load_jd


@live_only()
def test_flow_3_candidate_comparison(agent):
    agent.start(load_jd("senior_react_engineer.txt"), top_k=8)

    reply = agent.send("Compare the top 3 candidates side-by-side.")
    assert reply.kind == "comparison"
    assert reply.candidates and len(reply.candidates) >= 2
    assert "table" in reply.extras
    assert "recommendation" in reply.text.lower() or "verdict" in reply.text.lower()


def _demo() -> None:  # pragma: no cover
    from matching_agent import ResumeMatchingAgent
    from vectorstore.ingest import ResumeIndex
    from config import settings

    idx = ResumeIndex()
    idx.build_from_dir(settings.resumes_dir)
    idx.save()

    agent = ResumeMatchingAgent()
    print(">> Round 1:")
    print(agent.start(load_jd("senior_react_engineer.txt"), top_k=8).text)
    print("\n>> USER: Compare the top 3 candidates side-by-side.")
    print("\n>> AGENT:")
    print(agent.send("Compare the top 3 candidates side-by-side.").text)


if __name__ == "__main__":  # pragma: no cover
    _demo()
