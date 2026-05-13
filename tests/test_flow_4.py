"""
test_flow_4.py — Feedback-based reranking
==========================================

Scenario
--------
After seeing the first shortlist for the Senior Backend Engineer (Python) role,
the user gives qualitative feedback:

    "Re-rank based on Kafka experience and AWS certifications."

Expected behaviour
------------------
1. Agent parses the feedback into `boost_skills=['kafka']` and tags candidates
   with an AWS cert as preferred.
2. The graph re-ranks; candidates with Kafka + AWS cert (Daniel Okafor /
   Hiroshi Tanaka) move toward the top.
3. The ranking explanation references the boosted skills.

Run as:
    pytest -q tests/test_flow_4.py        (skipped unless RUN_LIVE_TESTS=1)
    python tests/test_flow_4.py            (interactive console run)
"""

from __future__ import annotations

from tests.conftest import live_only, load_jd


@live_only()
def test_flow_4_feedback_based_reranking(agent):
    agent.start(load_jd("backend_engineer_python.txt"), top_k=10)

    reply = agent.send("Re-rank based on Kafka experience and AWS certifications.")
    state = agent.state() or {}
    assert state.get("round", 1) >= 2

    # The two strongest "Kafka + AWS cert" candidates should bubble to the top.
    top_names = " ".join(c.name.lower() for c in reply.candidates[:3])
    assert ("daniel" in top_names or "hiroshi" in top_names), \
        f"Expected Daniel/Hiroshi to rise to the top after rerank, got: {top_names}"


def _demo() -> None:  # pragma: no cover
    from matching_agent import ResumeMatchingAgent
    from vectorstore.ingest import ResumeIndex
    from config import settings

    idx = ResumeIndex()
    idx.build_from_dir(settings.resumes_dir)
    idx.save()

    agent = ResumeMatchingAgent()
    print(">> Round 1:")
    print(agent.start(load_jd("backend_engineer_python.txt"), top_k=10).text)
    print("\n>> USER: Re-rank based on Kafka experience and AWS certifications.")
    reply = agent.send("Re-rank based on Kafka experience and AWS certifications.")
    print("\n>> AGENT:")
    print(reply.text)


if __name__ == "__main__":  # pragma: no cover
    _demo()
