"""
test_flow_5.py — Final hiring recommendation
=============================================

Scenario
--------
Full multi-round conversation for the ML Platform Engineer role:

    Round 1 → initial shortlist
    Round 2 → user asks to focus on LangChain / RAG experience
    Round 3 → user approves → final recommendation generated

Expected behaviour
------------------
1. After approval, the graph reaches `stage="done"`.
2. `final_recommendation` text is set on the state.
3. The "Hire:" line names Elena Rossi or Hiroshi Tanaka (the only two with
   LangChain / RAG production experience).

Run as:
    pytest -q tests/test_flow_5.py        (skipped unless RUN_LIVE_TESTS=1)
    python tests/test_flow_5.py            (interactive console run)
"""

from __future__ import annotations

from tests.conftest import live_only, load_jd


@live_only()
def test_flow_5_final_recommendation(agent):
    agent.start(load_jd("ml_platform_engineer.txt"), top_k=8)

    agent.send("Focus on candidates with LangChain or RAG production experience.")
    reply = agent.send("Approve — please finalize.")

    state = agent.state() or {}
    assert state.get("stage") == "done"
    final = state.get("final_recommendation") or reply.text
    assert "hire" in final.lower(), f"Final text missing 'hire': {final[:200]}"
    assert any(name in final.lower() for name in ("elena", "hiroshi")), \
        f"Expected Elena or Hiroshi in the final hire: {final[:300]}"


def _demo() -> None:  # pragma: no cover
    from matching_agent import ResumeMatchingAgent
    from vectorstore.ingest import ResumeIndex
    from config import settings

    idx = ResumeIndex()
    idx.build_from_dir(settings.resumes_dir)
    idx.save()

    agent = ResumeMatchingAgent()
    print(">> Round 1:")
    print(agent.start(load_jd("ml_platform_engineer.txt"), top_k=8).text)

    print("\n>> USER: Focus on candidates with LangChain or RAG production experience.")
    print(agent.send("Focus on candidates with LangChain or RAG production experience.").text)

    print("\n>> USER: Approve — please finalize.")
    print(agent.send("Approve — please finalize.").text)


if __name__ == "__main__":  # pragma: no cover
    _demo()
