"""
matching_agent.py
=================

High-level orchestrator that the UI and tests interact with.

The LangGraph workflow handles the heavy lifting (parse JD → extract
requirements → RAG → rank → report → human feedback → rerank → final).
This file wraps that graph in a small conversational class:

    agent = ResumeMatchingAgent()
    agent.start(jd_text="...")          # kicks the graph from START → human_feedback
    agent.send("compare top 3")         # interprets user message + runs a tool
    agent.send("approve")               # closes the loop → final recommendation

Conversation memory is persisted via LangGraph's `MemorySaver` (one
`thread_id` per session) so the agent can be paused and resumed.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage

from agents.graph_builder import build_graph
from agents.memory import ConversationMemory
from agents.prompts import router_prompt
from agents.state import (
    AgentState,
    CandidateMatch,
    HumanFeedback,
    Requirements,
)
from config import get_logger
from llm_client import get_chat_llm
from tools.candidate_comparator import (
    compare_candidates,
    comparison_table,
    select_candidates,
)
from tools.interview_generator import find_candidate, generate_interview_questions
from tools.requirement_extractor import _content

log = get_logger(__name__)


# Type alias for the progress callback signature used by the UI / tests.
# Receives one event per LangGraph node as it finishes executing:
#   {"node": "<node_name>", "stage": "completed"}
ProgressCallback = Callable[[Dict[str, Any]], None]


# Maps internal node names to human labels + a 0..1 progress fraction.  The
# UI uses this to drive a `st.progress(...)` bar.
NODE_PROGRESS: Dict[str, Dict[str, Any]] = {
    "parse_jd":              {"label": "Parsing job description",     "fraction": 0.15},
    "extract_requirements":  {"label": "Extracting requirements",     "fraction": 0.30},
    "search_resumes":        {"label": "Searching resumes (RAG)",     "fraction": 0.55},
    "rank_candidates":       {"label": "Ranking candidates",          "fraction": 0.75},
    "generate_match_report": {"label": "Generating match report",     "fraction": 0.95},
    "human_feedback":        {"label": "Awaiting feedback",           "fraction": 1.00},
    "rerank":                {"label": "Re-ranking with feedback",    "fraction": 0.60},
    "final_recommendation":  {"label": "Building final recommendation","fraction": 0.95},
}


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------
class AgentReply:
    """Container for agent responses returned to the UI / tests."""

    def __init__(
        self,
        text: str,
        candidates: Optional[List[CandidateMatch]] = None,
        stage: str = "",
        kind: str = "chat",
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.text = text
        self.candidates = candidates or []
        self.stage = stage
        self.kind = kind  # chat | report | comparison | interview | final
        self.extras = extras or {}

    def __repr__(self) -> str:  # pragma: no cover
        return f"<AgentReply kind={self.kind} stage={self.stage} text={self.text[:60]!r}>"


class ResumeMatchingAgent:
    """Conversational façade over the LangGraph workflow."""

    def __init__(self, thread_id: Optional[str] = None) -> None:
        self.memory = ConversationMemory()
        self.graph = build_graph(
            checkpointer=self.memory.checkpointer, interrupt_for_feedback=True
        )
        self.thread_id = thread_id or f"thread-{uuid.uuid4().hex[:8]}"
        self._started = False

    # -------------------------------------------------------------- helpers
    @property
    def config(self) -> dict:
        return self.memory.config_for(self.thread_id)

    def state(self) -> Optional[AgentState]:
        return self.memory.state(self.graph, self.thread_id)  # type: ignore[return-value]

    def reset(self) -> None:
        """Start a fresh thread (forgets all prior messages)."""
        self.thread_id = f"thread-{uuid.uuid4().hex[:8]}"
        self._started = False

    # ---------------------------------------------------------------- start
    def start(
        self,
        jd_text: str,
        top_k: int = 10,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> AgentReply:
        """
        Kick the graph from START.  The graph runs until it pauses before
        `human_feedback` (after producing the first match report).

        Parameters
        ----------
        jd_text:
            Raw job description text.
        top_k:
            Number of candidates to retrieve in round 1.
        progress_callback:
            Optional callable invoked once per LangGraph node as it completes
            (used by the Streamlit UI to drive a progress bar).
        """
        if not jd_text.strip():
            return AgentReply("Please provide a job description.", stage="idle")

        initial: AgentState = {
            "current_jd": jd_text.strip(),
            "messages": [HumanMessage(content="(new job description)\n\n" + jd_text.strip())],
            "stage": "parse_jd",
            "round": 1,
            "top_k": top_k,
            "feedback_history": [],
            "awaiting_feedback": False,
        }
        log.info("Starting agent thread=%s", self.thread_id)
        self._run_workflow(initial, progress_callback)
        self._started = True
        return self._reply_from_state(kind="report")

    # ----------------------------------------------------------------- send
    def send(
        self,
        user_message: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> AgentReply:
        """
        Send any user message.

        We route it via the LLM:
          • If it looks like a NEW JD → call .start().
          • If it's feedback / approval / refinement → resume the graph
            past the `human_feedback` interrupt.
          • If it's a tool request (compare / interview / explain) →
            answer directly, without resuming the graph.
        """
        message = (user_message or "").strip()
        if not message:
            return AgentReply("(empty message)", stage=self._current_stage())

        if not self._started or self._looks_like_jd(message):
            return self.start(message, progress_callback=progress_callback)

        intent_payload = self._classify_intent(message)
        intent = intent_payload.get("intent", "chat")
        targets = [t for t in intent_payload.get("targets", []) if t]
        log.info("Routed user message → intent=%s targets=%s", intent, targets)

        if intent == "new_jd":
            return self.start(message, progress_callback=progress_callback)
        if intent == "compare":
            return self._handle_compare(targets)
        if intent == "interview":
            return self._handle_interview(targets)
        if intent == "explain":
            return self._handle_explain(targets, message)
        if intent in {"feedback", "approve"}:
            return self._resume_with_feedback(
                message,
                approve=(intent == "approve"),
                progress_callback=progress_callback,
            )
        # Fallback: treat as feedback if we're awaiting it, else chat
        if self._awaiting_feedback():
            return self._resume_with_feedback(
                message, approve=False, progress_callback=progress_callback
            )
        return self._chat_only(message)

    # ----------------------------------------------------- direct tool calls
    def compare(self, targets: List[str]) -> AgentReply:
        return self._handle_compare(targets)

    def interview(self, target: str) -> AgentReply:
        return self._handle_interview([target])

    # ============================================================ internals
    def _current_stage(self) -> str:
        st = self.state() or {}
        return str(st.get("stage", "idle"))

    def _awaiting_feedback(self) -> bool:
        st = self.state() or {}
        return bool(st.get("awaiting_feedback"))

    def _candidates(self) -> List[CandidateMatch]:
        st = self.state() or {}
        return list(st.get("candidates") or [])

    def _requirements(self) -> Optional[Requirements]:
        st = self.state() or {}
        return st.get("requirements")

    # ---------- intent routing
    _JD_HINT_RE = re.compile(
        r"(job description|we are hiring|responsibilities|requirements:|qualifications)",
        re.IGNORECASE,
    )

    def _looks_like_jd(self, message: str) -> bool:
        # crude heuristic: long-form text with JD-ish keywords
        return len(message) > 300 and bool(self._JD_HINT_RE.search(message))

    def _classify_intent(self, message: str) -> Dict[str, Any]:
        candidates = self._candidates()
        known = [c.name for c in candidates] or ["(none yet)"]
        llm = get_chat_llm(temperature=0.0, max_tokens=200)
        msgs = router_prompt.format_messages(
            stage=self._current_stage(),
            known_candidates=", ".join(known),
            message=message,
        )
        try:
            resp = llm.invoke(msgs)
            raw = _content(resp)
            data = self._safe_json(raw)
            if isinstance(data, dict) and "intent" in data:
                return data
        except Exception as exc:  # noqa: BLE001
            log.warning("Intent classification failed: %s", exc)

        # Heuristic fallback
        lower = message.lower()
        if any(w in lower for w in ("compare", " vs ", "side by side")):
            return {"intent": "compare", "targets": []}
        if "interview" in lower or "question" in lower:
            return {"intent": "interview", "targets": []}
        if lower.startswith("why "):
            return {"intent": "explain", "targets": []}
        if any(w in lower for w in ("approve", "looks good", "go ahead", "finalize", "final recommendation")):
            return {"intent": "approve", "targets": []}
        return {"intent": "feedback", "targets": []}

    @staticmethod
    def _safe_json(raw: str) -> Dict[str, Any]:
        raw = (raw or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?", "", raw).strip()
            raw = re.sub(r"```$", "", raw).strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}

    # ---------- feedback / resume
    def _resume_with_feedback(
        self,
        message: str,
        *,
        approve: bool,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> AgentReply:
        feedback = self._interpret_feedback(message, approve=approve)

        self.graph.update_state(
            self.config,
            {
                "human_feedback": feedback,
                "awaiting_feedback": False,
                "messages": [HumanMessage(content=message)],
            },
        )
        log.info("Resuming graph with feedback (approve=%s)", feedback.approve)
        # `None` means "resume from the interrupt without injecting new input".
        self._run_workflow(None, progress_callback)

        kind = "final" if feedback.approve else "report"
        return self._reply_from_state(kind=kind)

    # ---------- shared workflow runner (used by start + resume)
    def _run_workflow(
        self,
        input_state: Optional[AgentState],
        progress_callback: Optional[ProgressCallback],
    ) -> None:
        """Run / resume the graph; if a callback is provided, stream node events."""
        if progress_callback is None:
            self.graph.invoke(input_state, config=self.config)
            return
        try:
            for chunk in self.graph.stream(
                input_state, config=self.config, stream_mode="updates"
            ):
                if not isinstance(chunk, dict):
                    continue
                for node_name in chunk.keys():
                    try:
                        progress_callback({"node": node_name, "stage": "completed"})
                    except Exception:  # noqa: BLE001
                        log.warning("Progress callback raised; continuing.", exc_info=True)
        except Exception:  # noqa: BLE001
            log.exception("Workflow streaming failed; falling back to invoke().")
            # Best-effort fallback so the UI still gets a final state.
            self.graph.invoke(input_state, config=self.config)

    def _interpret_feedback(self, message: str, *, approve: bool) -> HumanFeedback:
        """LLM-assisted feedback parsing with deterministic fallback."""
        from agents.prompts import interpret_feedback_prompt

        llm = get_chat_llm(temperature=0.0, max_tokens=400)
        msgs = interpret_feedback_prompt.format_messages(feedback=message)
        try:
            resp = llm.invoke(msgs)
            data = self._safe_json(_content(resp))
        except Exception:  # noqa: BLE001
            data = {}

        boost = [s.lower() for s in (data.get("boost_skills") or []) if s]
        penalize = [s.lower() for s in (data.get("penalize_skills") or []) if s]
        min_years = data.get("min_years_override")
        try:
            min_years = float(min_years) if min_years is not None else None
        except (TypeError, ValueError):
            min_years = None

        approve_llm = bool(data.get("approve"))
        return HumanFeedback(
            instruction=data.get("instruction") or message,
            boost_skills=boost,
            penalize_skills=penalize,
            min_years_override=min_years,
            approve=approve or approve_llm,
        )

    # ---------- direct tool dispatchers
    def _handle_compare(self, targets: List[str]) -> AgentReply:
        candidates = self._candidates()
        reqs = self._requirements()
        if not candidates or not reqs:
            return AgentReply("No candidates to compare yet — start with a job description.",
                              stage=self._current_stage())
        chosen = select_candidates(candidates, targets)
        text = compare_candidates(chosen, reqs)
        table = comparison_table(chosen, reqs)
        # Record the assistant message so future routing has context
        self.graph.update_state(
            self.config, {"messages": [AIMessage(content=text)]}
        )
        return AgentReply(
            text=text,
            candidates=chosen,
            stage=self._current_stage(),
            kind="comparison",
            extras={"table": table.to_dict(orient="records")},
        )

    def _handle_interview(self, targets: List[str]) -> AgentReply:
        candidates = self._candidates()
        reqs = self._requirements()
        if not candidates or not reqs:
            return AgentReply("Need a shortlist first. Provide a JD to start.",
                              stage=self._current_stage())
        target = (targets[0] if targets else candidates[0].name)
        candidate = find_candidate(candidates, target) or candidates[0]
        text = generate_interview_questions(candidate, reqs)
        rendered = f"### Interview questions for {candidate.name}\n\n{text}"
        self.graph.update_state(self.config, {"messages": [AIMessage(content=rendered)]})
        return AgentReply(
            text=rendered,
            candidates=[candidate],
            stage=self._current_stage(),
            kind="interview",
        )

    def _handle_explain(self, targets: List[str], message: str) -> AgentReply:
        candidates = self._candidates()
        if not candidates:
            return AgentReply("No candidates ranked yet.", stage=self._current_stage())

        # 1. Resolve the targets the user actually meant.
        chosen: List[CandidateMatch] = []
        seen_ids: set[str] = set()
        for raw in targets:
            cand = find_candidate(candidates, raw)
            if cand and cand.candidate_id not in seen_ids:
                chosen.append(cand)
                seen_ids.add(cand.candidate_id)

        # If the router didn't extract names (or only partial), scan the
        # message for first / last names of the known candidates.
        if not chosen:
            chosen = self._resolve_targets_from_message(message, candidates)

        # If we STILL can't identify anyone, ask for clarification instead
        # of silently explaining the top candidate (which is misleading).
        if not chosen:
            options = ", ".join(c.name.split()[0] for c in candidates[:5])
            return AgentReply(
                text=(
                    "I'm not sure which candidate(s) you're asking about. "
                    f"Try a name from the shortlist (e.g. **{options}**), or "
                    "ask things like *'why did Elena rank above Brian?'*."
                ),
                candidates=candidates[:3],
                stage=self._current_stage(),
                kind="chat",
            )

        # 2. Build the response.
        if len(chosen) >= 2 and self._is_comparative(message):
            text = self._comparative_explanation(chosen, message)
        else:
            chunks = [
                f"**{c.name}** (score {c.score:.1f}/100)\n\n"
                + (c.explanation or "No explanation captured.")
                for c in chosen
            ]
            text = "\n\n---\n\n".join(chunks)

        self.graph.update_state(self.config, {"messages": [AIMessage(content=text)]})
        return AgentReply(
            text=text,
            candidates=chosen + [c for c in candidates if c not in chosen][:2],
            stage=self._current_stage(),
            kind="chat",
        )

    # ---------- explanation helpers
    _COMPARATIVE_HINT_RE = re.compile(
        r"\b(over|vs\.?|versus|above|better than|rather than|compared to|instead of)\b",
        re.IGNORECASE,
    )

    @classmethod
    def _is_comparative(cls, message: str) -> bool:
        return bool(cls._COMPARATIVE_HINT_RE.search(message or ""))

    @staticmethod
    def _resolve_targets_from_message(
        message: str, candidates: List[CandidateMatch]
    ) -> List[CandidateMatch]:
        """Find candidates whose full name OR any name token matches the message.

        Handles inputs like "why elena over brian" where the user only used
        first names — the previous implementation required the full name
        substring and so always fell back to the top candidate (a bug).
        """
        if not message or not candidates:
            return []
        msg_lower = message.lower()
        msg_tokens = set(re.findall(r"[a-z]+", msg_lower))

        # Preserve the order in which names appear in the message so that
        # "why X over Y" returns [X, Y], not the candidate-list order.
        scored: List[tuple[int, CandidateMatch]] = []
        for c in candidates:
            name_lower = c.name.lower()
            position: Optional[int] = None
            if name_lower in msg_lower:
                position = msg_lower.find(name_lower)
            else:
                for part in re.findall(r"[a-z]+", name_lower):
                    if len(part) <= 2:
                        continue  # skip initials / "de" / "la" etc.
                    if part in msg_tokens:
                        position = msg_lower.find(part)
                        break
            if position is not None:
                scored.append((position, c))

        scored.sort(key=lambda x: x[0])
        out: List[CandidateMatch] = []
        seen: set[str] = set()
        for _, c in scored:
            if c.candidate_id not in seen:
                out.append(c)
                seen.add(c.candidate_id)
        return out

    def _comparative_explanation(
        self, chosen: List[CandidateMatch], message: str
    ) -> str:
        """Ask the LLM for a focused 'why A over B' answer."""
        reqs = self._requirements()
        payload = [
            {
                "name": c.name,
                "score": c.score,
                "years_experience": c.years_experience,
                "must_have_hits": c.must_have_hits,
                "must_have_misses": c.must_have_misses,
                "nice_to_have_hits": c.nice_to_have_hits,
                "strengths": c.strengths,
                "weaknesses": c.weaknesses,
                "excerpt": c.resume_excerpt[:500],
            }
            for c in chosen[:4]
        ]
        system = (
            "You are a senior technical recruiter.  The user asked a "
            "comparative question about two or more candidates.  Answer "
            "DIRECTLY in 3-6 sentences, citing concrete differences in "
            "must-have coverage, experience, and any decisive factors.  "
            "Plain markdown, no headers."
        )
        human = (
            f"User question: {message}\n\n"
            f"Job requirements:\n{reqs.summary() if reqs else '—'}\n\n"
            f"Candidates (JSON):\n{json.dumps(payload, indent=2)}"
        )
        try:
            llm = get_chat_llm(temperature=0.3, max_tokens=600)
            resp = llm.invoke([("system", system), ("human", human)])
            answer = _content(resp).strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("Comparative explanation LLM failed: %s", exc)
            answer = ""
        if not answer:
            # Deterministic fallback: per-candidate explanation blocks.
            return "\n\n---\n\n".join(
                f"**{c.name}** (score {c.score:.1f}/100)\n\n"
                + (c.explanation or "No explanation captured.")
                for c in chosen
            )
        # Prepend a tiny score line so the answer has receipts.
        header = " · ".join(f"**{c.name}** {c.score:.1f}" for c in chosen)
        return f"{header}\n\n{answer}"

    def _chat_only(self, message: str) -> AgentReply:
        """Plain chat reply that doesn't touch the workflow state."""
        llm = get_chat_llm(temperature=0.4)
        try:
            resp = llm.invoke([
                ("system",
                 "You are a friendly recruiting assistant.  Keep replies under 80 words."),
                ("human", message),
            ])
            text = _content(resp).strip()
        except Exception as exc:  # noqa: BLE001
            text = f"(chat error: {exc})"
        self.graph.update_state(self.config, {
            "messages": [HumanMessage(content=message), AIMessage(content=text)]
        })
        return AgentReply(text=text, stage=self._current_stage(), kind="chat")

    # ---------- state → reply
    def _reply_from_state(self, kind: str) -> AgentReply:
        st = self.state() or {}
        stage = str(st.get("stage", "idle"))
        candidates = list(st.get("candidates") or [])

        text = ""
        if kind == "final" or stage == "done":
            text = st.get("final_recommendation") or ""
            kind = "final"
        else:
            text = st.get("match_report") or ""
            if not text:
                # No report yet (e.g. parse error) — surface the last AI message.
                msgs = st.get("messages") or []
                for m in reversed(msgs):
                    if isinstance(m, AIMessage):
                        text = str(m.content)
                        break

        return AgentReply(text=text or "_(no response)_", candidates=candidates,
                          stage=stage, kind=kind)


# ---------------------------------------------------------------------------
# CLI: `python -m matching_agent --jd path/to/jd.txt`
# ---------------------------------------------------------------------------
def _cli() -> None:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Run the Resume Matching Agent in the terminal.")
    parser.add_argument("--jd", required=True, help="Path to a JD file (.txt/.md/.pdf/.docx)")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    from tools.file_tools import read_jd_text

    jd_text = read_jd_text(args.jd)
    agent = ResumeMatchingAgent()
    reply = agent.start(jd_text=jd_text, top_k=args.top_k)
    print("\n=== INITIAL MATCH REPORT ===\n")
    print(reply.text)
    print("\n(Type feedback, or 'approve' to finalise, or 'quit')")

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in {"quit", "exit"}:
            break
        reply = agent.send(line)
        print(f"\n--- {reply.kind} / stage={reply.stage} ---\n")
        print(reply.text)
        if reply.kind == "final":
            break


if __name__ == "__main__":  # pragma: no cover
    _cli()
