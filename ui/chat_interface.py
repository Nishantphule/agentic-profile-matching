"""
ui/chat_interface.py
====================

Main Streamlit page.

Layout goals
------------
1. **Quick Actions live in the sidebar** — they are always visible no matter
   how long the conversation grows on the main page.
2. **JD setup auto-collapses** after the first match so it doesn't dominate
   the screen.
3. **Chat history lives in a fixed-height scrollable container** so it never
   stretches the page vertically.
4. **Right-hand candidate panel** uses a compact table + an internally
   scrollable details container so the page never grows in height.
5. A **status strip** at the top of the main area gives at-a-glance KPIs
   (stage, round, shortlist size, top score).
"""

from __future__ import annotations

from typing import Literal, Optional

import streamlit as st

from config import get_logger, settings
from matching_agent import NODE_PROGRESS, AgentReply, ResumeMatchingAgent
from ui.components import (
    AUTHOR_NAME,
    AUTHOR_YEAR,
    render_candidate_details,
    render_candidate_table,
    render_chat_message,
    render_jd_input,
    render_requirements_panel,
    render_resume_uploader,
    render_screening_funnel,
    render_sidebar,
    render_sidebar_quick_actions,
    render_status_strip,
)
from vectorstore.ingest import ResumeIndex

log = get_logger(__name__)


SESSION_KEYS = {
    "agent": "agent",
    "messages": "messages",
    "started": "started",
    "round_history": "round_history",
}


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------
def _ensure_session() -> None:
    if SESSION_KEYS["agent"] not in st.session_state:
        st.session_state[SESSION_KEYS["agent"]] = ResumeMatchingAgent()
    if SESSION_KEYS["messages"] not in st.session_state:
        st.session_state[SESSION_KEYS["messages"]] = []
    if SESSION_KEYS["started"] not in st.session_state:
        st.session_state[SESSION_KEYS["started"]] = False
    if SESSION_KEYS["round_history"] not in st.session_state:
        st.session_state[SESSION_KEYS["round_history"]] = []


def _agent() -> ResumeMatchingAgent:
    return st.session_state[SESSION_KEYS["agent"]]


def _record_user(text: str) -> None:
    st.session_state[SESSION_KEYS["messages"]].append({"role": "user", "content": text})


def _record_assistant(reply: AgentReply) -> None:
    st.session_state[SESSION_KEYS["messages"]].append(
        {"role": "assistant", "content": reply.text, "kind": reply.kind, "stage": reply.stage}
    )
    cands = reply.candidates
    if cands:
        rnd = max(c.round_reached for c in cands)
        history = st.session_state[SESSION_KEYS["round_history"]]
        if not history or history[-1]["round"] != rnd:
            history.append({"round": rnd, "count": len(cands), "note": f"{cands[0].name} top"})
        else:
            history[-1]["count"] = len(cands)


def _reset_conversation() -> None:
    _agent().reset()
    st.session_state[SESSION_KEYS["messages"]] = []
    st.session_state[SESSION_KEYS["started"]] = False
    st.session_state[SESSION_KEYS["round_history"]] = []


# ---------------------------------------------------------------------------
# Knowledge base helpers
# ---------------------------------------------------------------------------
def _ingest_resumes() -> int:
    with st.spinner("Ingesting resumes into FAISS…"):
        idx = ResumeIndex()
        n = idx.build_from_dir(settings.resumes_dir)
        if n:
            idx.save()
            from tools.rag_search import reset_index
            reset_index()
        return n


def _count_indexed() -> int:
    """Cached resume-count read (uses the FAISS singleton from rag_search)."""
    try:
        from tools.rag_search import get_index

        return len(get_index().records)
    except FileNotFoundError:
        return 0
    except Exception:  # noqa: BLE001
        return 0


# ---------------------------------------------------------------------------
# Progress UI — human-readable “what the model / agent is doing” copy
# (UI-only; does not change LangGraph or matching_agent behaviour.)
# ---------------------------------------------------------------------------
ProgressKind = Literal["initial", "respond", "finalize"]

# Shown once the corresponding graph node has *finished* (past tense, honest
# about whether an LLM call was involved).
_NODE_JUST_FINISHED: dict[str, str] = {
    "parse_jd": "The language model finished reading your JD and produced a tight plain-language summary.",
    "extract_requirements": "The language model converted the JD into structured JSON: must-haves, nice-to-haves, years, certs, and responsibilities.",
    "search_resumes": "Local embeddings + FAISS retrieved the closest resume passages — no chat-model call in this hop.",
    "rank_candidates": "Deterministic scoring blended semantic similarity with your requirements, then the language model wrote short explanations for each shortlisted profile.",
    "generate_match_report": "The language model stitched the ranked profiles into a markdown shortlist report (with a table).",
    "human_feedback": "The graph consumed your latest instructions and chose the next branch (re-rank vs. finalize).",
    "rerank": "Candidates were re-filtered / re-scored with your feedback shaping; the language model refreshed narrative explanations where needed.",
    "final_recommendation": "The language model drafted the final hire / backup / pass recommendation with reasoning and next steps.",
}

# Shown *after* `node` completes, while the next node is still running.
_NODE_UP_NEXT_THINKING: dict[str, str] = {
    "parse_jd": "The model is now extracting structured requirements from the full JD text…",
    "extract_requirements": "Turning those requirements into an embedding query and scanning every resume in the FAISS index…",
    "search_resumes": "Scoring and ranking the retrieved profiles, then asking the model to explain each fit in natural language…",
    "rank_candidates": "Composing the conversational match report you will see in chat (this can take a little while)…",
    "generate_match_report": "Pausing the automated graph so you can keep refining in chat, compare candidates, or approve when ready.",
    "human_feedback": "Continuing the workflow on the server — either re-ranking with your feedback or writing the final recommendation, depending on what you asked.",
    "rerank": "Regenerating the markdown shortlist so the UI reflects the new ordering…",
    "final_recommendation": "Flushing the final memo to the conversation history — almost done.",
}

_INITIAL_THINKING: dict[ProgressKind, str] = {
    "initial": (
        "**🧠 Starting the pipeline.** The first hop asks the language model to read your entire JD "
        "and emit a recruiter-style summary. *Cold starts* can take a few seconds while weights load."
    ),
    "respond": (
        "**🧠 Picking up your message.** The router already classified the intent; the graph is now "
        "resuming with your feedback or approval. Watch each finished step below — some use the LLM, "
        "others are pure retrieval / math."
    ),
    "finalize": (
        "**🧠 Final pass.** Your approval is being written into agent state; the next hop asks the "
        "language model for an executive hire / backup / pass write-up with concrete rationale."
    ),
}


def _render_thinking_panel(
    thinking_area,
    *,
    just_finished: Optional[str],
    up_next: str,
) -> None:
    """Two-part status: what completed + what is happening now."""
    done_block = ""
    if just_finished:
        done_block = f"**✓ Just finished**  \n{just_finished}\n\n"
    thinking_area.markdown(
        done_block + f"**→ In progress**  \n{up_next}",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Progress-bar helper used while the LangGraph workflow is running
# ---------------------------------------------------------------------------
def _run_with_progress(
    label: str,
    action,
    *,
    expected_nodes: int = 5,
    progress_kind: ProgressKind = "initial",
):
    """Run `action(progress_callback)` while showing a live progress bar.

    The optional ``progress_kind`` only changes the *copy* shown to the user;
    the callable ``action`` and the LangGraph stream are untouched.
    """
    status = st.status(label, expanded=True)
    thinking_area = status.empty()
    progress = status.progress(0.0, text="Starting…")
    log_area = status.empty()
    completed_steps: list[tuple[str, str]] = []
    last_node: Optional[str] = None

    _render_thinking_panel(
        thinking_area,
        just_finished=None,
        up_next=_INITIAL_THINKING[progress_kind],
    )

    def callback(event: dict) -> None:
        nonlocal last_node
        node = event.get("node", "")
        info = NODE_PROGRESS.get(node)
        if info:
            label_text = info["label"]
            fraction = float(info["fraction"])
        else:
            label_text = node or "Working…"
            fraction = min(0.95, (len(completed_steps) + 1) / max(expected_nodes, 1))

        just_done = _NODE_JUST_FINISHED.get(
            node,
            f"Graph node `{node}` finished.",
        )
        up_next = _NODE_UP_NEXT_THINKING.get(
            node,
            "Continuing the LangGraph workflow on the server…",
        )

        completed_steps.append((node, label_text))
        last_node = node or last_node

        micro = label_text
        hint = _NODE_UP_NEXT_THINKING.get(node, "")
        if hint:
            micro = f"{label_text} — {hint[:70]}…"

        progress.progress(fraction, text=micro[:120])
        _render_thinking_panel(
            thinking_area,
            just_finished=just_done,
            up_next=up_next,
        )
        log_lines = []
        for i, (nkey, lbl) in enumerate(completed_steps, start=1):
            blurb = _NODE_JUST_FINISHED.get(nkey, "")
            short = blurb if len(blurb) <= 100 else blurb[:97] + "…"
            log_lines.append(f"{i}. ✅ **{lbl}** — _{short}_")
        log_area.markdown("\n".join(log_lines))

    try:
        result = action(callback)
        progress.progress(1.0, text="Complete")
        if not completed_steps:
            done_summary = (
                "No LangGraph nodes ran — this was handled as a lightweight chat reply "
                "(router skipped the heavy matching pipeline)."
            )
        else:
            done_summary = (
                _NODE_JUST_FINISHED[last_node]
                if last_node and last_node in _NODE_JUST_FINISHED
                else "All graph steps finished."
            )
        _render_thinking_panel(
            thinking_area,
            just_finished=done_summary,
            up_next="**✓ Pipeline idle.** You can keep chatting, use sidebar **Quick actions**, or paste a new JD.",
        )
        status.update(label=f"{label} — done", state="complete", expanded=False)
        return result
    except Exception as exc:  # noqa: BLE001
        thinking_area.markdown(
            "**⚠️ Run failed** — see the traceback below. The agent state was not advanced.",
            unsafe_allow_html=True,
        )
        status.update(label=f"{label} — failed", state="error", expanded=True)
        st.exception(exc)
        raise


# ---------------------------------------------------------------------------
# Sidebar quick-action handlers (closures so they can mutate session state)
# ---------------------------------------------------------------------------
def _make_quick_action_handlers():
    def on_compare(selected_names):
        if not selected_names or len(selected_names) < 2:
            st.toast("Pick at least two candidates to compare.", icon="⚠️")
            return
        _record_user(f"Compare: {', '.join(selected_names)}")
        with st.spinner("Comparing candidates…"):
            reply = _agent().compare(selected_names)
        _record_assistant(reply)
        st.rerun()

    def on_interview(target_name):
        if not target_name:
            return
        _record_user(f"Interview questions for {target_name}")
        with st.spinner("Generating interview questions…"):
            reply = _agent().interview(target_name)
        _record_assistant(reply)
        st.rerun()

    def on_finalize():
        _record_user("approve — please finalize")
        reply = _run_with_progress(
            "Building final hiring recommendation",
            lambda cb: _agent().send("approve — please finalize", progress_callback=cb),
            expected_nodes=2,
            progress_kind="finalize",
        )
        _record_assistant(reply)
        st.rerun()

    return on_compare, on_interview, on_finalize


# ---------------------------------------------------------------------------
# Theme CSS — complements `.streamlit/config.toml` [theme] (slate + cyan)
# ---------------------------------------------------------------------------
_CUSTOM_CSS = """
<style>
  /* --- layout rhythm --- */
  .block-container { padding-top: 1.25rem; padding-bottom: 1.5rem; max-width: 100%; }

  /* --- app chrome: subtle depth on top of Streamlit theme vars --- */
  .stApp {
    background: linear-gradient(
      165deg,
      #0b1120 0%,
      #0d1528 38%,
      #0a1628 100%
    );
  }

  /* --- sidebar: panel + accent rail --- */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #0b1120 100%) !important;
    border-right: 1px solid rgba(34, 211, 238, 0.18) !important;
  }
  section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

  /* --- primary buttons: slightly richer cyan --- */
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 12px rgba(6, 182, 212, 0.28);
  }
  .stButton > button[kind="primary"]:hover {
    filter: brightness(1.08);
    box-shadow: 0 4px 16px rgba(6, 182, 212, 0.38);
  }

  /* --- secondary / default buttons --- */
  .stButton > button[kind="secondary"] {
    border-color: rgba(148, 163, 184, 0.35) !important;
    color: #cbd5e1 !important;
  }

  /* --- tabs: accent underline --- */
  .stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    border-bottom: 1px solid rgba(34, 211, 238, 0.15) !important;
  }
  .stTabs [aria-selected="true"] {
    color: #22d3ee !important;
    border-bottom-color: #22d3ee !important;
  }

  /* --- expanders --- */
  details[data-testid="stExpander"] {
    border: 1px solid rgba(34, 211, 238, 0.12) !important;
    border-radius: 10px !important;
    background: rgba(17, 28, 47, 0.55) !important;
  }
  details[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: #e2e8f0 !important;
  }

  /* --- bordered containers (chat + details panels) --- */
  div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: rgba(34, 211, 238, 0.22) !important;
    border-radius: 14px !important;
    background: rgba(15, 23, 42, 0.42) !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.22);
  }

  /* --- chat bubbles --- */
  [data-testid="stChatMessage"] {
    border-radius: 12px !important;
    border: 1px solid rgba(148, 163, 184, 0.12) !important;
    background: rgba(15, 23, 42, 0.5) !important;
    margin-bottom: 0.6rem !important;
    padding: 0.35rem 0.5rem !important;
  }

  /* --- chat input --- */
  [data-testid="stChatInput"] {
    margin-top: 0.5rem;
    border-radius: 12px !important;
    border: 1px solid rgba(34, 211, 238, 0.2) !important;
  }

  /* --- status / progress during workflow --- */
  [data-testid="stStatusWidget"] summary {
    font-weight: 600 !important;
    color: #e2e8f0 !important;
  }

  /* --- dataframes --- */
  [data-testid="stDataFrame"] { width: 100%; }

  /* --- metrics (status strip + sidebar) --- */
  [data-testid="stMetric"] {
    background: rgba(17, 28, 47, 0.65);
    border: 1px solid rgba(34, 211, 238, 0.1);
    border-radius: 10px;
    padding: 0.5rem 0.65rem !important;
  }
  [data-testid="stMetricValue"] { color: #22d3ee !important; }

  section[data-testid="stSidebar"] [data-testid="stMetric"] { padding: 4px 8px !important; }
  section[data-testid="stSidebar"] [data-testid="stMetricValue"] { font-size: 1.35rem !important; }

  /* --- alerts / info boxes --- */
  div[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: 1px solid rgba(34, 211, 238, 0.15) !important;
  }
</style>
"""


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------
def render() -> None:
    st.set_page_config(
        page_title="Resume Matching Agent",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)
    _ensure_session()

    # ----- snapshot of current agent state (read once per rerun) ----------
    state = _agent().state() or {}
    candidates = list(state.get("candidates") or [])
    reqs = state.get("requirements")
    stage = str(state.get("stage", "idle"))
    rounds_done = max(
        int(state.get("round", 1)) if candidates else 0,
        len(st.session_state[SESSION_KEYS["round_history"]]),
    )

    # ----- sidebar: settings + quick actions ------------------------------
    sidebar_opts = render_sidebar(
        on_ingest=lambda: _on_ingest(),
        on_reset=lambda: _on_reset(),
        resumes_count=_count_indexed(),
        model_name=settings.model,
    )
    on_compare, on_interview, on_finalize = _make_quick_action_handlers()
    render_sidebar_quick_actions(
        candidates,
        on_compare=on_compare,
        on_interview=on_interview,
        on_finalize=on_finalize,
    )

    # ----- header + status strip ------------------------------------------
    title_col, author_col = st.columns([0.7, 0.3])
    with title_col:
        st.title("🧭 Resume Matching Agent")
        st.caption("Paste a JD, upload resumes, and refine the shortlist through conversation.")
    with author_col:
        st.markdown(
            f"""
            <div style="
                text-align: right;
                margin-top: 0.6rem;
                font-size: 0.85rem;
                line-height: 1.45;
                color: #94a3b8;
            ">
                Made by <strong style="color:#22d3ee;">{AUTHOR_NAME}</strong>™<br/>
                <span style="font-size: 0.78rem; color:#64748b;">© {AUTHOR_YEAR} · All rights reserved</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    render_status_strip(candidates, stage=stage, rounds_done=rounds_done)

    # ----- JD setup (auto-collapses after the first match) ----------------
    started = st.session_state[SESSION_KEYS["started"]]
    setup_label = (
        "📋 Job description setup" if not started else "📋 Job description setup (edit / new)"
    )
    with st.expander(setup_label, expanded=not started):
        render_resume_uploader(
            settings.resumes_dir,
            on_save=lambda: st.toast("Resumes saved.  Click Re-ingest in the sidebar."),
        )

        jd_text = render_jd_input()
        c1, c2 = st.columns([0.35, 0.65])
        with c1:
            start_clicked = st.button(
                "🚀 Start matching",
                type="primary",
                use_container_width=True,
                disabled=not jd_text,
            )
        with c2:
            st.caption(
                "Tip: once running, use the sidebar **Quick Actions** to compare, "
                "generate interview questions, or finalize."
            )

        if start_clicked and jd_text:
            if _count_indexed() == 0:
                st.warning("No resumes indexed yet.  Upload resumes above and click **Re-ingest** in the sidebar.")
            else:
                _record_user("(new JD)\n\n" + jd_text)
                reply = _run_with_progress(
                    "Matching resumes against the job description",
                    lambda cb: _agent().start(
                        jd_text, top_k=sidebar_opts["top_k"], progress_callback=cb
                    ),
                    progress_kind="initial",
                )
                _record_assistant(reply)
                st.session_state[SESSION_KEYS["started"]] = True
                st.rerun()  # collapse the setup expander on next render

    # ----- main two-column workspace --------------------------------------
    chat_col, panel_col = st.columns([0.58, 0.42], gap="large")

    # ----- LEFT: Conversation --------------------------------------------
    with chat_col:
        st.markdown(
            '<p style="margin:0 0 0.35rem 0; font-size:1.05rem; font-weight:600; '
            'color:#e2e8f0; border-left:3px solid #22d3ee; padding-left:0.55rem;">💬 Conversation</p>',
            unsafe_allow_html=True,
        )

        messages = st.session_state[SESSION_KEYS["messages"]]
        chat_box = st.container(height=560, border=True)
        with chat_box:
            if not messages:
                st.info(
                    "No messages yet. Paste a job description above and click "
                    "**Start matching** to begin."
                )
            for msg in messages:
                render_chat_message(msg["role"], msg["content"])

        user_msg = st.chat_input(
            "Refine the shortlist… try 'compare top 3', 'only 5+ years', 'why X over Y', 'approve'",
            disabled=not started,
        )
        if user_msg:
            _record_user(user_msg)
            reply = _run_with_progress(
                "Processing your message",
                lambda cb: _agent().send(user_msg, progress_callback=cb),
                expected_nodes=3,
                progress_kind="respond",
            )
            _record_assistant(reply)
            st.rerun()

    # ----- RIGHT: Shortlist panel ----------------------------------------
    with panel_col:
        st.markdown(
            '<p style="margin:0 0 0.35rem 0; font-size:1.05rem; font-weight:600; '
            'color:#e2e8f0; border-left:3px solid #22d3ee; padding-left:0.55rem;">📊 Live shortlist</p>',
            unsafe_allow_html=True,
        )
        render_requirements_panel(reqs)
        render_screening_funnel(st.session_state[SESSION_KEYS["round_history"]])
        render_candidate_table(candidates, height=260)

        if candidates:
            st.markdown(
                '<p style="margin:0.75rem 0 0.35rem 0; font-size:1.05rem; font-weight:600; '
                'color:#e2e8f0; border-left:3px solid #22d3ee; padding-left:0.55rem;">🧾 Details</p>',
                unsafe_allow_html=True,
            )
            details_box = st.container(height=420, border=True)
            with details_box:
                render_candidate_details(candidates, show_header=False)


# ---------------------------------------------------------------------------
# Sidebar handlers
# ---------------------------------------------------------------------------
def _on_ingest() -> None:
    n = _ingest_resumes()
    if n:
        st.toast(f"Indexed {n} resume(s) ✅")
    else:
        st.toast("No resumes found in data/resumes/", icon="⚠️")


def _on_reset() -> None:
    _reset_conversation()
    st.toast("Conversation reset.")
