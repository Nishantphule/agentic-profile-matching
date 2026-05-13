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
# Progress-bar helper used while the LangGraph workflow is running
# ---------------------------------------------------------------------------
def _run_with_progress(
    label: str,
    action,
    *,
    expected_nodes: int = 5,
):
    """Run `action(progress_callback)` while showing a live progress bar."""
    status = st.status(label, expanded=True)
    progress = status.progress(0.0, text="Starting…")
    log_area = status.empty()
    completed: list[str] = []

    def callback(event: dict) -> None:
        node = event.get("node", "")
        info = NODE_PROGRESS.get(node)
        if info:
            label_text = info["label"]
            fraction = float(info["fraction"])
        else:
            label_text = node or "Working…"
            fraction = min(0.95, (len(completed) + 1) / max(expected_nodes, 1))
        completed.append(label_text)
        progress.progress(fraction, text=label_text)
        log_area.markdown("\n".join(f"- ✅ {step}" for step in completed))

    try:
        result = action(callback)
        progress.progress(1.0, text="Complete")
        status.update(label=f"{label} — done", state="complete", expanded=False)
        return result
    except Exception as exc:  # noqa: BLE001
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
        )
        _record_assistant(reply)
        st.rerun()

    return on_compare, on_interview, on_finalize


# ---------------------------------------------------------------------------
# CSS tweaks (small, targeted — keeps the layout tight)
# ---------------------------------------------------------------------------
_CUSTOM_CSS = """
<style>
/* tighter section spacing */
.block-container { padding-top: 1.4rem; padding-bottom: 1.5rem; }

/* keep dataframes from overflowing horizontally */
[data-testid="stDataFrame"] { width: 100%; }

/* slimmer status (progress) widget header */
[data-testid="stStatusWidget"] summary { font-weight: 600; }

/* chat input always fully visible inside its column */
[data-testid="stChatInput"] { margin-top: .5rem; }

/* compact sidebar metric */
section[data-testid="stSidebar"] [data-testid="stMetric"] { padding: 4px 0; }
section[data-testid="stSidebar"] [data-testid="stMetricValue"] { font-size: 1.4rem; }
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
                line-height: 1.4;
                opacity: 0.85;
            ">
                Made by <strong>{AUTHOR_NAME}</strong>™<br/>
                <span style="font-size: 0.78rem;">© {AUTHOR_YEAR} · All rights reserved</span>
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
                )
                _record_assistant(reply)
                st.session_state[SESSION_KEYS["started"]] = True
                st.rerun()  # collapse the setup expander on next render

    # ----- main two-column workspace --------------------------------------
    chat_col, panel_col = st.columns([0.58, 0.42], gap="large")

    # ----- LEFT: Conversation --------------------------------------------
    with chat_col:
        st.markdown("##### 💬 Conversation")

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
            )
            _record_assistant(reply)
            st.rerun()

    # ----- RIGHT: Shortlist panel ----------------------------------------
    with panel_col:
        st.markdown("##### 📊 Live shortlist")
        render_requirements_panel(reqs)
        render_screening_funnel(st.session_state[SESSION_KEYS["round_history"]])
        render_candidate_table(candidates, height=260)

        if candidates:
            st.markdown("##### 🧾 Details")
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
