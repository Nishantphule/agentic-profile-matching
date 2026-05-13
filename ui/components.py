"""
ui/components.py
================

Reusable Streamlit widgets used by the chat interface.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional

import pandas as pd
import streamlit as st

from agents.state import CandidateMatch, Requirements


# ---------------------------------------------------------------------------
# Sidebar — settings + knowledge base
# ---------------------------------------------------------------------------
def render_sidebar(
    *,
    on_ingest,
    on_reset,
    resumes_count: int,
    model_name: str,
) -> dict:
    """Render the left sidebar.  Returns user-set options.

    Quick Actions live in `render_sidebar_quick_actions(...)` and are
    rendered separately by the caller (only after a shortlist exists).
    """
    with st.sidebar:
        st.title("🧭 Resume Matching")
        st.caption("LangGraph · LangChain · OpenRouter · FAISS")

        st.divider()
        st.markdown("##### 📚 Knowledge base")
        cols = st.columns([0.55, 0.45])
        cols[0].metric("Resumes", resumes_count)
        with cols[1]:
            if st.button("🔄 Re-ingest", use_container_width=True, key="sb_ingest"):
                on_ingest()

        st.markdown("##### ⚙️ Search settings")
        top_k = st.slider(
            "Top-K per round", 3, 30, 10, key="sb_topk",
            help="How many candidates the agent retrieves from FAISS for round 1.",
        )

        st.divider()
        if st.button("♻️ Reset conversation", type="secondary",
                     use_container_width=True, key="sb_reset"):
            on_reset()

        st.caption(f"Model: `{model_name}`")
        render_attribution_footer()
    return {"top_k": top_k}


# ---------------------------------------------------------------------------
# Attribution footer (sidebar + page bottom)
# ---------------------------------------------------------------------------
AUTHOR_NAME = "Nishant Phule"
AUTHOR_YEAR = 2026


def render_attribution_footer() -> None:
    """Small © attribution block. Safe to call multiple times per render."""
    st.markdown(
        f"""
        <div style="
            margin-top: 1.25rem;
            padding-top: 0.6rem;
            border-top: 1px solid rgba(128,128,128,0.25);
            text-align: center;
            font-size: 0.78rem;
            line-height: 1.4;
            opacity: 0.85;
        ">
            Made with care by<br/>
            <strong>{AUTHOR_NAME}</strong>™<br/>
            © {AUTHOR_YEAR} · All rights reserved
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar — Quick Actions (always-visible action panel)
# ---------------------------------------------------------------------------
def render_sidebar_quick_actions(
    candidates: List[CandidateMatch],
    *,
    on_compare: Callable[[List[str]], None],
    on_interview: Callable[[str], None],
    on_finalize: Callable[[], None],
) -> None:
    """Render the sticky Quick Actions panel inside the sidebar.

    Living in the sidebar means these controls stay visible no matter how
    long the conversation grows on the main page.
    """
    with st.sidebar:
        st.divider()
        st.markdown("##### ⚡ Quick actions")

        if not candidates:
            st.caption("Run a match to unlock actions.")
            return

        names = [c.name for c in candidates]

        with st.container(border=True):
            st.markdown("**Compare candidates**")
            sel = st.multiselect(
                "Pick 2+ candidates",
                names,
                default=names[: min(3, len(names))],
                key="sb_cmp_sel",
                label_visibility="collapsed",
                placeholder="Pick 2+ candidates",
            )
            if st.button(
                "Run comparison",
                use_container_width=True,
                key="sb_cmp_btn",
                disabled=len(sel) < 2,
            ):
                on_compare(sel)

        with st.container(border=True):
            st.markdown("**Interview questions**")
            target = st.selectbox(
                "Candidate",
                names,
                key="sb_iv_target",
                label_visibility="collapsed",
            )
            if st.button(
                "Generate questions",
                use_container_width=True,
                key="sb_iv_btn",
            ):
                on_interview(target)

        if st.button(
            "✅ Finalize recommendation",
            use_container_width=True,
            type="primary",
            key="sb_fin_btn",
        ):
            on_finalize()


# ---------------------------------------------------------------------------
# Compact status strip rendered above the chat / shortlist
# ---------------------------------------------------------------------------
def render_status_strip(
    candidates: List[CandidateMatch],
    *,
    stage: str = "idle",
    rounds_done: int = 0,
) -> None:
    """Show a row of small KPI metrics so the user always has at-a-glance context."""
    cols = st.columns(4)
    top = candidates[0] if candidates else None
    cols[0].metric("Stage", _pretty_stage(stage))
    cols[1].metric("Round", rounds_done if rounds_done else "—")
    cols[2].metric("Shortlist", len(candidates))
    cols[3].metric(
        "Top score",
        f"{top.score:.1f}" if top else "—",
        delta=top.name if top else None,
        delta_color="off",
    )


def _pretty_stage(stage: str) -> str:
    return {
        "idle": "Idle",
        "parse_jd": "Parsing JD",
        "extract_requirements": "Extracting reqs",
        "search_resumes": "Searching",
        "rank_candidates": "Ranking",
        "match_report": "Reporting",
        "human_feedback": "Awaiting you",
        "rerank": "Re-ranking",
        "final_recommendation": "Finalizing",
        "done": "Done ✅",
    }.get(stage, stage or "—")


# ---------------------------------------------------------------------------
# JD input area
# ---------------------------------------------------------------------------
def render_jd_input() -> str:
    """Two-tab JD input (paste OR file upload).  Returns the JD text or ''."""
    tab_paste, tab_upload = st.tabs(["📋 Paste JD", "📂 Upload JD"])
    text = ""
    with tab_paste:
        text = st.text_area(
            "Job Description",
            placeholder="Paste a complete job description here…",
            height=220,
            key="jd_textarea",
        )
    with tab_upload:
        up = st.file_uploader("Upload .txt / .md / .pdf / .docx", type=["txt", "md", "pdf", "docx"])
        if up is not None:
            from tools.file_tools import read_uploaded_file

            text = read_uploaded_file(up.name, up.getvalue())
            with st.expander("Preview"):
                st.text(text[:1200] + ("…" if len(text) > 1200 else ""))
    return text.strip()


# ---------------------------------------------------------------------------
# Resume upload
# ---------------------------------------------------------------------------
def render_resume_uploader(resumes_dir, on_save) -> None:
    with st.expander("➕ Add resumes to knowledge base"):
        files = st.file_uploader(
            "Upload one or more resumes",
            type=["txt", "md", "pdf", "docx"],
            accept_multiple_files=True,
            key="resume_uploader",
        )
        if files:
            from tools.file_tools import read_uploaded_file, save_resume_text

            for f in files:
                text = read_uploaded_file(f.name, f.getvalue())
                if text.strip():
                    save_resume_text(resumes_dir, f.name, text)
            st.success(f"Saved {len(files)} resume(s) to {resumes_dir}.  Click **Re-ingest**.")
            on_save()


# ---------------------------------------------------------------------------
# Candidate table + expanders
# ---------------------------------------------------------------------------
def render_candidate_table(
    candidates: List[CandidateMatch],
    *,
    height: Optional[int] = 260,
) -> None:
    if not candidates:
        st.info("No candidates yet — paste a JD and click **Start matching** to see results here.")
        return
    rows = [
        {
            "#": i + 1,
            "Candidate": c.name,
            "Score": round(c.score, 1),
            "Yrs": c.years_experience if c.years_experience is not None else "—",
            "Hits": ", ".join(c.must_have_hits) or "—",
            "Gaps": ", ".join(c.must_have_misses) or "—",
            "Round": c.round_reached,
        }
        for i, c in enumerate(candidates)
    ]
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=height,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100, format="%.1f",
            ),
        },
    )


def render_candidate_details(
    candidates: List[CandidateMatch],
    *,
    show_header: bool = True,
) -> None:
    if not candidates:
        return
    if show_header:
        st.markdown("##### 🧾 Candidate details")
    for i, c in enumerate(candidates, 1):
        with st.expander(f"#{i} · {c.name} — {c.score:.1f}/100"):
            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Strengths**")
                for s in c.strengths or ["—"]:
                    st.markdown(f"- {s}")
            with cols[1]:
                st.markdown("**Gaps**")
                for w in c.weaknesses or ["—"]:
                    st.markdown(f"- {w}")
            st.markdown("**Why this ranking**")
            st.write(c.explanation or "—")
            if c.resume_excerpt:
                with st.expander("Resume excerpt"):
                    st.text(c.resume_excerpt)


# ---------------------------------------------------------------------------
# Requirements summary
# ---------------------------------------------------------------------------
def render_requirements_panel(reqs: Requirements | None) -> None:
    if reqs is None:
        return
    title = "📑 Extracted requirements"
    if reqs.role_title:
        title = f"📑 {reqs.role_title}"
    with st.expander(title, expanded=False):
        cols = st.columns(3)
        cols[0].metric("Min Years", f"{reqs.min_years_experience or 0:g}")
        cols[1].metric("Must-Have", len(reqs.must_have_skills))
        cols[2].metric("Nice-to-Have", len(reqs.nice_to_have_skills))
        if reqs.seniority:
            st.markdown(f"**Seniority:** {reqs.seniority}")
        st.markdown("**Must-have:** " + (", ".join(reqs.must_have_skills) or "—"))
        st.markdown("**Nice-to-have:** " + (", ".join(reqs.nice_to_have_skills) or "—"))
        if reqs.certifications:
            st.markdown("**Certifications:** " + ", ".join(reqs.certifications))


# ---------------------------------------------------------------------------
# Multi-round screening visualisation
# ---------------------------------------------------------------------------
def render_screening_funnel(history_rounds: Iterable[dict]) -> None:
    rows = list(history_rounds)
    if not rows:
        return
    st.subheader("Multi-round screening")
    cols = st.columns(len(rows))
    for col, row in zip(cols, rows):
        col.metric(f"Round {row['round']}", f"{row['count']} candidate(s)", row.get("note", ""))


# ---------------------------------------------------------------------------
# Chat bubble helper
# ---------------------------------------------------------------------------
def render_chat_message(role: str, content: str) -> None:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(content)
