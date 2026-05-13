"""
agents/prompts.py
=================

Reusable prompt templates.  All prompts live in one place so they can be
versioned, A/B tested and shared between nodes and tools.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------------------------
# JD parsing / summarisation
# ---------------------------------------------------------------------------
PARSE_JD_SYSTEM = """You are a senior technical recruiter.
Given a raw job description, produce a CLEAN one-paragraph summary (3-5 sentences)
that captures:
  • the role and seniority
  • the main responsibilities
  • the core technologies / stack
  • the most important qualifications

Do not add bullet points. Return plain text only."""

parse_jd_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PARSE_JD_SYSTEM),
        ("human", "Job description:\n\n{jd}"),
    ]
)


# ---------------------------------------------------------------------------
# Requirements extraction (structured JSON)
# ---------------------------------------------------------------------------
EXTRACT_REQUIREMENTS_SYSTEM = """You are an expert technical recruiter.
Extract STRUCTURED requirements from the job description below.

Return ONLY valid JSON matching this exact schema (no markdown, no commentary):

{{
  "role_title": "string or null",
  "seniority": "junior | mid | senior | lead | principal | null",
  "must_have_skills": ["string", ...],
  "nice_to_have_skills": ["string", ...],
  "min_years_experience": number or null,
  "education": ["string", ...],
  "certifications": ["string", ...],
  "responsibilities": ["string", ...]
}}

Rules:
- "must_have_skills" = explicitly required (e.g. "required", "must have", "minimum").
- "nice_to_have_skills" = "preferred", "plus", "bonus", "nice to have".
- Use lowercase, canonical skill names (e.g. "react", "aws", "python").
- If a value is unknown, use null or an empty list.
"""

extract_requirements_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", EXTRACT_REQUIREMENTS_SYSTEM),
        ("human", "{jd}"),
    ]
)


# ---------------------------------------------------------------------------
# Per-candidate explanation
# ---------------------------------------------------------------------------
EXPLAIN_CANDIDATE_SYSTEM = """You are an expert technical recruiter writing
a SHORT explanation (3-5 sentences) of why a candidate matches a job.
Be specific, cite concrete skills / years of experience, and mention any gaps.
Plain text only, no markdown headers."""

explain_candidate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", EXPLAIN_CANDIDATE_SYSTEM),
        (
            "human",
            "Job requirements:\n{requirements}\n\n"
            "Candidate resume excerpt:\n{resume}\n\n"
            "Must-have hits: {must_have_hits}\n"
            "Must-have misses: {must_have_misses}\n"
            "Nice-to-have hits: {nice_to_have_hits}\n"
            "Score: {score}\n\n"
            "Write the explanation:",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Final match report
# ---------------------------------------------------------------------------
MATCH_REPORT_SYSTEM = """You are a recruiting analyst.
Write a concise markdown report summarising the current shortlist.

Include:
  ## Top Candidates
  A short ranked list (name + 1-line summary) of the top candidates.

  ## Key Themes
  2-3 bullets on what the strongest candidates share.

  ## Gaps in the Pool
  2-3 bullets on must-have skills that are under-represented.

  ## Suggested Next Step
  Exactly one sentence recommending what to do next (deeper screen,
  request feedback, schedule interviews, etc.).

Keep the entire report under 250 words."""

match_report_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", MATCH_REPORT_SYSTEM),
        (
            "human",
            "Requirements:\n{requirements}\n\nShortlist (JSON):\n{shortlist}\n\n"
            "Round: {round}",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Feedback interpretation
# ---------------------------------------------------------------------------
INTERPRET_FEEDBACK_SYSTEM = """You are a recruiting assistant.
The user gave free-form feedback on the current shortlist.  Convert it
into a STRUCTURED instruction in JSON:

{{
  "instruction": "<echo of the user's instruction>",
  "boost_skills":   ["skill", ...],
  "penalize_skills":["skill", ...],
  "min_years_override": number or null,
  "approve": true|false
}}

- "boost_skills" = skills/qualities the user wants to weight HIGHER.
- "penalize_skills" = skills/qualities the user wants to weight LOWER.
- "approve" = true ONLY if the user is happy and wants the final recommendation.

Return ONLY JSON.  Lowercase canonical skill names."""

interpret_feedback_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", INTERPRET_FEEDBACK_SYSTEM),
        ("human", "User feedback: {feedback}"),
    ]
)


# ---------------------------------------------------------------------------
# Final hiring recommendation
# ---------------------------------------------------------------------------
FINAL_RECOMMENDATION_SYSTEM = """You are the lead hiring manager.
Based on the final shortlist (already filtered through multiple rounds and
human feedback), produce a final hiring recommendation.

Format (markdown):

### Final Recommendation
**Hire:** <name(s) and 1-line rationale>
**Strong backups:** <name(s) or "None">
**Pass:** <name(s) or "None">

### Reasoning
2-3 sentences explaining the decision, calling out the decisive factors.

### Next Steps
- Bullet 1
- Bullet 2
"""

final_recommendation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", FINAL_RECOMMENDATION_SYSTEM),
        (
            "human",
            "Requirements:\n{requirements}\n\n"
            "Final shortlist (JSON):\n{shortlist}\n\n"
            "Feedback history:\n{feedback_history}",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Candidate comparison
# ---------------------------------------------------------------------------
COMPARE_CANDIDATES_SYSTEM = """You are a senior recruiter.
Compare the candidates side-by-side against the job requirements.

Return a markdown table with columns:
| Candidate | Years | Must-Have Hits | Gaps | Verdict |

After the table, add a "### Recommendation" section (2-3 sentences) naming
the strongest candidate and why."""

compare_candidates_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", COMPARE_CANDIDATES_SYSTEM),
        (
            "human",
            "Requirements:\n{requirements}\n\nCandidates:\n{candidates}",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Interview question generation
# ---------------------------------------------------------------------------
INTERVIEW_QUESTIONS_SYSTEM = """You are an experienced hiring panellist.
Generate interview questions tailored to the candidate's resume and the
job requirements.

Return markdown with these sections:

### Technical (5 questions)
Test the must-have skills with concrete scenarios.

### Behavioural (3 questions)
STAR-format prompts based on the candidate's experience level.

### Project-Specific (3 questions)
Reference real projects or roles from the candidate's resume.
"""

interview_questions_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", INTERVIEW_QUESTIONS_SYSTEM),
        (
            "human",
            "Job requirements:\n{requirements}\n\n"
            "Candidate resume:\n{resume}",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Conversational router (used by matching_agent.run_message)
# ---------------------------------------------------------------------------
ROUTER_SYSTEM = """You are a routing classifier for a resume-screening agent.
Given the user's latest message AND the current agent stage, decide what to do.

Return ONLY JSON:

{{
  "intent":  "new_jd" | "feedback" | "compare" | "interview" | "explain" | "approve" | "chat",
  "targets": [ "candidate name or id", ... ]
}}

Definitions:
- "new_jd"     → user pasted a new job description or asked to start over.
- "feedback"   → user wants to refine the ranking (filter, boost, penalize).
- "compare"    → user asks to compare specific candidates ("compare top 3", "compare X and Y").
- "interview"  → user asks for interview questions.
- "explain"    → user asks WHY a candidate ranked where/above another.
- "approve"    → user is satisfied and wants the final recommendation.
- "chat"       → small talk / clarifying question.

Rules for `targets`:
- Always include EVERY candidate the user named, even if they only used a
  first name (e.g. "elena", "brian") or a nickname.
- Match against the "Known candidates" list and copy the EXACT name from it.
- For comparison phrases like "X over Y", "X vs Y", "X above Y" — include
  BOTH names in the same order they appear in the user's message.
- Pseudo-targets like "top 3", "top 5" are allowed for the `compare` intent.
- Use an empty list ONLY if no candidate is referenced at all.

Examples
--------
User: "why elena over brian"          → {{"intent":"explain","targets":["Elena Rossi","Brian Davies"]}}
User: "compare alice and carla"       → {{"intent":"compare","targets":["Alice Chen","Carla Mendes"]}}
User: "interview questions for daniel"→ {{"intent":"interview","targets":["Daniel Okafor"]}}
User: "only show 5+ years of react"   → {{"intent":"feedback","targets":[]}}
User: "looks good, finalize"          → {{"intent":"approve","targets":[]}}
"""

router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ROUTER_SYSTEM),
        (
            "human",
            "Current stage: {stage}\nKnown candidates: {known_candidates}\n\n"
            "User message: {message}",
        ),
    ]
)
