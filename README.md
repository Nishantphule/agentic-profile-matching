# 🧭 Resume Matching Agent

> Made by **Nishant Phule**™ · © 2026 · All rights reserved.

A production-quality, conversational AI agent that parses job descriptions,
searches a resume library with **retrieval-augmented generation (RAG)**,
ranks and explains candidates, supports human-in-the-loop refinement,
runs **multi-round screening**, and produces a final hiring recommendation.

Built with **Python**, **LangGraph**, **LangChain**, **FAISS**, **SentenceTransformers**,
**Streamlit**, and the **OpenRouter** API (`openai/gpt-oss-120b:free`).

---

## ✨ Features

|   | Capability |
|---|---|
| ✅ | LangGraph workflow with conditional edges and a human-feedback **interrupt** |
| ✅ | Structured requirement extraction (must-have vs nice-to-have, years, certs) |
| ✅ | FAISS-based RAG search over resumes (`.txt` / `.md` / `.pdf` / `.docx`) |
| ✅ | Explainable weighted scoring (must-have coverage + experience + semantic + feedback shaping) |
| ✅ | Conversational refinement — *"only show 5+ years"*, *"boost Kafka"*, *"compare top 3"* |
| ✅ | Multi-round screening (10 → 5 → 3) with round tracking per candidate |
| ✅ | Side-by-side candidate comparison tables + LLM verdict |
| ✅ | Tailored interview question generation (technical / behavioural / project) |
| ✅ | Persistent conversation memory (`MemorySaver`) keyed per session |
| ✅ | Streamlit chat UI with live shortlist, expanders, and quick-actions |
| ✅ | 5 end-to-end conversation test flows (pytest + interactive demos) |

---

## 🏗 Architecture

```mermaid
graph TD
    A([START]) --> B[Parse JD]
    B --> C[Extract Requirements]
    C --> D[Search Resumes - FAISS RAG]
    D -->|empty pool| Z([END])
    D --> E[Rank Candidates]
    E --> F[Generate Match Report]
    F --> G{Human Feedback (interrupt)}
    G -->|refine| H[Re-rank]
    H --> F
    G -->|approve| I[Final Recommendation]
    I --> Z([END])
```

Full breakdown of nodes, edges, and state ↦ see [`architecture_diagram.md`](./architecture_diagram.md).

```
resume_matching_agent/
├── app.py                 ← Streamlit entrypoint
├── matching_agent.py      ← Conversational façade over the LangGraph
├── config.py              ← Loads .env, exposes Settings + logger
├── llm_client.py          ← ChatOpenAI (OpenRouter base_url)
├── requirements.txt
├── README.md
├── architecture_diagram.md
├── .env.example
│
├── agents/                ← LangGraph state, prompts, nodes, graph, memory
├── tools/                 ← requirement_extractor / rag_search / ranking_engine
│                            candidate_comparator / interview_generator / file_tools
├── vectorstore/           ← embeddings.py + ingest.py (FAISS)
├── ui/                    ← Streamlit components + chat_interface
├── data/
│   ├── resumes/           ← drop resumes here
│   └── sample_jds/        ← three sample JDs included
├── tests/                 ← test_flow_1..5.py (pytest + interactive)
└── diagrams/              ← state_machine.md (Mermaid)
```

---

## 🚀 Quick start

### 1. Install

```bash
# (recommended) create a virtualenv
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> The first run will download the SentenceTransformer model
> (`all-MiniLM-L6-v2`, ~80 MB). That's a one-time cost.

### 2. Configure OpenRouter

Copy the example env file and edit if needed:

```bash
# Windows (PowerShell)
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

The file already contains:

```env
OPENROUTER_API_KEY=sk-or-v1-...your-key...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MODEL=openai/gpt-oss-120b:free
```

> Get your own key at <https://openrouter.ai/keys>.

### 3. Ingest the sample resumes into FAISS

```bash
python -m vectorstore.ingest
```

You should see something like:

```
INFO | vectorstore.ingest | Ingested 8 resumes into FAISS index
INFO | vectorstore.ingest | Saved FAISS index → .vectorstore
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

Open the URL it prints (usually <http://localhost:8501>).

### 5. Try it out

* Paste any JD from `data/sample_jds/` into the **Paste JD** tab.
* Click **🚀 Start matching**.
* In the chat box, try:
  * *"Compare top 3"*
  * *"Only show candidates with 5+ years of experience"*
  * *"Re-rank based on Kafka and AWS certifications"*
  * *"Why did Alice rank above Brian?"*
  * *"Interview questions for Carla"*
  * *"Approve — please finalize"*

---

## 🧪 Command-line use

You can run the agent as a pure CLI:

```bash
python -m matching_agent --jd data/sample_jds/senior_react_engineer.txt
```

This prints the initial match report and drops you into an interactive
prompt where every line is sent to the agent until you `approve` or `quit`.

---

## 🧪 Running the test flows

The 5 conversation tests cover the assignment scenarios end-to-end.
Each file works as **both** a pytest test and a standalone interactive demo.

```bash
# Interactive walkthroughs (always work)
python tests/test_flow_1.py    # basic hiring query
python tests/test_flow_2.py    # requirement refinement
python tests/test_flow_3.py    # candidate comparison
python tests/test_flow_4.py    # feedback-based reranking
python tests/test_flow_5.py    # final hiring recommendation

# Automated assertions (uses live LLM calls; opt-in)
RUN_LIVE_TESTS=1 pytest -q tests/
```

> The pytest variants are skipped by default so they don't burn API quota
> in CI.  Set `RUN_LIVE_TESTS=1` to enable them.

---

## 🖼 Screenshots

> Placeholder — drop your own screenshots into a `docs/` folder and link
> them here.  Suggested captures:
>
> 1. The initial match report with the live shortlist table.
> 2. A multi-round screening funnel after two rounds of feedback.
> 3. A candidate comparison view with the LLM verdict at the bottom.
> 4. Generated interview questions for a top candidate.
>
> ```
> ![App overview](docs/screenshot_overview.png)
> ![Shortlist table](docs/screenshot_shortlist.png)
> ![Comparison](docs/screenshot_compare.png)
> ![Interview questions](docs/screenshot_interview.png)
> ```

---

## 💬 Example prompts

| Intent              | Example user message                                            |
|---------------------|-----------------------------------------------------------------|
| New JD              | (paste the entire JD text)                                      |
| Filter              | *"Only show candidates with 5+ years of React experience."*     |
| Boost a skill       | *"Re-rank based on backend-heavy experience and Kafka."*        |
| Penalise            | *"Down-rank anyone whose primary stack is Vue."*                |
| Compare             | *"Compare top 3 candidates."* / *"Compare Alice and Carla."*     |
| Explain             | *"Why did Daniel rank above Frank?"*                            |
| Interview questions | *"Interview questions for Hiroshi."*                            |
| Approve             | *"Approve — finalize."* / *"Looks great, give me the final."*    |

---

## 🛠 Configuration reference

All settings live in `config.py` and are read from `.env`:

| Variable               | Default                                                    | Purpose                                    |
|------------------------|------------------------------------------------------------|--------------------------------------------|
| `OPENROUTER_API_KEY`   | *(required)*                                               | OpenRouter API key                         |
| `OPENROUTER_BASE_URL`  | `https://openrouter.ai/api/v1`                             | OpenRouter API base URL                    |
| `MODEL`                | `openai/gpt-oss-120b:free`                                 | LLM model slug                             |
| `EMBEDDING_MODEL`      | `sentence-transformers/all-MiniLM-L6-v2`                   | Local embedding model                      |
| `VECTORSTORE_DIR`      | `./.vectorstore`                                           | Where FAISS index + metadata are persisted |
| `RESUMES_DIR`          | `./data/resumes`                                           | Resumes to ingest                          |
| `JDS_DIR`              | `./data/sample_jds`                                        | Sample JDs                                 |
| `LOG_LEVEL`            | `INFO`                                                     | Logging verbosity                          |

---

## 🧠 How the scoring works

```
score = 0.45 · must-have-coverage
      + 0.15 · nice-to-have-coverage
      + 0.20 · experience-fit
      + 0.20 · semantic-similarity
      + boost / penalty adjustments from human feedback
```

Scores are normalised to 0–100 and ordered descending.  See
`tools/ranking_engine.py` for the math and `agents/nodes.py` for how the
ranking is woven into the multi-round workflow.

---

## 🌱 Future improvements

* **PII redaction** — automatically mask names / emails before sending
  resume text to the LLM (privacy / bias hygiene).
* **Streaming** — surface token-by-token LLM responses in the chat panel.
* **Better PDF parsing** — switch from `pypdf` to `unstructured` or `marker`
  for layout-aware extraction (tables, two-column resumes).
* **Bias audit** — run periodic fairness checks on the ranking distribution
  per demographic proxy.
* **Persistent thread store** — replace `MemorySaver` with a SQLite or
  Redis checkpointer so conversations survive process restarts.
* **Recruiter notes** — let the user attach private notes to a candidate
  that travel through subsequent rounds.
* **Active ingestion** — watch `data/resumes/` and auto-re-index on file
  changes.

---

## 📜 License & attribution

© 2026 **Nishant Phule**™ — All rights reserved.

Released under the MIT license (see `LICENSE` — add your own copy if you
fork this project).

---

## 🙏 Acknowledgements

* [LangGraph](https://github.com/langchain-ai/langgraph) for the
  graph-based agent runtime.
* [LangChain](https://github.com/langchain-ai/langchain) for the LLM
  primitives and OpenAI-compatible client.
* [OpenRouter](https://openrouter.ai) for free access to a strong
  open-weights model.
* [SentenceTransformers](https://www.sbert.net/) and
  [FAISS](https://github.com/facebookresearch/faiss) for fast local
  retrieval.
