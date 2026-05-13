"""
vectorstore/ingest.py
=====================

Resume ingestion + FAISS index.

Responsibilities:
  • Walk a directory of resumes (.txt / .pdf / .docx).
  • Extract plain text per resume.
  • Embed each resume into a single vector (one chunk per file — resumes are
    typically short enough that splitting hurts more than it helps).
  • Persist the FAISS index and a side-car JSON metadata file.

The resulting `ResumeIndex` is loaded by `tools.rag_search`.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import faiss
import numpy as np

from config import get_logger, settings
from tools.file_tools import read_resume_text
from vectorstore.embeddings import Embedder

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Metadata record stored alongside the FAISS index
# ---------------------------------------------------------------------------
@dataclass
class ResumeRecord:
    candidate_id: str
    name: str
    file_path: str
    text: str
    years_experience: Optional[float] = None
    skills: List[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers for lightweight, deterministic resume parsing
# ---------------------------------------------------------------------------
_NAME_RE = re.compile(r"^(?:name\s*[:\-]\s*)(.+)$", re.IGNORECASE | re.MULTILINE)
_YEARS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs|yr)\b", re.IGNORECASE
)

# A small canonical skill vocabulary used for quick keyword matching.  The
# LLM does the heavy lifting later; this list just powers the deterministic
# `must_have_hits` calculation in the ranking engine.
CANONICAL_SKILLS: List[str] = [
    # languages
    "python", "java", "javascript", "typescript", "go", "golang", "c++", "c#",
    "rust", "ruby", "scala", "kotlin", "swift", "php",
    # frontend
    "react", "next.js", "vue", "angular", "redux", "tailwind", "html", "css",
    # backend / frameworks
    "node.js", "express", "fastapi", "django", "flask", "spring", "rails",
    # data / ml
    "pandas", "numpy", "pytorch", "tensorflow", "scikit-learn", "spark",
    "hadoop", "kafka", "airflow", "dbt", "snowflake", "bigquery",
    "langchain", "langgraph", "llm", "rag",
    # devops / cloud
    "aws", "gcp", "azure", "docker", "kubernetes", "terraform", "ansible",
    "jenkins", "ci/cd", "github actions",
    # db
    "postgres", "mysql", "mongodb", "redis", "elasticsearch", "dynamodb",
    "faiss", "chromadb", "pinecone",
    # misc
    "graphql", "rest", "microservices", "agile", "scrum",
]


def _guess_name(text: str, fallback: str) -> str:
    m = _NAME_RE.search(text)
    if m:
        return m.group(1).strip().splitlines()[0]
    # heuristic: first non-empty line if it looks like a name (1-4 words, no digits)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        words = line.split()
        if 1 <= len(words) <= 4 and all(w[:1].isalpha() for w in words) and not any(
            ch.isdigit() for ch in line
        ):
            return line
        break
    return fallback


def _guess_years(text: str) -> Optional[float]:
    matches = _YEARS_RE.findall(text)
    if not matches:
        return None
    try:
        return max(float(m) for m in matches)
    except ValueError:
        return None


def _detect_skills(text: str) -> List[str]:
    lowered = text.lower()
    hits = []
    for skill in CANONICAL_SKILLS:
        # word-boundary match (skill names may contain '.', '+', '#', '/')
        pattern = r"(?<![A-Za-z0-9])" + re.escape(skill) + r"(?![A-Za-z0-9])"
        if re.search(pattern, lowered):
            hits.append(skill)
    return sorted(set(hits))


# ---------------------------------------------------------------------------
# Index object
# ---------------------------------------------------------------------------
class ResumeIndex:
    """A FAISS-backed resume index with sidecar metadata."""

    INDEX_FILE = "resumes.faiss"
    META_FILE = "resumes.meta.json"

    def __init__(self, embedder: Optional[Embedder] = None) -> None:
        self.embedder = embedder or Embedder()
        self.records: List[ResumeRecord] = []
        self.index: Optional[faiss.Index] = None

    # -- build / persistence ------------------------------------------------
    def build_from_dir(self, resumes_dir: Path) -> int:
        """Ingest every resume file under `resumes_dir`.  Returns count."""
        resumes_dir = Path(resumes_dir)
        if not resumes_dir.exists():
            log.warning("Resumes dir does not exist: %s", resumes_dir)
            return 0

        records: List[ResumeRecord] = []
        for i, path in enumerate(_walk_resume_files(resumes_dir)):
            try:
                text = read_resume_text(path)
            except Exception as exc:  # noqa: BLE001
                log.warning("Skipping %s: %s", path, exc)
                continue
            if not text.strip():
                continue
            record = ResumeRecord(
                candidate_id=f"cand_{i+1:03d}",
                name=_guess_name(text, fallback=path.stem),
                file_path=str(path),
                text=text,
                years_experience=_guess_years(text),
                skills=_detect_skills(text),
            )
            records.append(record)

        if not records:
            log.warning("No resumes found in %s", resumes_dir)
            self.records = []
            self.index = None
            return 0

        vectors = self.embedder.embed([r.text for r in records])
        index = faiss.IndexFlatIP(self.embedder.dim)  # cosine via inner product
        index.add(vectors)

        self.records = records
        self.index = index
        log.info("Ingested %d resumes into FAISS index", len(records))
        return len(records)

    def save(self, directory: Optional[Path] = None) -> Path:
        directory = Path(directory or settings.vectorstore_dir)
        directory.mkdir(parents=True, exist_ok=True)
        if self.index is None:
            raise RuntimeError("Index is empty; call build_from_dir() first.")
        faiss.write_index(self.index, str(directory / self.INDEX_FILE))
        meta = {
            "embedding_model": self.embedder.model_name,
            "dim": self.embedder.dim,
            "records": [r.to_json() for r in self.records],
        }
        (directory / self.META_FILE).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        log.info("Saved FAISS index → %s", directory)
        return directory

    @classmethod
    def load(cls, directory: Optional[Path] = None) -> "ResumeIndex":
        directory = Path(directory or settings.vectorstore_dir)
        meta_path = directory / cls.META_FILE
        idx_path = directory / cls.INDEX_FILE
        if not meta_path.exists() or not idx_path.exists():
            raise FileNotFoundError(
                f"No FAISS index at {directory}.  Run `python -m vectorstore.ingest`."
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        embedder = Embedder(meta.get("embedding_model"))
        out = cls(embedder=embedder)
        out.index = faiss.read_index(str(idx_path))
        out.records = [ResumeRecord(**r) for r in meta["records"]]
        log.info("Loaded FAISS index with %d resumes", len(out.records))
        return out

    # -- query --------------------------------------------------------------
    def search(self, query: str, top_k: int = 10) -> List[Tuple[ResumeRecord, float]]:
        if not self.records or self.index is None:
            return []
        top_k = min(top_k, len(self.records))
        qv = self.embedder.embed_one(query).reshape(1, -1)
        scores, idxs = self.index.search(qv, top_k)
        out: List[Tuple[ResumeRecord, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            out.append((self.records[idx], float(score)))
        return out

    def get(self, candidate_id: str) -> Optional[ResumeRecord]:
        for r in self.records:
            if r.candidate_id == candidate_id or r.name.lower() == candidate_id.lower():
                return r
        return None


def _walk_resume_files(directory: Path) -> Iterable[Path]:
    exts = {".txt", ".md", ".pdf", ".docx"}
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


# ---------------------------------------------------------------------------
# CLI entrypoint:  python -m vectorstore.ingest
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("Ingesting resumes from %s …", settings.resumes_dir)
    idx = ResumeIndex()
    n = idx.build_from_dir(settings.resumes_dir)
    if n:
        idx.save()
        log.info("Done.  Indexed %d resume(s).", n)
    else:
        log.error("No resumes ingested. Drop .txt/.pdf/.docx files into %s",
                  settings.resumes_dir)


if __name__ == "__main__":
    main()
