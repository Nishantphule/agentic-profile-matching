"""
tests/conftest.py
=================

Pytest fixtures + helpers shared by all test flow scripts.

Set `RUN_LIVE_TESTS=1` to enable tests that actually hit the OpenRouter API.
Otherwise the tests are SKIPPED so CI doesn't burn quota.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make `from agents...` importable when the test is run from anywhere.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def live_only(reason: str = "Set RUN_LIVE_TESTS=1 to run") -> pytest.MarkDecorator:
    return pytest.mark.skipif(
        os.getenv("RUN_LIVE_TESTS") != "1", reason=reason
    )


@pytest.fixture(scope="session")
def ingest_resumes():
    """Build the FAISS index once for the whole test session."""
    from config import settings
    from vectorstore.ingest import ResumeIndex

    idx = ResumeIndex()
    n = idx.build_from_dir(settings.resumes_dir)
    if n == 0:
        pytest.skip("No sample resumes found.")
    idx.save()
    return n


@pytest.fixture
def agent(ingest_resumes):  # noqa: ARG001  — ensures the index is built first
    """Fresh `ResumeMatchingAgent` instance per test."""
    from matching_agent import ResumeMatchingAgent

    return ResumeMatchingAgent()


def load_jd(name: str) -> str:
    """Load a sample JD by file name."""
    from config import settings
    return (settings.jds_dir / name).read_text(encoding="utf-8")
