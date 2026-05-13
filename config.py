"""
config.py
=========

Central configuration for the Resume Matching Agent.

Loads environment variables from a local `.env` file (if present) and exposes
a single, type-checked `Settings` object that the rest of the application
imports.  This keeps secrets out of source code and gives every module one
place to look for runtime configuration.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Load .env (no-op if the file does not exist)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------
class Settings(BaseModel):
    """Runtime configuration for the agent."""

    # --- OpenRouter / LLM ---
    openrouter_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    openrouter_base_url: str = Field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
    )
    model: str = Field(
        default_factory=lambda: os.getenv("MODEL", "openai/gpt-oss-120b:free")
    )
    openrouter_referrer: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENROUTER_REFERRER")
    )
    openrouter_title: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENROUTER_TITLE", "Resume Matching Agent")
    )

    # --- Embeddings ---
    embedding_model: str = Field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    # --- Vector store ---
    vectorstore_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("VECTORSTORE_DIR", "./.vectorstore"))
    )

    # --- Data paths ---
    resumes_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("RESUMES_DIR", "./data/resumes"))
    )
    jds_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("JDS_DIR", "./data/sample_jds"))
    )

    # --- Logging ---
    log_level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper()
    )

    # --- LLM call defaults ---
    temperature: float = 0.2
    max_tokens: int = 1500
    request_timeout: int = 60


# Module-level singleton.  Import this everywhere.
settings = Settings()


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Return a configured logger.  Safe to call multiple times."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(settings.log_level)
    logger.propagate = False
    return logger
