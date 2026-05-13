"""
agents/memory.py
================

Conversation memory wrapper.

LangGraph already supports persistent state via `MemorySaver`, but exposing a
small wrapper keeps the rest of the codebase decoupled from the specific
checkpointer and lets us inject e.g. a SQLite / Redis backend later.
"""

from __future__ import annotations

from typing import Optional

from langgraph.checkpoint.memory import MemorySaver


class ConversationMemory:
    """
    Thin wrapper around LangGraph's `MemorySaver` (in-process checkpoints).

    Each chat session is identified by a `thread_id`.  When the LangGraph
    runner is invoked with `config={"configurable": {"thread_id": ...}}`, the
    checkpointer transparently persists state between turns.
    """

    def __init__(self) -> None:
        self._saver = MemorySaver()

    @property
    def checkpointer(self) -> MemorySaver:
        return self._saver

    @staticmethod
    def config_for(thread_id: str, recursion_limit: int = 50) -> dict:
        """Build the `config` dict expected by `graph.invoke / stream`."""
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": recursion_limit,
        }

    def state(self, graph, thread_id: str) -> Optional[dict]:
        """Fetch the current persisted state for a thread (for inspection / UI)."""
        try:
            snapshot = graph.get_state(self.config_for(thread_id))
            return snapshot.values if snapshot else None
        except Exception:  # noqa: BLE001
            return None
