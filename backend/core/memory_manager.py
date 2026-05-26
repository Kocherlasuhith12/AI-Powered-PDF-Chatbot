"""
backend/core/memory_manager.py
Per-session conversation memory with configurable turn limit.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Turn:
    role: str      # "user" or "assistant"
    content: str


@dataclass
class SessionMemory:
    session_id: str
    active_doc_id: str = ""
    active_filename: str = ""
    turns: List[Turn] = field(default_factory=list)

    def add_turn(self, role: str, content: str) -> None:
        self.turns.append(Turn(role=role, content=content))

    def get_history_text(self, max_turns: int = 10) -> str:
        """Return the last *max_turns* exchanges as a formatted string."""
        recent = self.turns[-(max_turns * 2):]
        lines = []
        for t in recent:
            prefix = "User" if t.role == "user" else "Assistant"
            lines.append(f"{prefix}: {t.content}")
        return "\n".join(lines)

    def get_messages(self, max_turns: int = 10) -> List[Dict[str, str]]:
        """Return as a list of dicts for direct use in Claude's messages API."""
        recent = self.turns[-(max_turns * 2):]
        return [{"role": t.role, "content": t.content} for t in recent]

    def clear(self) -> None:
        self.turns.clear()


class MemoryManager:
    """
    In-memory store of SessionMemory objects keyed by session_id.
    For production: replace _store with Redis or a DB-backed store.
    """

    def __init__(self) -> None:
        self._store: Dict[str, SessionMemory] = defaultdict(
            lambda: SessionMemory(session_id="")
        )

    def get(self, session_id: str) -> SessionMemory:
        if session_id not in self._store:
            self._store[session_id] = SessionMemory(session_id=session_id)
        return self._store[session_id]

    def set_active_doc(self, session_id: str, doc_id: str, filename: str) -> None:
        mem = self.get(session_id)
        if mem.active_doc_id != doc_id:
            # New document — clear prior conversation
            mem.clear()
        mem.active_doc_id = doc_id
        mem.active_filename = filename

    def add_exchange(self, session_id: str, question: str, answer: str) -> None:
        mem = self.get(session_id)
        mem.add_turn("user", question)
        mem.add_turn("assistant", answer)

    def clear_session(self, session_id: str) -> None:
        if session_id in self._store:
            self._store[session_id].clear()

    def list_sessions(self) -> List[str]:
        return list(self._store.keys())


# Module-level singleton
memory_manager = MemoryManager()
