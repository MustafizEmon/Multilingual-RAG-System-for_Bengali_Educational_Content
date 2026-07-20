from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from app.core.config import SETTINGS, get_logger

_log = get_logger("session")


class SessionMode(str, Enum):
    """The two supported query modes."""
    FRESH = "fresh"        # Mode 1: no previous context
    SESSION = "session"    # Mode 2: last N Q&A pairs available


@dataclass
class QATurn:
    """A single question/answer turn kept only in RAM."""
    question: str
    answer: str


class SessionStore:

    def __init__(self, max_turns: int = SETTINGS.session_max_turns):
        self.max_turns = max_turns
        self._sessions: dict[str, deque[QATurn]] = {}
        self._lock = Lock()

    def _get_or_create(self, session_id: str) -> deque[QATurn]:
        """Return the deque for session_id, creating an empty one if needed."""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = deque(maxlen=self.max_turns)
            return self._sessions[session_id]

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        turns = self._get_or_create(session_id)
        with self._lock:
            turns.append(QATurn(question=question, answer=answer))

    def get_history_pairs(self, session_id: str) -> list[list[str]]:
        with self._lock:
            turns = self._sessions.get(session_id)
            if not turns:
                return []
            return [[t.question, t.answer] for t in turns]

    def format_context(self, session_id: str) -> str | None:
        pairs = self.get_history_pairs(session_id)
        if not pairs:
            return None
        lines = [f"Q: {q}\nA: {a}" for q, a in pairs]
        return "\n\n".join(lines)

    def clear(self, session_id: str) -> bool:
        with self._lock:
            existed = session_id in self._sessions and len(self._sessions[session_id]) > 0
            self._sessions.pop(session_id, None)
            return existed

    def __len__(self) -> int:
        """Number of distinct sessions currently held in memory."""
        with self._lock:
            return len(self._sessions)


# Module-level session store — lives only for this process's lifetime.
# Restarting the service (or a container) resets this to empty automatically;
# nothing needs to be explicitly wiped, by design (no database, no disk).
SESSION_STORE = SessionStore()

def resolve_session_context(session_id: str, mode: SessionMode) -> str | None:
    if mode == SessionMode.FRESH:
        return None
    return SESSION_STORE.format_context(session_id)
