"""
Episodic memory: every goal, subgoal, lesson, and skill that ever existed.
Backed by SQLite with sqlite-vec for semantic similarity.

Data model:
  sessions(id TEXT PRIMARY KEY, started_at, goal, outcome)
  episodes(id INTEGER PRIMARY KEY, session_id, kind, content, subgoal_id, ts)
  lessons(id INTEGER PRIMARY KEY, session_id, text, severity, tags JSON, ts)
  lesson_vec(lesson_id INTEGER PRIMARY KEY, embedding BLOB)   <- vec0 virtual table

Memory is layered:
  - Local: per-session SQLite file (AGENT_DATA) — full episode history + lessons
  - Shared: cross-session pool (AGENT_SHARED_DATA) — deduplicated transferable
    lessons from retrospectives, queried alongside local at recall time

When a new lesson is added via `add_lesson()`, it lands locally. To propagate
to siblings in a batch, call `publish_lesson()` — retrospectives do this
automatically for lessons of medium+ confidence.
"""
from __future__ import annotations
import json
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Optional

import sqlite_vec
from sentence_transformers import SentenceTransformer

from ..config import CONFIG
from ..state import HistoryEvent, Lesson
from .shared import SharedMemoryPool, get_default_pool

logger = logging.getLogger(__name__)


_EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_EMBED_DIM = 384

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    goal       TEXT NOT NULL,
    outcome    TEXT
);

CREATE TABLE IF NOT EXISTS episodes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    kind       TEXT NOT NULL,
    content    TEXT NOT NULL,
    subgoal_id TEXT,
    data       TEXT,
    ts         REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);

CREATE TABLE IF NOT EXISTS lessons (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    text       TEXT NOT NULL,
    severity   TEXT NOT NULL DEFAULT 'info',
    tags       TEXT,
    ts         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lessons_session ON lessons(session_id);
"""

_VEC_SCHEMA = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS lesson_vec USING vec0(
    lesson_id INTEGER PRIMARY KEY,
    embedding FLOAT[{_EMBED_DIM}]
);
"""


class Memory:
    _embedder: Optional[SentenceTransformer] = None

    def __init__(
        self,
        db_path: Optional[Path] = None,
        shared_pool: Optional[SharedMemoryPool] = None,
        use_shared_pool: bool = True,
    ) -> None:
        self.db_path = Path(db_path) if db_path else CONFIG["paths"].memory_db
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        # Auto-detect shared pool unless caller explicitly opts out
        if shared_pool is not None:
            self.shared = shared_pool
        elif use_shared_pool:
            self.shared = get_default_pool()
        else:
            self.shared = None

        if self.shared:
            logger.debug("Memory: shared pool enabled at %s", self.shared.db_path)

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            conn.executescript(_VEC_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        conn = self._connect()
        try:
            yield conn.cursor()
            conn.commit()
        finally:
            conn.close()

    # ---- embedding ---------------------------------------------------------

    @classmethod
    def embedder(cls) -> SentenceTransformer:
        if cls._embedder is None:
            cls._embedder = SentenceTransformer(_EMBED_MODEL_NAME)
        return cls._embedder

    def _embed(self, text: str) -> bytes:
        import numpy as np
        vec = self.embedder().encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32).tobytes()

    # ---- writes ------------------------------------------------------------

    def start_session(self, goal: str) -> str:
        sid = str(uuid.uuid4())
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO sessions(id, started_at, goal) VALUES (?, ?, ?)",
                (sid, time.time(), goal),
            )
        return sid

    def finish_session(self, session_id: str, outcome: str) -> None:
        with self._cursor() as cur:
            cur.execute("UPDATE sessions SET outcome=? WHERE id=?", (outcome, session_id))

    def log_event(self, session_id: str, event: HistoryEvent) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO episodes(session_id, kind, content, subgoal_id, data, ts) VALUES (?,?,?,?,?,?)",
                (session_id, event.kind, event.content, event.subgoal_id,
                 json.dumps(event.data, default=str), time.time()),
            )

    def add_lesson(self, session_id: str, lesson: Lesson) -> int:
        """Store a lesson in local memory only. Use `publish_lesson()` to also
        broadcast to the shared pool for sibling sessions."""
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO lessons(session_id, text, severity, tags, ts) VALUES (?,?,?,?,?)",
                (session_id, lesson.text, lesson.severity, json.dumps(lesson.tags), time.time()),
            )
            lesson_id = cur.lastrowid
            cur.execute(
                "INSERT INTO lesson_vec(lesson_id, embedding) VALUES (?, ?)",
                (lesson_id, self._embed(lesson.text)),
            )
            return lesson_id

    def publish_lesson(
        self,
        session_id: str,
        lesson: Lesson,
        confidence: str = "medium",
        task_id: Optional[str] = None,
    ) -> tuple[int, Optional[int]]:
        """Store a lesson locally AND publish to the shared pool.

        The shared pool dedupes via semantic similarity — if a sibling session
        has already contributed an equivalent lesson, ref_count is bumped on
        the existing record rather than creating a new one.

        Returns (local_id, shared_id). shared_id is None if no pool is active.
        """
        local_id = self.add_lesson(session_id, lesson)
        shared_id = None
        if self.shared is not None:
            try:
                shared_id = self.shared.publish(
                    text=lesson.text,
                    session_id=session_id,
                    severity=lesson.severity,
                    confidence=confidence,
                    tags=lesson.tags,
                    task_id=task_id,
                )
            except Exception as e:
                logger.warning("publish to shared pool failed: %s", e)
        return local_id, shared_id

    # ---- reads -------------------------------------------------------------

    def recall_lessons(self, query: str, k: int = 5, include_shared: bool = True) -> list[dict]:
        """Semantic KNN over prior lessons. Returns dicts with text+severity+distance.

        When a shared pool is active and `include_shared=True`, results are
        merged from BOTH local memory and shared memory, deduplicated by text
        and sorted by distance.
        """
        local = self._recall_local(query, k=k)
        if not include_shared or self.shared is None:
            return local

        try:
            shared = self.shared.query(query, k=k, min_confidence="low")
        except Exception as e:
            logger.warning("shared pool query failed, returning local only: %s", e)
            return local

        # Merge: dedup by text, keep the lower distance on ties
        by_text: dict[str, dict] = {}
        for row in local:
            by_text[row["text"]] = {**row, "source": "local"}
        for sh in shared:
            key = sh.text
            incoming = {
                "id": f"shared:{sh.id}",
                "text": sh.text,
                "severity": sh.severity,
                "tags": sh.tags,
                "distance": sh.distance if sh.distance is not None else 1.0,
                "source": "shared",
                "confidence": sh.confidence,
                "ref_count": sh.ref_count,
            }
            if key in by_text:
                # Keep the record with lower distance (= closer match)
                if incoming["distance"] < by_text[key]["distance"]:
                    by_text[key] = incoming
            else:
                by_text[key] = incoming

        merged = sorted(by_text.values(), key=lambda r: r.get("distance") or 1.0)[:k]
        return merged

    def _recall_local(self, query: str, k: int = 5) -> list[dict]:
        qvec = self._embed(query)
        with self._cursor() as cur:
            rows = cur.execute(
                """
                SELECT l.id, l.text, l.severity, l.tags, lv.distance
                FROM lesson_vec lv
                JOIN lessons l ON l.id = lv.lesson_id
                WHERE lv.embedding MATCH ? AND k = ?
                ORDER BY lv.distance
                """,
                (qvec, k),
            ).fetchall()
        return [
            {"id": r[0], "text": r[1], "severity": r[2],
             "tags": json.loads(r[3] or "[]"), "distance": r[4]}
            for r in rows
        ]

    def recent_sessions(self, n: int = 5) -> list[dict]:
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT id, goal, outcome, started_at FROM sessions ORDER BY started_at DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [{"id": r[0], "goal": r[1], "outcome": r[2], "started_at": r[3]} for r in rows]

    def events_for_session(self, session_id: str) -> list[dict]:
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT kind, content, subgoal_id, ts FROM episodes WHERE session_id=? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [{"kind": r[0], "content": r[1], "subgoal_id": r[2], "ts": r[3]} for r in rows]
