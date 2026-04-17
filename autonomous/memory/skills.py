"""
Voyager-style skill library.
A "skill" is a named, reusable sequence of tool calls that accomplished something.
When the agent faces a new subgoal, we semantic-search the skill library and
inject top matches into the planner's context as "you have done this before".
"""
from __future__ import annotations
import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterator, Optional

import sqlite_vec
from .episodic import Memory  # reuse embedder

from ..config import CONFIG

_EMBED_DIM = 384

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS skills (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT UNIQUE NOT NULL,
    description   TEXT NOT NULL,
    trigger_text  TEXT NOT NULL,         -- canonical text for matching
    steps         TEXT NOT NULL,         -- JSON array of {{tool,args,why}}
    success_count INTEGER DEFAULT 1,
    last_used_ts  REAL,
    created_ts    REAL NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS skill_vec USING vec0(
    skill_id INTEGER PRIMARY KEY,
    embedding FLOAT[{_EMBED_DIM}]
);
"""


@dataclass
class SkillStep:
    tool: str
    args: dict
    why: str = ""


@dataclass
class Skill:
    name: str
    description: str
    trigger_text: str
    steps: list[SkillStep] = field(default_factory=list)
    success_count: int = 1


class SkillLibrary:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or CONFIG["paths"].skill_db
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
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

    def _embed(self, text: str) -> bytes:
        import numpy as np
        vec = Memory.embedder().encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32).tobytes()

    def save(self, skill: Skill) -> int:
        steps_json = json.dumps([asdict(s) if isinstance(s, SkillStep) else s for s in skill.steps])
        with self._cursor() as cur:
            # Upsert by name
            existing = cur.execute("SELECT id, success_count FROM skills WHERE name=?", (skill.name,)).fetchone()
            if existing:
                cur.execute(
                    "UPDATE skills SET description=?, trigger_text=?, steps=?, success_count=success_count+1, last_used_ts=? WHERE id=?",
                    (skill.description, skill.trigger_text, steps_json, time.time(), existing[0]),
                )
                sid = existing[0]
            else:
                cur.execute(
                    "INSERT INTO skills(name, description, trigger_text, steps, success_count, last_used_ts, created_ts) VALUES (?,?,?,?,?,?,?)",
                    (skill.name, skill.description, skill.trigger_text, steps_json, skill.success_count, time.time(), time.time()),
                )
                sid = cur.lastrowid
                cur.execute("INSERT INTO skill_vec(skill_id, embedding) VALUES (?,?)", (sid, self._embed(skill.trigger_text)))
        return sid

    def recall(self, query: str, k: int = 3) -> list[dict]:
        qvec = self._embed(query)
        with self._cursor() as cur:
            rows = cur.execute(
                """
                SELECT s.id, s.name, s.description, s.steps, s.success_count, sv.distance
                FROM skill_vec sv
                JOIN skills s ON s.id = sv.skill_id
                WHERE sv.embedding MATCH ? AND k = ?
                ORDER BY sv.distance
                """,
                (qvec, k),
            ).fetchall()
        return [
            {"id": r[0], "name": r[1], "description": r[2],
             "steps": json.loads(r[3]), "success_count": r[4], "distance": r[5]}
            for r in rows
        ]

    def list_all(self) -> list[dict]:
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT id, name, description, success_count, last_used_ts FROM skills ORDER BY last_used_ts DESC NULLS LAST"
            ).fetchall()
        return [{"id": r[0], "name": r[1], "description": r[2], "success_count": r[3], "last_used_ts": r[4]} for r in rows]
