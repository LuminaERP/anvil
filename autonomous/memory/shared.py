"""Cross-session shared memory pool for mid-run lesson propagation.

When agents run in parallel (e.g. a benchmark batch), each task's retrospective
learns lessons that would benefit its still-running siblings. A per-task SQLite
file isolates state for purity, but loses transfer. This module provides the
shared layer that sits alongside per-task memory — writes propagate immediately,
reads pick up freshly-published lessons at the next query.

Design:
  - Same sqlite-vec backend as episodic.Memory, but a distinct database file
  - WAL mode so many readers can query while one writer publishes
  - Semantic dedup on publish: lessons with cosine similarity > 0.92 merge
    into a single record (ref_count bumped)
  - Provenance tracking: we record which session first contributed each lesson
  - Confidence filter on query: callers can exclude low-confidence noise

Auto-detection:
  - If AGENT_SHARED_DATA env var is set, the pool opens a file there
  - Otherwise falls back to ~/.anvil/shared/  (user-level shared brain)
  - If neither path is writable, the pool is disabled silently and Memory
    behaves as before (pure local)
"""
from __future__ import annotations

import collections
import json
import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import sqlite_vec

logger = logging.getLogger(__name__)


_EMBED_DIM = 384  # bge-small-en-v1.5; must match episodic.Memory


_SCHEMA = """
CREATE TABLE IF NOT EXISTS shared_lessons (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    text             TEXT NOT NULL,
    severity         TEXT NOT NULL DEFAULT 'info',
    confidence       TEXT NOT NULL DEFAULT 'medium',  -- 'low' | 'medium' | 'high'
    tags             TEXT,                             -- JSON array
    provenance       TEXT,                             -- JSON {session_id, task_id, contributed_at}
    ref_count        INTEGER NOT NULL DEFAULT 1,       -- bumped on dedup merge + every recall
    created_at       REAL NOT NULL,
    last_retrieved   REAL,
    decay_score      REAL NOT NULL DEFAULT 1.0         -- salience multiplier; drops during consolidation
);
CREATE INDEX IF NOT EXISTS idx_shared_lessons_confidence ON shared_lessons(confidence);
CREATE INDEX IF NOT EXISTS idx_shared_lessons_created ON shared_lessons(created_at);
CREATE INDEX IF NOT EXISTS idx_shared_lessons_retrieved ON shared_lessons(last_retrieved);

-- Associative network: which lessons are retrieved together
CREATE TABLE IF NOT EXISTS shared_co_retrievals (
    lesson_a INTEGER NOT NULL,
    lesson_b INTEGER NOT NULL,
    count    INTEGER NOT NULL DEFAULT 1,
    updated_at REAL NOT NULL,
    PRIMARY KEY (lesson_a, lesson_b)
);
CREATE INDEX IF NOT EXISTS idx_shared_co_retrievals_a ON shared_co_retrievals(lesson_a);
"""

_VEC_SCHEMA = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS shared_lesson_vec USING vec0(
    lesson_id INTEGER PRIMARY KEY,
    embedding FLOAT[{_EMBED_DIM}]
);
"""


_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass
class SharedLesson:
    id: int
    text: str
    severity: str
    confidence: str
    tags: list[str]
    provenance: dict
    ref_count: int
    distance: float | None = None


# ---- auto-detect pool location -----------------------------------------------

def _default_shared_path() -> Path | None:
    """Return the path where the shared pool should live, or None if disabled."""
    explicit = os.environ.get("AGENT_SHARED_DATA")
    if explicit:
        p = Path(explicit)
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p / "shared_lessons.sqlite"
        except OSError as e:
            logger.debug("shared: could not create AGENT_SHARED_DATA dir %s: %s", p, e)
            return None

    # Fall-back: per-user shared brain
    home_shared = Path.home() / ".anvil" / "shared"
    try:
        home_shared.mkdir(parents=True, exist_ok=True)
        return home_shared / "shared_lessons.sqlite"
    except OSError:
        return None


_SINGLETON: "SharedMemoryPool | None" = None
_SINGLETON_SENTINEL: object = object()
_SINGLETON_PATH: Path | None | object = _SINGLETON_SENTINEL


def get_default_pool() -> "SharedMemoryPool | None":
    """Lazy singleton. Returns None if no writable location is available
    OR if AGENT_SHARED_DISABLED=1 is set in env."""
    global _SINGLETON, _SINGLETON_PATH
    if os.environ.get("AGENT_SHARED_DISABLED") == "1":
        return None

    # Resolve path every call so env changes propagate (cheap, just env lookup)
    path = _default_shared_path()
    if _SINGLETON_PATH is _SINGLETON_SENTINEL or _SINGLETON_PATH != path:
        _SINGLETON_PATH = path
        _SINGLETON = SharedMemoryPool(path) if path else None

    return _SINGLETON


# ---- the pool itself ---------------------------------------------------------

class SharedMemoryPool:
    """Ambient knowledge base for lessons that should propagate across sessions.

    Behaves like a simplified hippocampus → neocortex pipeline:

      write path (publish):
        lesson → embed → near-neighbour search → merge-or-insert → confidence tier

      read path (query):
        query → embed → KNN → salience re-rank (distance × recency × ref_count × confidence)
               → update working-memory cache → record co-retrieval pairs

      consolidation (sleep cycle):
        periodic / on-demand:
          - merge clusters whose cosine similarity exceeds threshold into a canonical
            lesson (keeps ref_count = sum)
          - decay lessons not recalled in N days (multiply decay_score by 0.9)
          - promote confidence on lessons with high ref_count (strong rehearsal)
    """

    DEDUP_SIMILARITY_THRESHOLD = 0.92   # cosine similarity > this => merge on publish
    CONSOLIDATE_SIMILARITY_THRESHOLD = 0.90  # slightly looser for sleep-cycle merges
    DEDUP_CANDIDATE_K = 3                # how many near-neighbours to inspect on publish

    # Salience weighting: score = distance * (1/(1+log(1+ref_count))) * recency_penalty * conf_bonus
    CONFIDENCE_BONUS = {"low": 1.3, "medium": 1.0, "high": 0.75}  # lower = better (ranked ascending)
    RECENCY_HALFLIFE_DAYS = 14.0          # lessons not hit in this many days start losing salience

    # Working-memory cache: hot query results stay here for sub-ms repeat access
    WORKING_MEMORY_SLOTS = 24

    # Promotion thresholds during consolidation
    PROMOTE_TO_MEDIUM_AT_REFS = 3
    PROMOTE_TO_HIGH_AT_REFS = 8

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        # Thread-local working memory (per-agent-process scope)
        self._wm_lock = threading.Lock()
        self._wm: "collections.OrderedDict[str, list[SharedLesson]]" = collections.OrderedDict()

        # Associative co-retrieval buffer — flushed to SQL periodically
        self._pending_co_retrievals: list[tuple[int, int]] = []

    # ---- schema ----

    def _init_db(self) -> None:
        conn = self._connect(init=True)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(_SCHEMA)
            conn.executescript(_VEC_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _connect(self, init: bool = False) -> sqlite3.Connection:
        # Short timeout for init (fail fast). Longer for writes (contend with siblings).
        conn = sqlite3.connect(str(self.db_path), timeout=5.0 if init else 30.0)
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

    # ---- embedding ----

    @staticmethod
    def _embed(text: str) -> bytes:
        """Use the same embedder as episodic.Memory so vectors are comparable."""
        # Lazy import to avoid circular
        from .episodic import Memory
        import numpy as np
        vec = Memory.embedder().encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32).tobytes()

    # ---- public API: publish ----

    def publish(
        self,
        text: str,
        session_id: str = "unknown",
        severity: str = "info",
        confidence: str = "medium",
        tags: list[str] | None = None,
        task_id: str | None = None,
    ) -> int:
        """Write a lesson to the shared pool.

        Deduplication: if a lesson with cosine similarity > 0.92 already exists,
        that one's ref_count is bumped instead of inserting a new row. The
        confidence level is upgraded if the new contribution has higher confidence.

        Returns the lesson ID (new or existing).
        """
        if not text or not text.strip():
            raise ValueError("cannot publish empty lesson")

        tags = tags or []
        qvec = self._embed(text)
        now = time.time()
        provenance = {
            "session_id": session_id,
            "task_id": task_id,
            "contributed_at": now,
        }

        # Find near-neighbour candidates
        with self._cursor() as cur:
            existing = cur.execute(
                """
                SELECT l.id, l.text, l.confidence, l.tags, l.ref_count, lv.distance
                FROM shared_lesson_vec lv
                JOIN shared_lessons l ON l.id = lv.lesson_id
                WHERE lv.embedding MATCH ? AND k = ?
                ORDER BY lv.distance
                """,
                (qvec, self.DEDUP_CANDIDATE_K),
            ).fetchall()

            # vec0 uses L2 distance on normalised vectors. For unit vectors:
            #   L2 distance = sqrt(2 - 2 * cos_sim)
            # so cos_sim > 0.92 <=> L2_distance < sqrt(2 - 2 * 0.92) ≈ 0.4
            max_dist = math.sqrt(max(0.0, 2.0 - 2.0 * self.DEDUP_SIMILARITY_THRESHOLD))
            dup = next(
                ((r_id, r_conf, r_tags, r_ref) for r_id, _, r_conf, r_tags, r_ref, dist in existing
                 if dist is not None and dist < max_dist),
                None,
            )

            if dup is not None:
                r_id, r_conf, r_tags, r_ref = dup
                # Merge: bump ref_count, upgrade confidence if higher, merge tags
                new_conf = _max_confidence(r_conf, confidence)
                merged_tags = sorted(set(json.loads(r_tags or "[]")) | set(tags))
                cur.execute(
                    "UPDATE shared_lessons SET ref_count = ?, confidence = ?, tags = ? WHERE id = ?",
                    (r_ref + 1, new_conf, json.dumps(merged_tags), r_id),
                )
                logger.debug("shared.publish: merged with existing lesson id=%d (ref_count %d -> %d)",
                             r_id, r_ref, r_ref + 1)
                return r_id

            # Insert new
            cur.execute(
                "INSERT INTO shared_lessons(text, severity, confidence, tags, provenance, ref_count, created_at) "
                "VALUES (?, ?, ?, ?, ?, 1, ?)",
                (text, severity, confidence, json.dumps(tags),
                 json.dumps(provenance, default=str), now),
            )
            lesson_id = cur.lastrowid
            cur.execute(
                "INSERT INTO shared_lesson_vec(lesson_id, embedding) VALUES (?, ?)",
                (lesson_id, qvec),
            )
            logger.info("shared.publish: new lesson id=%d (%s) from %s", lesson_id, confidence, session_id)
            return lesson_id

    # ---- public API: query ----

    def query(
        self,
        text: str,
        k: int = 5,
        min_confidence: str = "low",
        use_cache: bool = True,
        spread_activation: bool = True,
    ) -> list[SharedLesson]:
        """Salience-weighted KNN query.

        The raw vec-distance KNN is only the first pass. Results are re-ranked
        by a salience score combining:
          - vector distance (closer = better)
          - reference count (more recalls = stronger memory)
          - recency (long unused = penalised via half-life decay)
          - confidence (high-confidence lessons sort above low-confidence)

        If `spread_activation` is on, we also pull in lessons that have been
        frequently co-retrieved with our top hits (associative recall).

        Working-memory cache: a small LRU keyed by (text, k, min_confidence).
        Hot repeat queries skip the SQL round-trip entirely.
        """
        if not text or not text.strip():
            return []

        # Working-memory check
        cache_key = f"{text}::k={k}::mc={min_confidence}::spread={spread_activation}"
        if use_cache:
            with self._wm_lock:
                cached = self._wm.get(cache_key)
                if cached is not None:
                    self._wm.move_to_end(cache_key)
                    return cached

        min_rank = _CONFIDENCE_ORDER.get(min_confidence, 0)
        qvec = self._embed(text)

        # Pull a wider set than k so salience re-ranking can reorder
        fetch_k = max(k * 3, 10)

        with self._cursor() as cur:
            rows = cur.execute(
                """
                SELECT l.id, l.text, l.severity, l.confidence, l.tags, l.provenance,
                       l.ref_count, l.last_retrieved, l.decay_score, lv.distance
                FROM shared_lesson_vec lv
                JOIN shared_lessons l ON l.id = lv.lesson_id
                WHERE lv.embedding MATCH ? AND k = ?
                ORDER BY lv.distance
                """,
                (qvec, fetch_k),
            ).fetchall()

            # Compute salience scores (lower = better)
            now = time.time()
            scored: list[tuple[float, SharedLesson]] = []
            for (lid, ltext, lsev, lconf, ltags, lprov, lref, last_ret, decay, dist) in rows:
                if _CONFIDENCE_ORDER.get(lconf, 0) < min_rank:
                    continue
                salience = self._salience(dist or 1.0, lref, last_ret, decay, lconf, now)
                scored.append((salience, SharedLesson(
                    id=lid,
                    text=ltext,
                    severity=lsev,
                    confidence=lconf,
                    tags=json.loads(ltags or "[]"),
                    provenance=json.loads(lprov or "{}"),
                    ref_count=lref,
                    distance=dist,
                )))

            scored.sort(key=lambda x: x[0])
            selected = [lesson for _, lesson in scored[:k]]

            # Associative spread: pull co-retrieval neighbours of our top hit
            if spread_activation and selected:
                extras = self._associative_recall(cur, selected[0].id, exclude={l.id for l in selected}, k=2)
                # Only add if the extra is at least remotely related (confidence >= low,
                # which is already the min filter) — take first 1-2
                for extra in extras[: max(0, k - len(selected))]:
                    selected.append(extra)

            # Update ref_count + last_retrieved for hits (rehearsal boosts memory)
            if selected:
                ids = [l.id for l in selected]
                placeholders = ",".join(["?"] * len(ids))
                cur.execute(
                    f"UPDATE shared_lessons "
                    f"SET last_retrieved = ?, ref_count = ref_count + 1 "
                    f"WHERE id IN ({placeholders})",
                    (now, *ids),
                )

                # Record co-retrieval pairs (associative learning)
                if len(selected) > 1:
                    ts = now
                    for i, la in enumerate(selected):
                        for lb in selected[i + 1:]:
                            a, b = sorted([la.id, lb.id])
                            cur.execute(
                                "INSERT INTO shared_co_retrievals(lesson_a, lesson_b, count, updated_at) "
                                "VALUES (?, ?, 1, ?) "
                                "ON CONFLICT(lesson_a, lesson_b) DO UPDATE SET "
                                "count = count + 1, updated_at = excluded.updated_at",
                                (a, b, ts),
                            )

        # Update working memory cache
        if use_cache:
            with self._wm_lock:
                self._wm[cache_key] = selected
                self._wm.move_to_end(cache_key)
                while len(self._wm) > self.WORKING_MEMORY_SLOTS:
                    self._wm.popitem(last=False)

        return selected

    @classmethod
    def _salience(
        cls,
        distance: float,
        ref_count: int,
        last_retrieved: float | None,
        decay_score: float,
        confidence: str,
        now: float,
    ) -> float:
        """Lower is better. Combines four signals:

        - distance:   vector distance (already lower=closer)
        - ref_count:  log(1+refs) — rehearsed lessons get a boost
        - recency:    exponential decay based on halflife
        - confidence: multiplier from CONFIDENCE_BONUS (high < medium < low)
        """
        # Rehearsal boost: refs amplify salience (lower the score)
        ref_bonus = 1.0 / (1.0 + 0.25 * math.log1p(ref_count))

        # Recency: exp(-days_since_last_hit / halflife)
        if last_retrieved is None:
            recency_penalty = 1.0
        else:
            days = max(0.0, (now - last_retrieved) / 86400.0)
            recency_penalty = 2.0 - math.exp(-days / cls.RECENCY_HALFLIFE_DAYS)

        conf_mult = cls.CONFIDENCE_BONUS.get(confidence, 1.0)
        decay_mult = max(0.1, decay_score)

        return distance * ref_bonus * recency_penalty * conf_mult / decay_mult

    def _associative_recall(
        self, cur: sqlite3.Cursor, seed_lesson_id: int, exclude: set[int], k: int
    ) -> list[SharedLesson]:
        """Given a seed lesson, return lessons frequently co-retrieved with it
        (spreading activation through the associative graph)."""
        neighbour_rows = cur.execute(
            """
            SELECT
              CASE WHEN lesson_a = ? THEN lesson_b ELSE lesson_a END AS partner,
              count
            FROM shared_co_retrievals
            WHERE (lesson_a = ? OR lesson_b = ?) AND count >= 2
            ORDER BY count DESC
            LIMIT ?
            """,
            (seed_lesson_id, seed_lesson_id, seed_lesson_id, k * 2),
        ).fetchall()

        out: list[SharedLesson] = []
        for partner_id, count in neighbour_rows:
            if partner_id in exclude:
                continue
            row = cur.execute(
                """
                SELECT id, text, severity, confidence, tags, provenance, ref_count
                FROM shared_lessons WHERE id = ?
                """,
                (partner_id,),
            ).fetchone()
            if not row:
                continue
            out.append(SharedLesson(
                id=row[0],
                text=row[1],
                severity=row[2],
                confidence=row[3],
                tags=json.loads(row[4] or "[]"),
                provenance=json.loads(row[5] or "{}"),
                ref_count=row[6],
                distance=None,  # associative, not vector-distance
            ))
            if len(out) >= k:
                break
        return out

    # ---- consolidation (sleep cycle) -----------------------------------------

    def consolidate(
        self,
        merge_similarity_threshold: float | None = None,
        decay_age_days: float = 30.0,
        decay_factor: float = 0.9,
        dry_run: bool = False,
    ) -> dict:
        """Offline consolidation pass — analogous to REM sleep.

        Three operations:
          1. merge: clusters of near-duplicate lessons get collapsed to a
             single canonical entry (the highest-confidence, highest-ref-count
             one wins; others are deleted; ref_counts are summed)
          2. decay: lessons not retrieved in `decay_age_days` get `decay_score`
             multiplied by `decay_factor` (so they rank lower in future queries)
          3. promote: lessons with very high ref_count get their confidence
             bumped up (the brain believes what it rehearses)

        Returns a stats dict with counts of each operation.
        """
        thresh = merge_similarity_threshold or self.CONSOLIDATE_SIMILARITY_THRESHOLD
        now = time.time()
        cutoff = now - decay_age_days * 86400
        stats = {"merged_pairs": 0, "decayed": 0, "promoted": 0, "dry_run": dry_run}

        with self._cursor() as cur:
            # --- merge pass: scan lessons, check if any near-duplicate exists ---
            all_lessons = cur.execute(
                "SELECT id, text, confidence, ref_count FROM shared_lessons ORDER BY ref_count DESC, created_at ASC"
            ).fetchall()
            kept: set[int] = set()
            for (lid, ltext, lconf, lref) in all_lessons:
                if lid in kept:
                    continue
                # Find near-duplicates via vec0
                qvec_row = cur.execute(
                    "SELECT embedding FROM shared_lesson_vec WHERE lesson_id = ?",
                    (lid,),
                ).fetchone()
                if not qvec_row:
                    continue
                qvec = qvec_row[0]
                neighbours = cur.execute(
                    """
                    SELECT l.id, l.confidence, l.ref_count, lv.distance
                    FROM shared_lesson_vec lv
                    JOIN shared_lessons l ON l.id = lv.lesson_id
                    WHERE lv.embedding MATCH ? AND k = ?
                    """,
                    (qvec, 5),
                ).fetchall()

                merge_into_ids = []
                max_d = math.sqrt(max(0.0, 2.0 - 2.0 * thresh))
                for (n_id, n_conf, n_ref, n_dist) in neighbours:
                    if n_id == lid:
                        continue
                    if n_id in kept:
                        continue
                    if n_dist is not None and n_dist < max_d:
                        merge_into_ids.append((n_id, n_conf, n_ref))

                if not merge_into_ids:
                    kept.add(lid)
                    continue

                # Merge: canonical = lid (loop ordered by highest ref_count). Absorb others.
                absorbed_ref_count = sum(ref for _, _, ref in merge_into_ids)
                absorbed_ids = [m[0] for m in merge_into_ids]
                new_conf = lconf
                for _, mconf, _ in merge_into_ids:
                    new_conf = _max_confidence(new_conf, mconf)

                if not dry_run:
                    placeholders = ",".join(["?"] * len(absorbed_ids))
                    cur.execute(
                        "UPDATE shared_lessons SET ref_count = ref_count + ?, confidence = ? WHERE id = ?",
                        (absorbed_ref_count, new_conf, lid),
                    )
                    cur.execute(
                        f"DELETE FROM shared_lesson_vec WHERE lesson_id IN ({placeholders})",
                        absorbed_ids,
                    )
                    cur.execute(
                        f"DELETE FROM shared_lessons WHERE id IN ({placeholders})",
                        absorbed_ids,
                    )
                    # Rewrite co-retrieval edges pointing at absorbed ids
                    for ab in absorbed_ids:
                        cur.execute(
                            "UPDATE OR IGNORE shared_co_retrievals SET lesson_a = ? WHERE lesson_a = ?",
                            (lid, ab),
                        )
                        cur.execute(
                            "UPDATE OR IGNORE shared_co_retrievals SET lesson_b = ? WHERE lesson_b = ?",
                            (lid, ab),
                        )
                        cur.execute("DELETE FROM shared_co_retrievals WHERE lesson_a = lesson_b")

                stats["merged_pairs"] += len(absorbed_ids)
                kept.add(lid)
                for ab in absorbed_ids:
                    kept.add(ab)

            # --- decay pass ---
            decay_rows = cur.execute(
                """
                SELECT id FROM shared_lessons
                WHERE (last_retrieved IS NULL AND created_at < ?)
                   OR (last_retrieved IS NOT NULL AND last_retrieved < ?)
                """,
                (cutoff, cutoff),
            ).fetchall()
            if decay_rows and not dry_run:
                ids = [r[0] for r in decay_rows]
                placeholders = ",".join(["?"] * len(ids))
                cur.execute(
                    f"UPDATE shared_lessons SET decay_score = decay_score * ? WHERE id IN ({placeholders})",
                    (decay_factor, *ids),
                )
            stats["decayed"] = len(decay_rows)

            # --- promotion pass ---
            promoted = cur.execute(
                """
                SELECT id, ref_count, confidence FROM shared_lessons
                WHERE (confidence = 'low' AND ref_count >= ?)
                   OR (confidence = 'medium' AND ref_count >= ?)
                """,
                (self.PROMOTE_TO_MEDIUM_AT_REFS, self.PROMOTE_TO_HIGH_AT_REFS),
            ).fetchall()
            if promoted and not dry_run:
                for (lid, ref, conf) in promoted:
                    new_conf = "high" if ref >= self.PROMOTE_TO_HIGH_AT_REFS else "medium"
                    cur.execute("UPDATE shared_lessons SET confidence = ? WHERE id = ?", (new_conf, lid))
            stats["promoted"] = len(promoted)

        # Invalidate working memory — data has changed
        with self._wm_lock:
            self._wm.clear()

        logger.info("shared.consolidate: %s", stats)
        return stats

    def invalidate_cache(self) -> None:
        """Manually clear the working-memory cache. Normally done on consolidation."""
        with self._wm_lock:
            self._wm.clear()

    # ---- public API: stats ----

    def stats(self) -> dict:
        with self._cursor() as cur:
            total = cur.execute("SELECT COUNT(*) FROM shared_lessons").fetchone()[0]
            by_conf = {}
            for row in cur.execute("SELECT confidence, COUNT(*) FROM shared_lessons GROUP BY confidence"):
                by_conf[row[0]] = row[1]
            latest = cur.execute(
                "SELECT text, confidence, created_at FROM shared_lessons ORDER BY created_at DESC LIMIT 5"
            ).fetchall()
            popular = cur.execute(
                "SELECT text, ref_count FROM shared_lessons ORDER BY ref_count DESC LIMIT 5"
            ).fetchall()

        return {
            "path": str(self.db_path),
            "total_lessons": total,
            "by_confidence": by_conf,
            "latest": [{"text": r[0][:100], "confidence": r[1], "created_at": r[2]} for r in latest],
            "most_recalled": [{"text": r[0][:100], "ref_count": r[1]} for r in popular],
        }

    def clear(self) -> int:
        """Wipe the pool — used by tests and fresh benchmark runs."""
        with self._cursor() as cur:
            before = cur.execute("SELECT COUNT(*) FROM shared_lessons").fetchone()[0]
            cur.execute("DELETE FROM shared_lesson_vec")
            cur.execute("DELETE FROM shared_lessons")
        return before


# ---- helpers -----------------------------------------------------------------

def _max_confidence(a: str, b: str) -> str:
    """Return whichever of a or b has higher confidence rank."""
    ra = _CONFIDENCE_ORDER.get(a, 0)
    rb = _CONFIDENCE_ORDER.get(b, 0)
    return a if ra >= rb else b
