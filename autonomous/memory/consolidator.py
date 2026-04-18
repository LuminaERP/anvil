"""Background consolidation worker for the shared memory pool.

The SharedMemoryPool already knows how to consolidate (merge near-duplicates,
decay stale lessons, promote rehearsed ones); this module schedules that pass
so memory quality doesn't degrade over weeks of runs.

Usage modes:

  1. One-shot: call `run_once(pool)` — useful at session end or between
     benchmark batches.

  2. Scheduled daemon thread: call `start_background(pool, interval_sec=3600)`
     once at application startup; a daemon thread wakes every hour to
     consolidate. Safe to call multiple times (idempotent).

  3. CLI: `python -m autonomous.memory.consolidator` for manual runs.

Consolidation is cheap enough to run hourly — the pool rarely has more than
a few thousand lessons, and the merge/decay pass scans in seconds.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)


_SINGLETON_THREAD: threading.Thread | None = None
_STOP_EVENT: threading.Event | None = None


def run_once(pool=None, **kwargs) -> dict:
    """Run a single consolidation pass. Returns the stats dict."""
    if pool is None:
        from .shared import get_default_pool
        pool = get_default_pool()
        if pool is None:
            return {"error": "no shared pool configured"}
    return pool.consolidate(**kwargs)


def start_background(
    pool=None,
    interval_sec: float = 3600.0,
    jitter_sec: float = 60.0,
) -> threading.Thread:
    """Kick off a daemon thread that calls consolidate() every interval_sec."""
    global _SINGLETON_THREAD, _STOP_EVENT

    if _SINGLETON_THREAD is not None and _SINGLETON_THREAD.is_alive():
        logger.debug("consolidator: background thread already running")
        return _SINGLETON_THREAD

    if pool is None:
        from .shared import get_default_pool
        pool = get_default_pool()
    if pool is None:
        logger.info("consolidator: no shared pool — skipping background consolidation")
        return None  # type: ignore[return-value]

    _STOP_EVENT = threading.Event()
    stop_evt = _STOP_EVENT

    def _loop():
        import random
        # Initial jitter so N parallel workers don't all fire at once
        first_wait = interval_sec + random.uniform(0, jitter_sec)
        if stop_evt.wait(first_wait):
            return
        while not stop_evt.is_set():
            try:
                stats = pool.consolidate()
                logger.info("consolidator background pass: %s", stats)
            except Exception as e:
                logger.warning("consolidator background pass failed: %s", e)
            next_wait = interval_sec + random.uniform(0, jitter_sec)
            if stop_evt.wait(next_wait):
                break

    t = threading.Thread(target=_loop, name="anvil-consolidator", daemon=True)
    t.start()
    _SINGLETON_THREAD = t
    logger.info("consolidator: background thread started, interval=%.0fs", interval_sec)
    return t


def stop_background() -> None:
    """Signal the background thread to exit. Mostly for tests."""
    global _SINGLETON_THREAD, _STOP_EVENT
    if _STOP_EVENT is not None:
        _STOP_EVENT.set()
    if _SINGLETON_THREAD is not None and _SINGLETON_THREAD.is_alive():
        _SINGLETON_THREAD.join(timeout=5)
    _SINGLETON_THREAD = None
    _STOP_EVENT = None


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Consolidate the shared memory pool.")
    ap.add_argument("--decay-days", type=float, default=30.0,
                    help="lessons not retrieved in this many days get decay_score multiplied")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    from .shared import get_default_pool
    pool = get_default_pool()
    if pool is None:
        print(json.dumps({"error": "no shared pool (set AGENT_SHARED_DATA)"}, indent=2))
        raise SystemExit(1)

    print(json.dumps({"before": pool.stats()}, indent=2))
    stats = pool.consolidate(decay_age_days=args.decay_days, dry_run=args.dry_run)
    print(json.dumps({"consolidation": stats}, indent=2))
    print(json.dumps({"after": pool.stats()}, indent=2))


if __name__ == "__main__":
    _cli()
