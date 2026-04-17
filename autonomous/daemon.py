"""
Continuous-improvement daemon.

Outer loop:
  while not stopped:
    1. if backlog has items → pop one as the goal
       else → meta_planner.propose_next_goal() based on project state + memory
    2. run_goal(goal) via the LangGraph
    3. retrospective.run_retrospective() → lesson + maybe skill
    4. append goal summary to avoid-list (don't immediately repeat)
    5. brief pause, then loop

Stop signals:
  - SIGINT / SIGTERM: clean shutdown after current session
  - writing a file at /workspace/swarm/autonomous/STOP
  - --max-cycles N reaches zero

Backlog: a simple newline-delimited file at /workspace/swarm/autonomous/BACKLOG.txt.
The daemon drains it top-down; users can append to it at any time.
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

from .main import run_goal
from .memory import Memory
from .nodes.meta_planner import propose_next_goal
from .nodes.retrospective import run_retrospective
from .safety.approval import set_yolo
from .config import CONFIG


STOP_FILE = Path("/workspace/swarm/autonomous/STOP")
BACKLOG_FILE = Path("/workspace/swarm/autonomous/BACKLOG.txt")


class StopRequested(Exception):
    pass


def _install_signal_handlers():
    def _handler(signum, frame):
        print(f"\n[daemon] signal {signum} received — stopping after current cycle", file=sys.stderr)
        STOP_FILE.touch()
    signal.signal(signal.SIGINT, _handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handler)


def _pop_backlog_goal() -> Optional[str]:
    if not BACKLOG_FILE.exists():
        return None
    try:
        lines = BACKLOG_FILE.read_text().splitlines()
    except Exception:
        return None
    lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
    if not lines:
        return None
    goal, rest = lines[0], lines[1:]
    try:
        BACKLOG_FILE.write_text("\n".join(rest) + ("\n" if rest else ""))
    except Exception:
        pass
    return goal


def _should_stop() -> bool:
    return STOP_FILE.exists()


def _clear_stop():
    if STOP_FILE.exists():
        try:
            STOP_FILE.unlink()
        except Exception:
            pass


def _log_cycle(cycle: int, goal: str, result: dict, ts: float) -> None:
    audit = CONFIG["paths"].audit_log
    audit.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": datetime.datetime.fromtimestamp(ts).isoformat(),
        "cycle": cycle,
        "goal": goal,
        "result": result,
    }
    with audit.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def run_daemon(
    workspace: str,
    max_cycles: int | None = None,
    pause_s: int = 15,
    yolo: bool = False,
) -> None:
    if yolo:
        set_yolo(True)
    _install_signal_handlers()
    _clear_stop()

    cycle = 0
    recent_goals: list[str] = []

    print(f"[daemon] starting. workspace={workspace} yolo={yolo} max_cycles={max_cycles}", file=sys.stderr)
    print(f"[daemon] append to {BACKLOG_FILE} to queue tasks. `touch {STOP_FILE}` to stop.", file=sys.stderr)

    while True:
        if _should_stop():
            print("[daemon] stop signal — exiting", file=sys.stderr)
            break
        if max_cycles is not None and cycle >= max_cycles:
            print(f"[daemon] reached max_cycles={max_cycles} — exiting", file=sys.stderr)
            break

        cycle += 1
        cycle_start = time.time()
        print(f"\n{'='*60}\n[daemon] CYCLE {cycle}\n{'='*60}", file=sys.stderr)

        # 1. Get goal (backlog first, else meta-planner)
        goal = _pop_backlog_goal()
        source = "backlog"
        if not goal:
            try:
                prop = propose_next_goal(
                    workspace=workspace,
                    backlog=[],
                    avoid=recent_goals[-20:],
                )
                if prop.get("idle"):
                    print(f"[daemon] meta-planner says IDLE — {prop.get('rationale','')}. Pausing for {pause_s*4}s.", file=sys.stderr)
                    time.sleep(pause_s * 4)
                    continue
                goal = prop.get("goal", "").strip()
                if not goal:
                    print("[daemon] meta-planner returned empty goal; sleeping", file=sys.stderr)
                    time.sleep(pause_s)
                    continue
                val = prop.get("_validation") or {}
                source = "meta-fallback" if val.get("fallback") else "meta"
                print(f"[daemon] {source} goal (attempt {val.get('attempt','?')}): {goal[:180]}", file=sys.stderr)
                print(f"[daemon] rationale: {prop.get('rationale','')}", file=sys.stderr)
            except Exception as e:
                print(f"[daemon] meta-planner failed: {e} — sleeping", file=sys.stderr)
                time.sleep(pause_s)
                continue

        print(f"[daemon] goal ({source}): {goal}", file=sys.stderr)

        # 2. Run the goal through the graph
        try:
            result = run_goal(goal)
        except Exception as e:
            print(f"[daemon] run_goal exception: {e}", file=sys.stderr)
            result = {"session_id": None, "status": "crashed", "error": str(e)}

        # 3. Retrospective (only if session_id exists)
        retro = {}
        sid = result.get("session_id")
        if sid:
            try:
                retro = run_retrospective(sid, goal, result.get("status", "unknown"))
                if retro.get("optimization_lesson"):
                    print(f"[daemon] retro lesson: {retro['optimization_lesson']}", file=sys.stderr)
                if retro.get("skill_saved"):
                    print(f"[daemon] skill saved: {retro['skill_saved']}", file=sys.stderr)
            except Exception as e:
                print(f"[daemon] retrospective failed: {e}", file=sys.stderr)

        # 4. Record + move on
        recent_goals.append(goal)
        _log_cycle(cycle, goal, {**result, "retrospective": retro, "source": source, "dur_s": time.time() - cycle_start}, cycle_start)

        # Brief pause so we don't hammer the fleet if something's wrong
        time.sleep(pause_s)

    print("[daemon] done", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(prog="autonomous.daemon")
    p.add_argument("--workspace", default=str(CONFIG["paths"].workspace))
    p.add_argument("--max-cycles", type=int, default=None)
    p.add_argument("--pause-s", type=int, default=15)
    p.add_argument("--yolo", action="store_true")
    args = p.parse_args()
    run_daemon(args.workspace, args.max_cycles, args.pause_s, args.yolo)


if __name__ == "__main__":
    main()
