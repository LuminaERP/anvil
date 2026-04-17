"""
CLI entrypoint.

Usage:
  python -m autonomous.main "goal text here"
  python -m autonomous.main --yolo "goal text"
  python -m autonomous.main --resume SESSION_ID
  python -m autonomous.main --list-sessions
  python -m autonomous.main --list-skills
  python -m autonomous.main --show-session SESSION_ID
"""
from __future__ import annotations
import argparse
import datetime
import json
import sys
import time
from typing import Any

from .config import CONFIG
from .graph import compile_graph_with_checkpointer
from .memory import Memory, SkillLibrary
from .safety.approval import set_yolo
from .state import AgentState


def _print_event(ev: dict) -> None:
    """Stream a history event to stderr during execution."""
    k = ev.get("kind", "?")
    sub = ev.get("subgoal_id")
    head = f"[{k}]" + (f"[sg={sub}]" if sub else "")
    print(f"{head} {ev.get('content', '')}", file=sys.stderr, flush=True)


def run_goal(goal: str, session_id: str | None = None) -> dict[str, Any]:
    if session_id is None:
        session_id = Memory().start_session(goal)
    else:
        # Resuming — don't start a new session
        pass

    # Snapshot test baseline for regression gate (best-effort; cheap if tests are fast)
    try:
        from .safety.checks import take_test_baseline
        from .state import SESSION_TEST_BASELINE
        tb = take_test_baseline("/workspace/swarm/test_output")
        SESSION_TEST_BASELINE.set(tb)
        print(f"[baseline] {len(tb.ok_tests)} passing, {len(tb.failed_tests)} failing before edits", file=sys.stderr)
    except Exception as e:
        print(f"[baseline] snapshot failed: {e}", file=sys.stderr)

    init_state: AgentState = {
        "goal": goal,
        "session_id": session_id,
        "subgoals": [],
        "current_subgoal_idx": -1,
        "last_tool_calls": [],
        "last_observation": "",
        "lessons": [],
        "consecutive_failures": 0,
        "stuck": False,
        "history": [],
        "memory_context": "",
        "status": "planning",
        "iterations": 0,
        "error": None,
        "final_answer": None,
    }

    graph = compile_graph_with_checkpointer()
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 100}

    print(f"\n=== session {session_id} ===", file=sys.stderr)
    print(f"goal: {goal}\n", file=sys.stderr)

    final_state: dict[str, Any] = {}
    try:
        # Stream state updates as they happen
        for update in graph.stream(init_state, config=config, stream_mode="updates"):
            for node_name, node_out in update.items():
                print(f"\n--- node: {node_name} ---", file=sys.stderr)
                new_events = node_out.get("history") or []
                for ev in new_events:
                    d = ev.__dict__ if hasattr(ev, "__dict__") else ev
                    _print_event(d)
                if node_out.get("status"):
                    print(f"  status -> {node_out['status']}", file=sys.stderr)
                final_state.update(node_out)
    except Exception as e:
        print(f"\nGRAPH EXCEPTION: {e}", file=sys.stderr)
        import traceback; traceback.print_exc(file=sys.stderr)
        Memory().finish_session(session_id, f"failed: {e}")
        return {"session_id": session_id, "status": "failed", "error": str(e)}

    outcome = final_state.get("status", "unknown")
    Memory().finish_session(session_id, outcome)

    return {
        "session_id": session_id,
        "status": outcome,
        "final_answer": final_state.get("final_answer"),
        "iterations": final_state.get("iterations", 0),
    }


def cmd_list_sessions():
    rows = Memory().recent_sessions(20)
    for r in rows:
        ts = datetime.datetime.fromtimestamp(r["started_at"]).isoformat(timespec="seconds")
        print(f"{r['id']}  {ts}  [{r['outcome'] or 'running'}]  {r['goal'][:80]}")


def cmd_show_session(session_id: str):
    for e in Memory().events_for_session(session_id):
        ts = datetime.datetime.fromtimestamp(e["ts"]).isoformat(timespec="seconds")
        sub = f"[sg={e['subgoal_id']}]" if e["subgoal_id"] else ""
        print(f"{ts} [{e['kind']}]{sub} {e['content']}")


def cmd_list_skills():
    rows = SkillLibrary().list_all()
    if not rows:
        print("(no skills saved yet)")
        return
    for r in rows:
        print(f"{r['name']} (used {r['success_count']}x): {r['description']}")


def main() -> int:
    p = argparse.ArgumentParser(prog="autonomous")
    p.add_argument("goal", nargs="?", help="The high-level goal to pursue.")
    p.add_argument("--yolo", action="store_true", help="Auto-approve all shell/write operations (dangerous).")
    p.add_argument("--resume", metavar="SESSION_ID", help="Resume a prior session's checkpoint.")
    p.add_argument("--list-sessions", action="store_true")
    p.add_argument("--show-session", metavar="SESSION_ID")
    p.add_argument("--list-skills", action="store_true")
    args = p.parse_args()

    if args.list_sessions:
        cmd_list_sessions(); return 0
    if args.show_session:
        cmd_show_session(args.show_session); return 0
    if args.list_skills:
        cmd_list_skills(); return 0

    if args.yolo:
        set_yolo(True)

    if args.resume:
        # For resume, we need the original goal. Look it up.
        sessions = Memory().recent_sessions(100)
        match = next((s for s in sessions if s["id"].startswith(args.resume)), None)
        if not match:
            print(f"no session matching {args.resume}", file=sys.stderr); return 2
        result = run_goal(match["goal"], session_id=match["id"])
    elif args.goal:
        result = run_goal(args.goal)
    else:
        # Read goal from stdin (piped)
        goal = sys.stdin.read().strip()
        if not goal:
            p.print_help(sys.stderr); return 2
        result = run_goal(goal)

    print("\n" + "=" * 60)
    print(f"session: {result['session_id']}")
    print(f"status:  {result['status']}")
    print(f"iters:   {result.get('iterations', 0)}")
    print("=" * 60)
    if result.get("final_answer"):
        print(result["final_answer"])
    return 0 if result["status"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
