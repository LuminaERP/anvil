"""
Human-in-loop approval gate.
In --yolo mode, auto-approves everything (dev/test only).
Interactive mode prints a diff-style prompt to stderr and reads stdin.
"""
from __future__ import annotations
import os
import sys


_yolo = os.environ.get("AGENT_YOLO", "0") in ("1", "true", "yes")


def set_yolo(value: bool) -> None:
    global _yolo
    _yolo = value


def approve(kind: str, summary: str, details: str = "") -> bool:
    """
    kind: "shell" | "write" | "destructive"
    summary: one-line description shown to the user
    details: optional multi-line context (diff, full command, etc.)
    """
    if _yolo:
        print(f"[YOLO] auto-approve {kind}: {summary}", file=sys.stderr)
        return True

    print(f"\n=== APPROVAL REQUIRED [{kind}] ===", file=sys.stderr)
    print(summary, file=sys.stderr)
    if details:
        print("--- details ---", file=sys.stderr)
        print(details[:4000], file=sys.stderr)
        if len(details) > 4000:
            print(f"... (truncated, {len(details) - 4000} more chars)", file=sys.stderr)
    print("--- approve? [y/N] ---", file=sys.stderr, flush=True)

    if not sys.stdin.isatty():
        print("[non-interactive stdin; denying by default]", file=sys.stderr)
        return False

    try:
        ans = input().strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")
