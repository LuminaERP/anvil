"""`verify_docstring_examples` tool — agent-facing wrapper around doctest_check.

Why expose this as a tool:
  - The agent can call it proactively during execution to self-check
    its work before declaring done
  - It composes with iteration: verify → see failures → edit → verify again
  - Free test cases hidden in the problem itself (no external grader needed)

The reflector also calls into the underlying engine automatically (see
autonomous/nodes/reflector.py) — this tool is the agent-facing half.
"""
from __future__ import annotations
import logging
import os
from pathlib import Path

from .base import Tool, register
from ..safety.doctest_check import find_examples, run_examples

logger = logging.getLogger(__name__)


def _verify(path: str, entry_point: str | None = None, timeout_sec: int = 10) -> str:
    """Run a Python file's docstring examples and return a pass/fail report."""
    workspace_root = Path(os.environ.get("AGENT_WORKSPACE", "."))
    target = Path(path)
    if not target.is_absolute():
        target = workspace_root / path

    if not target.exists():
        return f"[verify error] file not found: {target}"
    if target.suffix != ".py":
        return f"[verify error] not a Python file: {target}"

    try:
        source = target.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"[verify error] could not read file: {e}"

    examples = find_examples(source, entry_point=entry_point)
    if not examples:
        hint = (
            "[verify] No docstring examples found in this file. "
            "Docstring examples look like `>>> f(x)\\n<expected>` or "
            "`f(x) == <expected>` inside the function's docstring."
        )
        return hint

    report = run_examples(target, examples, timeout_sec=float(timeout_sec))
    return report.summary()


register(Tool(
    name="verify_docstring_examples",
    description=(
        "Run a Python function's docstring examples (>>> style or `f(x) == expected` style) "
        "against the current implementation. Returns a structured pass/fail report with the "
        "failing examples and their actual outputs. Use this AFTER writing or editing code "
        "that has docstring examples — the docstring is a contract, and verifying against it "
        "catches algorithm bugs before the task is marked done."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the Python file (absolute, or relative to workspace)",
            },
            "entry_point": {
                "type": "string",
                "description": "Name of a specific function to verify; omit to verify all functions in the file",
            },
            "timeout_sec": {
                "type": "integer",
                "description": "Total timeout budget across all examples (default 10)",
                "default": 10,
            },
        },
        "required": ["path"],
    },
    category="read",
    fn=_verify,
))
