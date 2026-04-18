"""`run_public_tests` tool — agent-facing self-verification for competitive tasks.

When a benchmark task provides visible input/output examples (LCB's
public_test_cases, BCB's public examples), the agent should check its solution
against them before declaring done. Same pattern as docstring verification,
different test format.

The adapter saves the tests to `public_tests.json` in the task workspace at
seed time. The agent calls this tool pointing at its solution file to verify.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from .base import Tool, register
from ..safety.public_test_verifier import verify_public_tests

logger = logging.getLogger(__name__)


def _run_public_tests(
    solution_path: str,
    tests_path: str = "public_tests.json",
    timeout_sec: int = 8,
) -> str:
    """Run public test cases against a solution and return a human-readable report."""
    workspace = Path(os.environ.get("AGENT_WORKSPACE", "."))

    sol = Path(solution_path)
    if not sol.is_absolute():
        sol = workspace / solution_path
    if not sol.exists():
        return f"[public-tests error] solution not found: {sol}"

    tests_file = Path(tests_path)
    if not tests_file.is_absolute():
        tests_file = workspace / tests_path
    if not tests_file.exists():
        return (f"[public-tests error] tests file not found: {tests_file}. "
                f"If your task ships public_test_cases in the prompt, manually create "
                f"{tests_file} as a JSON array of {{input, output, testtype}} objects first.")

    try:
        tests = json.loads(tests_file.read_text(encoding="utf-8"))
        if not isinstance(tests, list):
            return f"[public-tests error] tests file should be a JSON array, got {type(tests).__name__}"
    except json.JSONDecodeError as e:
        return f"[public-tests error] could not parse {tests_file}: {e}"

    try:
        code = sol.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"[public-tests error] could not read {sol}: {e}"

    report = verify_public_tests(code, tests, per_test_timeout=float(timeout_sec))
    return report.summary()


register(Tool(
    name="run_public_tests",
    description=(
        "Run public test cases from a benchmark task against your solution file. "
        "Use this AFTER writing code but BEFORE declaring done — if any public test "
        "fails, fix the solution and re-verify. Public tests are stored as "
        "`public_tests.json` (JSON array of {input, output, testtype='stdin'|'functional'}) "
        "in the workspace; benchmark adapters seed this automatically. Same contract "
        "as verify_docstring_examples but for competitive-programming task formats."
    ),
    parameters={
        "type": "object",
        "properties": {
            "solution_path": {"type": "string", "description": "Path to solution .py file"},
            "tests_path":    {"type": "string", "description": "Path to tests JSON (default: public_tests.json)", "default": "public_tests.json"},
            "timeout_sec":   {"type": "integer", "description": "Per-test timeout", "default": 8},
        },
        "required": ["solution_path"],
    },
    category="read",
    fn=_run_public_tests,
))
