"""
Targeted code-editing + test-running tools.

edit_file — surgical line-range replacement (much cheaper than rewriting whole files).
apply_patch — unified-diff application via git apply.
run_pytest — structured pytest invocation with summary parsing.
python_eval — safe Python expression eval (math, string munging, data inspection).
"""
from __future__ import annotations
import ast
import json
import re
import subprocess
from pathlib import Path

from .base import Tool, ToolError, register
from ..safety.sandbox import classify_write, classify_shell
from ..safety.approval import approve
from ..config import CONFIG


def _validate_python(text: str) -> str | None:
    """Return error message if text is not valid Python, else None."""
    try:
        ast.parse(text)
        return None
    except SyntaxError as e:
        return f"SyntaxError line {e.lineno}: {e.msg}"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def _edit_file(path: str, start_line: int, end_line: int, new_content: str) -> str:
    """Replace lines start_line..end_line (1-indexed inclusive) with new_content.
    For .py files, auto-validates syntax and ROLLS BACK the change if the result
    doesn't parse. Returns an error observation so the agent can retry."""
    verdict, reason = classify_write(path)
    if verdict == "deny":
        raise ToolError(f"edit denied: {reason}")

    p = Path(path)
    if not p.exists():
        raise ToolError(f"file not found: {path}")

    original_bytes = p.read_bytes()
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    total = len(lines)
    if start_line < 1 or end_line < start_line or start_line > total + 1:
        raise ToolError(f"bad range {start_line}:{end_line} for file with {total} lines")
    end_line = min(end_line, total)

    old_slab = "".join(lines[start_line - 1 : end_line])
    new_slab = new_content if new_content.endswith("\n") or new_content == "" else new_content + "\n"

    if verdict == "review":
        preview = f"--- {path}:{start_line}:{end_line} (old, {len(old_slab)}b) ---\n{old_slab[:800]}\n--- (new, {len(new_slab)}b) ---\n{new_slab[:800]}"
        if not approve("write", f"edit {path} lines {start_line}..{end_line}", preview):
            raise ToolError(f"edit not approved: {path}")

    updated_text = "".join(lines[: start_line - 1] + [new_slab] + lines[end_line:])

    # Gate 1: Python syntax
    if path.endswith(".py"):
        err = _validate_python(updated_text)
        if err:
            return (f"ROLLED BACK — edit would break syntax: {err}\n"
                    f"File {path} was NOT modified. Re-read the surrounding lines "
                    f"and propose a new_content that preserves valid Python structure.")

    # Apply the write so the next gates can see the actual file on disk
    p.write_text(updated_text, encoding="utf-8")

    # Gate 2: static analysis (F821/F822/F823) using ORIGINAL as baseline
    if path.endswith(".py"):
        from ..safety.checks import ruff_check, format_ruff_failure
        original_text = original_bytes.decode("utf-8", errors="replace")
        res = ruff_check(path, baseline_content=original_text)
        if not res.ok:
            # Rollback — restore original bytes
            p.write_bytes(original_bytes)
            return format_ruff_failure(res, path)

    # Gate 3: test regression (only if we're editing a non-test .py inside a test_output-adjacent project)
    if path.endswith(".py") and "/test_output/" not in path:
        from ..safety.checks import test_regression_check, format_regression_failure
        from ..state import SESSION_TEST_BASELINE  # may or may not exist
        baseline = SESSION_TEST_BASELINE.get() if hasattr(SESSION_TEST_BASELINE, "get") else None
        if baseline is not None:
            test_dir = "/workspace/swarm/test_output"
            res2 = test_regression_check(test_dir, baseline)
            if not res2.ok:
                p.write_bytes(original_bytes)
                return format_regression_failure(res2)

    return f"edited {path} lines {start_line}..{end_line}  -{len(old_slab)}b / +{len(new_slab)}b (checks OK)"


def _apply_patch(patch_text: str, cwd: str = ".", check_only: bool = False) -> str:
    """Apply a unified diff via `git apply`. Use check_only=True to dry-run."""
    if not patch_text.strip():
        raise ToolError("empty patch")
    # Write patch to a temp file for git apply
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile("w", suffix=".patch", delete=False) as tf:
        tf.write(patch_text)
        patch_path = tf.name
    args = ["git", "apply"]
    if check_only:
        args.append("--check")
    args.append(patch_path)
    try:
        r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=30)
    except Exception as e:
        raise ToolError(f"git apply failed: {e}")
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        return f"PATCH FAILED (exit={r.returncode}):\n{out}"
    return ("DRY-RUN OK" if check_only else "PATCH APPLIED") + (f"\n{out}" if out else "")


def _run_pytest(target: str = "", cwd: str = ".", extra_args: str = "") -> str:
    cmd = ["python", "-m", "pytest", "-q", "--tb=short", "--no-header"]
    if target:
        cmd.append(target)
    if extra_args:
        cmd.extend(extra_args.split())
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        raise ToolError("pytest timed out after 180s")
    out = (r.stdout or "") + ("\n" + r.stderr if r.stderr else "")
    # Extract summary line
    summary = ""
    for line in reversed(out.splitlines()):
        if re.search(r"\d+ (passed|failed|error|skipped)", line):
            summary = line.strip(); break
    header = f"exit={r.returncode}  summary={summary or '(no summary found)'}"
    if len(out) > 8000:
        out = out[:4000] + "\n... (truncated) ...\n" + out[-4000:]
    return header + "\n" + out


def _python_eval(expression: str) -> str:
    """Safe-ish eval of a Python expression (no statements, no imports, no builtins with side effects)."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ToolError(f"syntax error: {e}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Call,)):
            # allow only a whitelist of callables
            if isinstance(node.func, ast.Name) and node.func.id in {
                "len", "sum", "min", "max", "sorted", "set", "list", "dict", "tuple",
                "abs", "round", "str", "int", "float", "bool", "range", "enumerate",
                "zip", "map", "filter", "any", "all"
            }:
                continue
            raise ToolError(f"call to {ast.unparse(node.func)} not allowed; whitelist only: len/sum/sorted/etc")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ToolError("imports not allowed")
    try:
        result = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {})
    except Exception as e:
        raise ToolError(f"eval error: {e}")
    s = repr(result)
    return s[:4000]


register(Tool(name="edit_file", description="Replace lines start_line..end_line (1-indexed, inclusive) in a file with new_content. Much cheaper than rewriting the whole file. Use this for targeted changes.",
    parameters={"type": "object", "properties": {
        "path": {"type": "string"},
        "start_line": {"type": "integer"},
        "end_line": {"type": "integer"},
        "new_content": {"type": "string", "description": "The replacement text. A trailing newline is added if missing."},
    }, "required": ["path", "start_line", "end_line", "new_content"]},
    category="write", fn=_edit_file))

register(Tool(name="apply_patch", description="Apply a unified-diff patch via `git apply`. Use check_only=true to dry-run first.",
    parameters={"type": "object", "properties": {
        "patch_text": {"type": "string", "description": "Full unified diff."},
        "cwd": {"type": "string", "default": "."},
        "check_only": {"type": "boolean", "default": False},
    }, "required": ["patch_text"]}, category="write", fn=_apply_patch))

register(Tool(name="run_pytest", description="Run pytest. Target is a file/dir/test-id; empty = auto-discovery in cwd.",
    parameters={"type": "object", "properties": {
        "target": {"type": "string", "default": ""},
        "cwd": {"type": "string", "default": "."},
        "extra_args": {"type": "string", "default": "", "description": "Extra args as a single string (e.g. '-k test_foo -x')."},
    }}, category="shell", fn=_run_pytest))

register(Tool(name="python_eval", description="Evaluate a simple Python expression (no imports, only whitelisted builtins: len, sum, min, max, sorted, set, list, dict, tuple, abs, round, str, int, float, bool, range, enumerate, zip, map, filter, any, all).",
    parameters={"type": "object", "properties": {
        "expression": {"type": "string"},
    }, "required": ["expression"]}, category="read", fn=_python_eval))
