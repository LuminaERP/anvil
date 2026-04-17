"""
Filesystem tools: read_file, write_file, list_dir, glob.
Writes are gated by the safety/sandbox + safety/approval modules.
"""
from __future__ import annotations
import fnmatch
from pathlib import Path

from .base import Tool, ToolError, register
from ..safety.sandbox import classify_write
from ..safety.approval import approve
from ..config import CONFIG


_MAX_READ = 200_000  # bytes


def _read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
    p = Path(path)
    if not p.exists():
        raise ToolError(f"file not found: {path}")
    if not p.is_file():
        raise ToolError(f"not a file: {path}")
    try:
        data = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        raise ToolError(f"read failed: {e}")

    lines = data.splitlines(keepends=True)
    if end_line is None:
        end_line = len(lines)
    # Clamp (1-indexed inclusive)
    start_line = max(1, start_line)
    end_line = min(len(lines), end_line)
    if start_line > end_line:
        return f"(empty range {start_line}:{end_line})"
    slab = "".join(lines[start_line - 1 : end_line])
    if len(slab) > _MAX_READ:
        slab = slab[:_MAX_READ] + f"\n... (truncated at {_MAX_READ} bytes)"
    # Prepend line numbers for easier downstream reasoning
    numbered = []
    for i, line in enumerate(slab.splitlines(), start=start_line):
        numbered.append(f"{i:5d} | {line}")
    return "\n".join(numbered) if numbered else "(empty file)"


def _write_file(path: str, content: str) -> str:
    verdict, reason = classify_write(path)
    if verdict == "deny":
        raise ToolError(f"write denied: {reason}")
    if verdict == "review":
        preview = content if len(content) <= 2000 else content[:2000] + f"\n... ({len(content) - 2000} more chars)"
        if not approve("write", f"write {path} ({len(content)} bytes)", preview):
            raise ToolError(f"write not approved by human: {path}")

    # Python syntax guard
    if path.endswith(".py"):
        import ast
        try:
            ast.parse(content)
        except SyntaxError as e:
            return f"REJECTED — would write invalid Python: SyntaxError line {e.lineno}: {e.msg}. File {path} NOT written."

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    baseline = p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
    try:
        p.write_text(content, encoding="utf-8")
    except Exception as e:
        raise ToolError(f"write failed: {e}")

    # Static-analysis gate (F821 undefined-name, F822 undefined-export, F823 used-before-assignment)
    if path.endswith(".py"):
        from ..safety.checks import ruff_check, format_ruff_failure
        res = ruff_check(path, baseline_content=baseline)
        if not res.ok:
            # Rollback
            if baseline is not None:
                p.write_text(baseline, encoding="utf-8")
            else:
                try: p.unlink()
                except OSError: pass
            return format_ruff_failure(res, path)

    # Test regression check (only for edits to non-test source files)
    if path.endswith(".py") and "/test_output/" not in path and baseline is not None:
        try:
            from ..state import SESSION_TEST_BASELINE
            tb = SESSION_TEST_BASELINE.get() if hasattr(SESSION_TEST_BASELINE, "get") else None
        except ImportError:
            tb = None
        if tb is not None:
            from ..safety.checks import test_regression_check, format_regression_failure
            res2 = test_regression_check("/workspace/swarm/test_output", tb)
            if not res2.ok:
                p.write_text(baseline, encoding="utf-8")
                return format_regression_failure(res2)

    return f"wrote {len(content)} bytes to {path}"


def _list_dir(path: str, show_hidden: bool = False) -> str:
    p = Path(path)
    if not p.exists():
        raise ToolError(f"not found: {path}")
    if not p.is_dir():
        raise ToolError(f"not a directory: {path}")
    entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    lines = []
    for e in entries:
        if not show_hidden and e.name.startswith("."):
            continue
        marker = "/" if e.is_dir() else ""
        try:
            size = e.stat().st_size if e.is_file() else ""
        except OSError:
            size = "?"
        lines.append(f"{e.name}{marker}\t{size}")
    return "\n".join(lines) if lines else "(empty)"


def _glob_files(pattern: str, root: str = ".") -> str:
    r = Path(root)
    if not r.exists():
        raise ToolError(f"root not found: {root}")
    # fnmatch wants a relative pattern; we walk to support **
    matches = []
    if "**" in pattern:
        matches = list(r.rglob(pattern.replace("**/", "")))
    else:
        matches = list(r.glob(pattern))
    matches = sorted(str(m.relative_to(r)) for m in matches)[:500]
    return "\n".join(matches) if matches else "(no matches)"


# --- Registrations ----------------------------------------------------------

register(Tool(
    name="read_file",
    description="Read a text file. Returns lines with line numbers. Use start_line/end_line to read ranges (1-indexed, inclusive).",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or cwd-relative path."},
            "start_line": {"type": "integer", "description": "1-indexed start line (default 1).", "default": 1},
            "end_line": {"type": "integer", "description": "1-indexed end line inclusive (default EOF)."},
        },
        "required": ["path"],
    },
    category="read",
    fn=_read_file,
))

register(Tool(
    name="write_file",
    description="Write text content to a file, creating parents as needed. Overwrites existing file. Requires human approval for paths outside workspace.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Target file path."},
            "content": {"type": "string", "description": "Full file contents to write."},
        },
        "required": ["path", "content"],
    },
    category="write",
    fn=_write_file,
))

register(Tool(
    name="list_dir",
    description="List entries in a directory. Shows size for files, trailing / for dirs.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "show_hidden": {"type": "boolean", "default": False},
        },
        "required": ["path"],
    },
    category="read",
    fn=_list_dir,
))

register(Tool(
    name="glob_files",
    description="Find files matching a glob pattern (supports **). Returns up to 500 paths.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern, e.g. **/*.py"},
            "root": {"type": "string", "description": "Root directory to search from.", "default": "."},
        },
        "required": ["pattern"],
    },
    category="read",
    fn=_glob_files,
))
