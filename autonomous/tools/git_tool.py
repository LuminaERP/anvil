"""
Git tools: status, diff, log, show, branch. Read-only by default.
Git write ops (commit, push) are gated heavily via safety.
"""
from __future__ import annotations
import subprocess

from .base import Tool, ToolError, register
from ..safety.approval import approve


def _git(subcmd: list[str], cwd: str = ".", timeout_s: int = 30) -> str:
    try:
        r = subprocess.run(["git", *subcmd], cwd=cwd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise ToolError(f"git {' '.join(subcmd)} timed out")
    except Exception as e:
        raise ToolError(f"git invocation failed: {e}")
    out = r.stdout or ""
    err = r.stderr or ""
    if len(out) > 30_000:
        out = out[:30_000] + "\n... (truncated)"
    if r.returncode != 0:
        return f"exit={r.returncode}\n{err}\n{out}"
    return out or "(empty output)"


def _git_status(cwd: str = ".") -> str:
    return _git(["status", "--short", "--branch"], cwd=cwd)


def _git_diff(cwd: str = ".", path: str = "", staged: bool = False) -> str:
    args = ["diff"]
    if staged:
        args.append("--staged")
    if path:
        args.extend(["--", path])
    return _git(args, cwd=cwd)


def _git_log(cwd: str = ".", n: int = 10, path: str = "") -> str:
    args = ["log", f"-{n}", "--oneline", "--no-decorate"]
    if path:
        args.extend(["--", path])
    return _git(args, cwd=cwd)


def _git_show(cwd: str = ".", ref: str = "HEAD") -> str:
    return _git(["show", "--stat", ref], cwd=cwd)


def _git_blame(cwd: str = ".", path: str = "", start: int | None = None, end: int | None = None) -> str:
    if not path:
        raise ToolError("path required")
    args = ["blame", "--line-porcelain" if False else "-l"]
    if start and end:
        args.extend(["-L", f"{start},{end}"])
    args.extend(["--", path])
    return _git(args, cwd=cwd)


register(Tool(name="git_status", description="Show working tree status (short form) and current branch.",
    parameters={"type": "object", "properties": {"cwd": {"type": "string", "default": "."}}}, category="read", fn=_git_status))
register(Tool(name="git_diff", description="Show uncommitted diff. Use staged=true for staged-only. path= for one file.",
    parameters={"type": "object", "properties": {
        "cwd": {"type": "string", "default": "."},
        "path": {"type": "string", "default": ""},
        "staged": {"type": "boolean", "default": False},
    }}, category="read", fn=_git_diff))
register(Tool(name="git_log", description="Show recent commits (oneline).",
    parameters={"type": "object", "properties": {
        "cwd": {"type": "string", "default": "."},
        "n": {"type": "integer", "default": 10},
        "path": {"type": "string", "default": ""},
    }}, category="read", fn=_git_log))
register(Tool(name="git_show", description="Show a commit with stats. ref defaults to HEAD.",
    parameters={"type": "object", "properties": {
        "cwd": {"type": "string", "default": "."},
        "ref": {"type": "string", "default": "HEAD"},
    }}, category="read", fn=_git_show))
register(Tool(name="git_blame", description="Show git blame for a file or line range.",
    parameters={"type": "object", "properties": {
        "cwd": {"type": "string", "default": "."},
        "path": {"type": "string"},
        "start": {"type": "integer"},
        "end": {"type": "integer"},
    }, "required": ["path"]}, category="read", fn=_git_blame))
