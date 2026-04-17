"""
Shell + grep tools. Every shell invocation goes through safety/sandbox + approval.
"""
from __future__ import annotations
import subprocess
import time

from .base import Tool, ToolError, register
from ..safety.sandbox import classify_shell
from ..safety.approval import approve
from ..config import CONFIG


def _run_bash(cmd: str, timeout_s: int = 60, cwd: str | None = None) -> str:
    verdict, reason = classify_shell(cmd)
    if verdict == "deny":
        raise ToolError(f"shell denied: {reason}")
    if verdict == "review":
        if not approve("shell", f"run: {cmd}", f"reason: {reason}\ncwd: {cwd or '(default)'}"):
            raise ToolError(f"shell not approved by human: {cmd}")

    start = time.time()
    try:
        proc = subprocess.run(
            ["bash", "-lc", cmd],
            capture_output=True, text=True, timeout=timeout_s, cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        raise ToolError(f"command timed out after {timeout_s}s")
    except Exception as e:
        raise ToolError(f"subprocess failed: {e}")

    dur_ms = int((time.time() - start) * 1000)
    out = proc.stdout or ""
    err = proc.stderr or ""
    if len(out) > CONFIG["budget"].max_context_chars:
        out = out[:CONFIG["budget"].max_context_chars] + f"\n... (stdout truncated)"
    if len(err) > 4000:
        err = err[:4000] + "\n... (stderr truncated)"
    parts = [f"exit={proc.returncode}  duration={dur_ms}ms"]
    if out:
        parts.append("stdout:\n" + out)
    if err:
        parts.append("stderr:\n" + err)
    return "\n".join(parts)


def _grep(pattern: str, path: str = ".", flags: str = "") -> str:
    """
    Regex search across files. Uses rg (ripgrep) when available, falls back to grep -rn.
    flags like '-i' (case-insensitive), '-w' (word), '-F' (literal) are forwarded.
    """
    # Prefer rg for speed; grep -rn as fallback.
    which = subprocess.run(["which", "rg"], capture_output=True, text=True)
    cmd = (
        f"rg --hidden --line-number --no-heading {flags} -- {subprocess_quote(pattern)} {subprocess_quote(path)}"
        if which.returncode == 0
        else f"grep -rn {flags} {subprocess_quote(pattern)} {subprocess_quote(path)}"
    )
    # grep/rg don't need approval — read-only
    try:
        proc = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=30)
    except Exception as e:
        raise ToolError(f"grep failed: {e}")
    out = proc.stdout or "(no matches)"
    if len(out) > 20_000:
        out = out[:20_000] + "\n... (truncated; refine pattern)"
    return out


def subprocess_quote(s: str) -> str:
    # Minimal shell-safe quoting
    if not s or any(c in s for c in " \t\n\"'\\$`!*?[]()<>|&;"):
        return "'" + s.replace("'", "'\\''") + "'"
    return s


register(Tool(
    name="run_bash",
    description="Execute a bash command. Commands matching an allowlist run without confirmation; others require human approval; destructive ones are denied.",
    parameters={
        "type": "object",
        "properties": {
            "cmd": {"type": "string", "description": "The bash command to execute."},
            "timeout_s": {"type": "integer", "default": 60, "description": "Timeout in seconds."},
            "cwd": {"type": "string", "description": "Working directory (optional)."},
        },
        "required": ["cmd"],
    },
    category="shell",
    fn=_run_bash,
))

register(Tool(
    name="grep",
    description="Search files for a regex pattern. Uses ripgrep when available. Read-only, no approval required.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path":    {"type": "string", "default": ".", "description": "File or directory to search."},
            "flags":   {"type": "string", "default": "", "description": "e.g. '-i' for case-insensitive, '-F' for literal."},
        },
        "required": ["pattern"],
    },
    category="read",
    fn=_grep,
))
