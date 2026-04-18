"""Tool-output truncation — the #1 fix for context-overflow failures.

Every tool's observation string flows into the executor's context window. A
2000-line `read_file` or a 500-entry `list_dir` can eat 30K tokens before the
agent even thinks. This module caps every observation to a budgeted size and
leaves behind a structured follow-up hint so the agent can request more if it
actually needs it.

Integration is intentionally minimal: executor wraps `tool.fn(...)` through
`truncate_observation(name, args, raw)` before appending to the conversation.

Design:
  - Cap per-observation chars (default ~4KB, ≈ 1K tokens)
  - For structured tools (read_file, list_dir, grep) leave a `[truncated, use
    offset=N / show_more(...)]` tail that tells the agent exactly how to paginate
  - Idempotent: running twice yields the same result
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Budget tunable via env for A/B testing
DEFAULT_MAX_CHARS = int(os.environ.get("AGENT_TOOL_OUTPUT_MAX_CHARS", "4096"))
DEFAULT_MAX_LINES = int(os.environ.get("AGENT_TOOL_OUTPUT_MAX_LINES", "120"))

# Tools whose output should always be kept in full (they're fundamentally short)
_SHORT_OUTPUT_TOOLS = frozenset({
    "git_status", "git_branch", "git_log",
    "pskit_git_status", "pskit_git_branch",
    "verify_docstring_examples",
    "which", "pskit_which",
    "python_eval",
})

# Tools whose output is naturally line-structured; truncate by line count
_LINE_STRUCTURED_TOOLS = frozenset({
    "read_file", "read_file_range", "pskit_read_file", "pskit_read_file_range",
    "list_dir", "list_directory", "pskit_list_directory",
    "glob_files", "find_files", "pskit_find_files",
    "grep", "search_code", "pskit_search_code",
    "git_diff", "git_show", "git_blame",
    "pskit_git_diff", "pskit_git_show", "pskit_git_blame",
    "run_bash", "run_command", "pskit_run_command",
})


@dataclass
class TruncationResult:
    text: str
    truncated: bool
    original_chars: int
    shown_chars: int


def truncate_observation(
    tool_name: str,
    args: dict,
    raw_output: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_lines: int = DEFAULT_MAX_LINES,
) -> TruncationResult:
    """Cap a tool observation to `max_chars` chars / `max_lines` lines.

    Returns a `TruncationResult` with the shortened text and metadata. If the
    observation was already under the budget, `truncated=False` and the text
    is returned unchanged.

    For line-structured tools, the truncation is line-aware (keeps whole
    lines) and appends a hint telling the agent how to paginate. For other
    tools, the truncation is char-based with a simple tail marker.
    """
    if not raw_output:
        return TruncationResult(text="", truncated=False, original_chars=0, shown_chars=0)

    original = len(raw_output)

    if tool_name in _SHORT_OUTPUT_TOOLS:
        return TruncationResult(text=raw_output, truncated=False,
                                original_chars=original, shown_chars=original)

    if original <= max_chars:
        # Also check line budget
        line_count = raw_output.count("\n") + 1
        if line_count <= max_lines:
            return TruncationResult(text=raw_output, truncated=False,
                                    original_chars=original, shown_chars=original)

    if tool_name in _LINE_STRUCTURED_TOOLS:
        return _truncate_line_structured(tool_name, args, raw_output, max_chars, max_lines)
    else:
        return _truncate_chars(tool_name, raw_output, max_chars)


def _truncate_line_structured(
    tool_name: str,
    args: dict,
    raw: str,
    max_chars: int,
    max_lines: int,
) -> TruncationResult:
    """Keep whole lines, truncate on either char or line budget, whichever hits first."""
    lines = raw.splitlines()
    total_lines = len(lines)

    # Pick the limit that bites hardest
    budget_chars = max_chars
    kept_lines: list[str] = []
    accumulated = 0
    for i, line in enumerate(lines):
        # +1 for the newline
        if accumulated + len(line) + 1 > budget_chars or i >= max_lines:
            break
        kept_lines.append(line)
        accumulated += len(line) + 1

    if len(kept_lines) == total_lines:
        # No truncation actually needed — was below lines+chars budget after all
        return TruncationResult(text=raw, truncated=False,
                                original_chars=len(raw), shown_chars=len(raw))

    body = "\n".join(kept_lines)
    hint = _build_pagination_hint(tool_name, args, len(kept_lines), total_lines)
    out = body + "\n" + hint
    return TruncationResult(
        text=out,
        truncated=True,
        original_chars=len(raw),
        shown_chars=len(body),
    )


def _truncate_chars(tool_name: str, raw: str, max_chars: int) -> TruncationResult:
    """Simple char-based truncation for free-form tool output."""
    if len(raw) <= max_chars:
        return TruncationResult(text=raw, truncated=False,
                                original_chars=len(raw), shown_chars=len(raw))
    kept = raw[:max_chars]
    remainder = len(raw) - max_chars
    tail = f"\n...[truncated — {remainder:,} more chars]"
    return TruncationResult(
        text=kept + tail,
        truncated=True,
        original_chars=len(raw),
        shown_chars=max_chars,
    )


def _build_pagination_hint(tool_name: str, args: dict, shown: int, total: int) -> str:
    """Give the agent a concrete next-call hint tailored to the tool."""
    remainder = total - shown
    base = f"[truncated: showed {shown} of {total} lines ({remainder} more). "

    if tool_name in ("read_file", "pskit_read_file"):
        path = args.get("path", "<path>")
        return (base + f"To see more, call `read_file_range(path={path!r}, "
                f"start_line={shown + 1}, end_line={min(total, shown + 1 + 200)})`.]")

    if tool_name in ("read_file_range", "pskit_read_file_range"):
        path = args.get("path", "<path>")
        current_end = args.get("end_line", shown)
        return (base + f"To see more, call `read_file_range(path={path!r}, "
                f"start_line={current_end + 1}, end_line={current_end + 200})`.]")

    if tool_name in ("list_dir", "list_directory", "pskit_list_directory"):
        return (base + "Re-call with a more specific path, or use `glob_files` with a pattern.]")

    if tool_name in ("glob_files", "find_files", "pskit_find_files"):
        return (base + "Narrow the pattern to reduce result count.]")

    if tool_name in ("grep", "search_code", "pskit_search_code"):
        return (base + "Narrow the pattern or add `path=` to restrict the search tree.]")

    if tool_name in ("git_diff", "pskit_git_diff"):
        return (base + "Call `git_diff` with an explicit path to scope the diff.]")

    if tool_name in ("run_bash", "run_command", "pskit_run_command"):
        return (base + "Rerun the command with `| head -<N>`, `| tail -<N>`, or a more targeted filter.]")

    return base + "]"
