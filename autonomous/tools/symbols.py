"""Symbol-aware code navigation tools.

SWE-bench (and to a lesser extent BCB) surfaces the need for code-aware
navigation. The agent currently reads entire 500-line files to find a 10-line
bug — that's 50x more context than it needs.

Two new tools:

  list_symbols(path)        → file structure: top-level classes, functions,
                              decorators, line ranges. Bodies elided.
  read_symbol(path, name)   → just the named symbol's source (def/class body)

Both are AST-based (stdlib `ast`) so they work offline, no LSP server needed.
Fallback regex scan for non-Python files.
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Optional

from .base import Tool, ToolError, register


# ---- list_symbols ------------------------------------------------------------

def _list_symbols(path: str) -> str:
    """Return a structural outline of a source file — no function bodies."""
    p = _resolve_path(path)
    if not p.exists():
        raise ToolError(f"file not found: {p}")
    if not p.is_file():
        raise ToolError(f"not a file: {p}")

    try:
        source = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise ToolError(f"could not read {p}: {e}")

    if p.suffix == ".py":
        return _python_outline(source, str(p))
    return _regex_outline(source, str(p))


def _python_outline(source: str, path_str: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"[list_symbols] {path_str}: SYNTAX ERROR at line {e.lineno}: {e.msg}"

    lines = [f"[symbols in {path_str}]"]

    # Module-level imports: summarise in one line
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            imports.extend(f"{mod}.{a.name}" for a in node.names)
    if imports:
        preview = ", ".join(imports[:8])
        if len(imports) > 8:
            preview += f", ... +{len(imports) - 8} more"
        lines.append(f"  imports: {preview}")

    # Top-level definitions
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorators = "".join(f"@{_format_decorator(d)} " for d in node.decorator_list)
            start = node.lineno
            end = _end_line(node)
            args = _format_args(node.args)
            async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
            lines.append(f"  {start:4d}-{end:<4d} {decorators}{async_prefix}def {node.name}({args})")
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(_format_node(b) for b in node.bases)
            start = node.lineno
            end = _end_line(node)
            decorators = "".join(f"@{_format_decorator(d)} " for d in node.decorator_list)
            lines.append(f"  {start:4d}-{end:<4d} {decorators}class {node.name}({bases})")
            # Methods one indent deeper
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    c_start = child.lineno
                    c_end = _end_line(child)
                    c_args = _format_args(child.args)
                    c_async = "async " if isinstance(child, ast.AsyncFunctionDef) else ""
                    lines.append(f"    {c_start:4d}-{c_end:<4d} {c_async}def {child.name}({c_args})")
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.isupper() or name.startswith("_"):
                lines.append(f"  {node.lineno:4d}      {name} = ... (module const)")

    return "\n".join(lines)


def _regex_outline(source: str, path_str: str) -> str:
    """Fallback for non-Python files: pick out function/class-like declarations."""
    lines = [f"[symbols in {path_str}] (regex fallback — non-Python file)"]
    ext = Path(path_str).suffix.lower()
    patterns = {
        ".js":  [r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)", r"^\s*class\s+(\w+)"],
        ".ts":  [r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)", r"^\s*class\s+(\w+)",
                 r"^\s*(?:export\s+)?interface\s+(\w+)"],
        ".go":  [r"^\s*func\s+(?:\([^)]+\)\s+)?(\w+)", r"^\s*type\s+(\w+)\s+(?:struct|interface)"],
        ".rs":  [r"^\s*(?:pub\s+)?fn\s+(\w+)", r"^\s*(?:pub\s+)?(?:struct|enum|trait)\s+(\w+)"],
        ".java": [r"^\s*(?:public|private|protected)?\s*(?:static\s+)?\w+\s+(\w+)\s*\([^)]*\)\s*\{",
                 r"^\s*(?:public\s+)?class\s+(\w+)"],
        ".cpp": [r"^\s*\w+(?:\s*\*)?\s+(\w+)\s*\([^)]*\)\s*\{", r"^\s*class\s+(\w+)"],
        ".c":   [r"^\s*\w+(?:\s*\*)?\s+(\w+)\s*\([^)]*\)\s*\{"],
    }
    file_patterns = patterns.get(ext, [r"^\s*function\s+(\w+)", r"^\s*class\s+(\w+)"])
    compiled = [re.compile(p) for p in file_patterns]

    for lineno, line in enumerate(source.splitlines(), start=1):
        for pat in compiled:
            m = pat.match(line)
            if m:
                lines.append(f"  {lineno:4d}  {line.strip()[:100]}")
                break

    if len(lines) == 1:
        lines.append("  (no symbol-like declarations found; use grep or read_file)")
    return "\n".join(lines)


# ---- read_symbol -------------------------------------------------------------

def _read_symbol(path: str, name: str) -> str:
    """Return source of just one class/function, identified by name.

    Supports `ClassName.method_name` syntax to target a method within a class.
    """
    p = _resolve_path(path)
    if not p.exists():
        raise ToolError(f"file not found: {p}")
    source = p.read_text(encoding="utf-8", errors="replace")

    if p.suffix != ".py":
        return _regex_find_symbol(source, name, str(p))

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"[read_symbol] {p}: SYNTAX ERROR at line {e.lineno}: {e.msg}"

    parts = name.split(".", 1)
    parent_name = parts[0]
    child_name = parts[1] if len(parts) > 1 else None

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name != parent_name:
                continue

            if child_name is None:
                # Return the whole parent node
                return _slice_source(source, node, str(p))

            # Drill into a class
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == child_name:
                        return _slice_source(source, child, str(p))
                raise ToolError(f"class {parent_name} has no method named {child_name}")
            raise ToolError(f"{parent_name} is not a class — cannot drill into {child_name}")

    raise ToolError(f"symbol {name!r} not found in {p}")


def _slice_source(source: str, node: ast.AST, path_str: str) -> str:
    start = node.lineno
    end = _end_line(node)
    lines = source.splitlines()
    body = "\n".join(lines[start - 1 : end])
    header = f"[{path_str} lines {start}-{end}]"
    return header + "\n" + body


def _regex_find_symbol(source: str, name: str, path_str: str) -> str:
    """Best-effort for non-Python files."""
    lines = source.splitlines()
    for i, line in enumerate(lines):
        if re.search(rf"\b(?:function|class|fn|func|def|interface|struct|trait|type|enum)\s+{re.escape(name)}\b", line):
            # Grab up to 40 following lines as a heuristic body
            end = min(len(lines), i + 40)
            return f"[{path_str} lines {i+1}-{end}] (regex match)\n" + "\n".join(lines[i:end])
    return f"[read_symbol] {name!r} not found in {path_str}"


# ---- helpers -----------------------------------------------------------------

def _resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    workspace = Path(os.environ.get("AGENT_WORKSPACE", "."))
    return (workspace / path).resolve()


def _end_line(node: ast.AST) -> int:
    end = getattr(node, "end_lineno", None)
    if end is not None:
        return end
    # Python <3.8 fallback: walk children
    max_line = getattr(node, "lineno", 0)
    for child in ast.walk(node):
        line = getattr(child, "lineno", 0)
        if line > max_line:
            max_line = line
    return max_line


def _format_args(args: ast.arguments) -> str:
    parts = []
    defaults = list(args.defaults)
    n_defaults = len(defaults)
    positional = args.args
    n_positional = len(positional)
    for i, arg in enumerate(positional):
        offset = i - (n_positional - n_defaults)
        ann = f": {_format_node(arg.annotation)}" if arg.annotation else ""
        if offset >= 0:
            parts.append(f"{arg.arg}{ann} = ...")
        else:
            parts.append(f"{arg.arg}{ann}")
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    return ", ".join(parts)


def _format_node(node: Optional[ast.AST]) -> str:
    if node is None:
        return "None"
    try:
        return ast.unparse(node)
    except Exception:
        return "<?>"


def _format_decorator(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "<?>"


# ---- register ----------------------------------------------------------------

register(Tool(
    name="list_symbols",
    description=(
        "Return a structural outline of a source file — every top-level class, function, "
        "and (for classes) their methods with line ranges. Bodies are elided. Use this "
        "FIRST when you need to navigate an unfamiliar file — it's 50x cheaper on context "
        "than read_file for the same information. Supports Python natively + regex "
        "fallback for JS/TS/Go/Rust/Java/C/C++."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path (relative to workspace or absolute)"},
        },
        "required": ["path"],
    },
    category="read",
    fn=_list_symbols,
))


register(Tool(
    name="read_symbol",
    description=(
        "Return the source of a single class or function by name — just that symbol, "
        "not the whole file. Use `ClassName.method_name` to target a method inside a "
        "class. After `list_symbols` tells you what's in a file, use this to zoom in "
        "on the relevant symbol without pulling unrelated code into context."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
            "name": {"type": "string", "description": "Symbol name (e.g. 'MyClass' or 'MyClass.method')"},
        },
        "required": ["path", "name"],
    },
    category="read",
    fn=_read_symbol,
))
