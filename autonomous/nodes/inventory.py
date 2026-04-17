"""
Workspace inventory — deterministic ground-truth extraction.

Runs BEFORE any LLM planning. Injects concrete facts into prompts so the
planner can't invent function names, file paths, or tests that don't exist.

Facts extracted:
  - Every Python function (module, name, line, signature)
  - Every Python class (module, name, line)
  - Every test file + the test function names it defines
  - Every bare `except:` location
  - Every function missing a docstring / type hints
  - Git status (modified / untracked files)
  - Recently changed files (git log)

The output is a SMALL, STRUCTURED context block that gets pasted verbatim
into the planner's user message. The LLM is explicitly instructed to only
propose goals referencing these names.
"""
from __future__ import annotations
import ast
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class FunctionFact:
    module: str       # file path, relative to workspace
    name: str
    lineno: int
    signature: str    # single-line "def foo(a: int) -> bool"
    has_docstring: bool
    has_return_hint: bool
    has_param_hints: bool


@dataclass
class ClassFact:
    module: str
    name: str
    lineno: int


@dataclass
class TestFact:
    module: str       # test file path
    name: str         # test function or class name
    lineno: int


@dataclass
class BareExceptFact:
    module: str
    lineno: int


@dataclass
class Inventory:
    workspace: str
    functions: list[FunctionFact] = field(default_factory=list)
    classes: list[ClassFact] = field(default_factory=list)
    tests: list[TestFact] = field(default_factory=list)
    bare_excepts: list[BareExceptFact] = field(default_factory=list)
    missing_docstrings: list[str] = field(default_factory=list)   # "module::func"
    missing_return_hints: list[str] = field(default_factory=list)
    git_status: str = ""
    recent_commits: str = ""
    source_files: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)

    def real_function_names(self) -> set[str]:
        return {f.name for f in self.functions}

    def real_class_names(self) -> set[str]:
        return {c.name for c in self.classes}


# ---- scanners -------------------------------------------------------------

_SKIP_DIRS = {"__pycache__", ".git", "node_modules", "venv", ".venv", "build", "dist", "models", "logs"}


def _is_under_skip(p: Path) -> bool:
    return any(part in _SKIP_DIRS for part in p.parts)


def _scan_py_file(path: Path, workspace: Path) -> tuple[list[FunctionFact], list[ClassFact], list[TestFact], list[BareExceptFact]]:
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [], [], [], []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return [], [], [], []

    rel = str(path.relative_to(workspace))
    is_test = "test_" in path.name or path.name.startswith("test_") or "/tests/" in rel or "/test/" in rel

    funcs: list[FunctionFact] = []
    classes: list[ClassFact] = []
    tests: list[TestFact] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            has_doc = bool(ast.get_docstring(node))
            has_ret = node.returns is not None
            has_params = all(a.annotation is not None for a in node.args.args if a.arg not in ("self", "cls")) if node.args.args else True
            try:
                sig = ast.unparse(node).splitlines()[0]
            except Exception:
                sig = f"def {node.name}(...)"
            funcs.append(FunctionFact(
                module=rel, name=node.name, lineno=node.lineno,
                signature=sig[:200], has_docstring=has_doc,
                has_return_hint=has_ret, has_param_hints=has_params,
            ))
            if is_test and node.name.startswith("test_"):
                tests.append(TestFact(module=rel, name=node.name, lineno=node.lineno))
        elif isinstance(node, ast.ClassDef):
            classes.append(ClassFact(module=rel, name=node.name, lineno=node.lineno))
            if is_test and node.name.startswith("Test"):
                tests.append(TestFact(module=rel, name=node.name, lineno=node.lineno))

    # Bare excepts
    bare_excepts: list[BareExceptFact] = []
    for i, line in enumerate(src.splitlines(), start=1):
        s = line.strip()
        if re.match(r"^except\s*:", s):
            bare_excepts.append(BareExceptFact(module=rel, lineno=i))

    return funcs, classes, tests, bare_excepts


def _git(cmd: list[str], cwd: str) -> str:
    try:
        r = subprocess.run(["git", *cmd], cwd=cwd, capture_output=True, text=True, timeout=5)
        return r.stdout
    except Exception:
        return ""


def build_inventory(workspace: str, max_files: int = 500) -> Inventory:
    """Scan the workspace for Python facts."""
    root = Path(workspace).resolve()
    inv = Inventory(workspace=str(root))

    if not root.exists() or not root.is_dir():
        return inv

    count = 0
    for p in sorted(root.rglob("*.py")):
        if _is_under_skip(p.relative_to(root)):
            continue
        if count >= max_files:
            break
        count += 1
        rel = str(p.relative_to(root))
        is_test = "test_" in p.name or p.name.startswith("test_")
        if is_test:
            inv.test_files.append(rel)
        else:
            inv.source_files.append(rel)

        fns, cls, tests, bx = _scan_py_file(p, root)
        inv.functions.extend(fns)
        inv.classes.extend(cls)
        inv.tests.extend(tests)
        inv.bare_excepts.extend(bx)

        for f in fns:
            key = f"{f.module}::{f.name}"
            if not f.has_docstring:
                inv.missing_docstrings.append(key)
            if not f.has_return_hint:
                inv.missing_return_hints.append(key)

    inv.git_status = _git(["status", "--short"], workspace)[:1500]
    inv.recent_commits = _git(["log", "-10", "--oneline"], workspace)[:1500]
    return inv


def format_inventory_for_prompt(inv: Inventory, max_funcs: int = 40) -> str:
    """Compact text block to inject into planner prompts."""
    lines = [
        f"=== WORKSPACE GROUND TRUTH ({inv.workspace}) ===",
        f"  source files: {len(inv.source_files)}  test files: {len(inv.test_files)}",
        f"  functions: {len(inv.functions)}  classes: {len(inv.classes)}",
        f"  tests: {len(inv.tests)}  bare-excepts: {len(inv.bare_excepts)}",
        "",
        "REAL FUNCTIONS THAT EXIST (module::name  @ line):",
    ]
    for f in inv.functions[:max_funcs]:
        marks = ""
        if not f.has_docstring: marks += " [no-docstring]"
        if not f.has_return_hint: marks += " [no-return-hint]"
        lines.append(f"  {f.module}::{f.name}  @ L{f.lineno}{marks}")
    if len(inv.functions) > max_funcs:
        lines.append(f"  ... and {len(inv.functions) - max_funcs} more")

    if inv.classes:
        lines.append("")
        lines.append("REAL CLASSES:")
        for c in inv.classes[:20]:
            lines.append(f"  {c.module}::{c.name}  @ L{c.lineno}")

    if inv.tests:
        lines.append("")
        lines.append("EXISTING TESTS (don't duplicate):")
        for t in inv.tests[:30]:
            lines.append(f"  {t.module}::{t.name}")

    if inv.bare_excepts:
        lines.append("")
        lines.append(f"BARE except: CLAUSES ({len(inv.bare_excepts)}):")
        for bx in inv.bare_excepts[:15]:
            lines.append(f"  {bx.module}:{bx.lineno}")

    if inv.missing_docstrings:
        lines.append("")
        lines.append(f"FUNCTIONS MISSING DOCSTRINGS ({len(inv.missing_docstrings)}):")
        for k in inv.missing_docstrings[:15]:
            lines.append(f"  {k}")

    if inv.git_status.strip():
        lines.append("")
        lines.append("GIT STATUS (short):")
        lines.append(inv.git_status.strip()[:600])

    lines.append("")
    lines.append("HARD RULE: any goal you propose must reference ONLY names that appear in the "
                 "lists above. Do NOT invent function names, file paths, or tests. If there's no "
                 "meaningful work left, say so — don't make something up.")
    return "\n".join(lines)


# ---- validation ------------------------------------------------------------

# Extract candidate "function name" tokens from a goal string.
# Match things that look like identifiers in backticks, or after `def `, or
# standalone snake_case words containing underscores (heuristic).
_NAME_PATTERNS = [
    re.compile(r"`([a-zA-Z_][a-zA-Z0-9_]*)`"),                 # `name`
    re.compile(r"\bdef\s+([a-zA-Z_][a-zA-Z0-9_]*)"),           # def name
    re.compile(r"\b([a-z_][a-z0-9_]*)\s*\("),                   # name(
]


def extract_function_references(text: str) -> set[str]:
    """Pull out every token that looks like a function name reference."""
    names: set[str] = set()
    for pat in _NAME_PATTERNS:
        for m in pat.finditer(text):
            n = m.group(1)
            # ignore built-ins + common English words
            if n in {"def", "class", "return", "if", "else", "for", "while", "try", "except",
                     "import", "from", "True", "False", "None", "self", "cls",
                     "print", "len", "int", "str", "dict", "list", "set", "tuple"}:
                continue
            if n.isupper() and len(n) < 4:
                continue
            names.add(n)
    return names


@dataclass
class ValidationResult:
    ok: bool
    unknown_names: list[str] = field(default_factory=list)
    message: str = ""


def validate_goal_against_inventory(goal: str, inv: Inventory) -> ValidationResult:
    """Check that every snake_case-looking function reference in the goal exists."""
    referenced = extract_function_references(goal)
    real = inv.real_function_names() | inv.real_class_names()

    # Heuristic: only flag tokens that look like python identifiers with underscores
    # AND don't exist. Pure-ASCII words without underscore could be any word.
    suspicious = {n for n in referenced
                  if "_" in n and n.lower() == n and len(n) > 2
                  and n not in real}

    if not suspicious:
        return ValidationResult(ok=True)
    return ValidationResult(
        ok=False,
        unknown_names=sorted(suspicious),
        message=(f"Goal references {len(suspicious)} name(s) that do NOT exist in the workspace: "
                 f"{sorted(suspicious)}. Real function names include: {sorted(list(real))[:15]}."),
    )
