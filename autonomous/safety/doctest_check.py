"""Docstring example verifier.

Parses a Python file, finds `>>>` and `func() == expected` examples in every
function's docstring, executes them in a subprocess with timeout + capture,
and returns a structured report.

Zero external deps — stdlib only. Safe to import from any layer of Anvil.

Public API:
    find_examples(source: str, entry_point: str | None) -> list[Example]
    run_examples(code_path: Path, examples: list[Example], ...) -> VerificationReport
    verify_file(code_path: Path, entry_point: str | None, ...) -> VerificationReport
"""
from __future__ import annotations

import ast
import json
import logging
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---- data model --------------------------------------------------------------

@dataclass
class Example:
    """A single executable assertion extracted from a docstring."""
    source: str          # the code to evaluate, e.g. "is_bored('Hi there')"
    expected: str        # string form of the expected value
    line: int            # approximate line in the source file
    kind: str            # '>>>' | 'eq' — parse origin
    entry_point: str     # function the example belongs to


@dataclass
class ExampleResult:
    example: Example
    passed: bool
    actual: str | None = None     # repr of actual value, or None on error
    error: str | None = None      # exception message, or None
    duration_ms: float = 0.0


@dataclass
class VerificationReport:
    entry_point: str | None
    total: int
    passed: int
    results: list[ExampleResult] = field(default_factory=list)
    subprocess_error: str | None = None
    timed_out: bool = False
    wall_ms: float = 0.0

    @property
    def ok(self) -> bool:
        """True iff we found examples and all of them passed."""
        return (
            self.total > 0
            and self.passed == self.total
            and not self.subprocess_error
            and not self.timed_out
        )

    @property
    def failed(self) -> list[ExampleResult]:
        return [r for r in self.results if not r.passed]

    def summary(self, max_failures: int = 3) -> str:
        if self.subprocess_error:
            return f"[verify] runner crashed: {self.subprocess_error}"
        if self.timed_out:
            return f"[verify] TIMEOUT — one or more examples exceeded the budget"
        if self.total == 0:
            return "[verify] No docstring examples found. Nothing to check."

        lines = [f"[verify] {self.passed}/{self.total} examples passed ({self.wall_ms:.0f}ms)"]
        failures = self.failed
        if failures:
            lines.append(f"  {len(failures)} failed — showing first {min(len(failures), max_failures)}:")
            for r in failures[:max_failures]:
                exp = r.example.expected[:80]
                act = (r.actual or "<no output>")[:80]
                err = f"  error: {r.error}" if r.error else ""
                lines.append(f"    >>> {r.example.source}")
                lines.append(f"        expected: {exp}")
                lines.append(f"        actual:   {act}{err}")
            if len(failures) > max_failures:
                lines.append(f"    ... and {len(failures) - max_failures} more")
        return "\n".join(lines)


# ---- parsing -----------------------------------------------------------------

def find_examples(source: str, entry_point: str | None = None) -> list[Example]:
    """Extract all executable examples from every function's docstring.

    Detects two patterns:
      - classic doctest: `>>> f(x)\\n<expected>`
      - assertion style: `f(x) == <expected>` inline in docstring text
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        logger.debug("find_examples: source is not parseable: %s", e)
        return []

    out: list[Example] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if entry_point and node.name != entry_point:
            continue
        doc = ast.get_docstring(node)
        if not doc:
            continue
        out.extend(_parse_doctest_style(doc, node.name, node.lineno))
        out.extend(_parse_assertion_style(doc, node.name, node.lineno))

    return _dedupe(out)


def _parse_doctest_style(docstring: str, entry_point: str, base_line: int) -> list[Example]:
    """Find classic doctest `>>> expr\\n<expected>` pairs."""
    import doctest
    parser = doctest.DocTestParser()
    try:
        dt = parser.get_doctest(docstring, {}, entry_point, "<docstring>", base_line)
    except Exception as e:
        logger.debug("doctest parser failed: %s", e)
        return []

    out = []
    for ex in dt.examples:
        source = ex.source.rstrip()
        expected = ex.want.rstrip()
        if not source:
            continue
        out.append(Example(
            source=source,
            expected=expected,
            line=base_line + (ex.lineno or 0),
            kind=">>>",
            entry_point=entry_point,
        ))
    return out


def _parse_assertion_style(docstring: str, entry_point: str, base_line: int) -> list[Example]:
    """Find `f(args) == expected` and `f(args) => expected` patterns anywhere
    in the docstring (not in code blocks prefixed by `>>>`)."""
    out = []
    # Don't double-parse stuff already caught by the doctest parser
    non_doctest_lines = []
    for line in docstring.splitlines():
        if line.lstrip().startswith(">>>") or line.lstrip().startswith("..."):
            continue
        non_doctest_lines.append(line)
    text = "\n".join(non_doctest_lines)

    name = re.escape(entry_point)
    # Match: `name(...) == value` or `name(...) => value` or `name(...) ➞ value`
    patterns = [
        re.compile(
            r"(" + name + r"\s*\([^\n]*?\))\s*(?:==|=>|➞|->)\s*(.+?)(?=\n|$|#)",
            re.MULTILINE,
        ),
    ]
    for pattern in patterns:
        for match in pattern.finditer(text):
            call = match.group(1).strip()
            expected = match.group(2).strip().rstrip(",;")
            if not call or not expected:
                continue
            # Skip if the right-hand side looks like more prose (e.g. starts with a word)
            if re.match(r"^[A-Za-z_]\w*\s+\w+", expected):
                continue
            out.append(Example(
                source=call,
                expected=expected,
                line=base_line,
                kind="eq",
                entry_point=entry_point,
            ))
    return out


def _dedupe(examples: list[Example]) -> list[Example]:
    """Merge examples that parse to the same (source, expected) pair."""
    seen: set[tuple[str, str]] = set()
    out = []
    for ex in examples:
        key = (ex.source, ex.expected)
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out


# ---- execution ---------------------------------------------------------------

_RUNNER_TEMPLATE = r"""
import importlib.util, json, math, sys, time, traceback

def _eq(actual, expected_repr):
    # Exact repr match (cheap fast-path).
    if repr(actual) == expected_repr:
        return True
    # Try literal-eval comparison (handles whitespace, quotes differences).
    try:
        import ast
        parsed = ast.literal_eval(expected_repr)
        if _deep_eq(actual, parsed):
            return True
    except (ValueError, SyntaxError):
        pass
    # Try numeric comparison with tolerance for floats.
    try:
        parsed = ast.literal_eval(expected_repr)
        if isinstance(actual, float) and isinstance(parsed, (int, float)):
            return math.isclose(actual, parsed, rel_tol=1e-6, abs_tol=1e-9)
    except (ValueError, SyntaxError):
        pass
    # String equality as last resort (useful for formatted outputs).
    if isinstance(actual, str) and actual == expected_repr:
        return True
    return False

def _deep_eq(a, b):
    if isinstance(a, float) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return False
        return all(_deep_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_deep_eq(a[k], b[k]) for k in a)
    return a == b

# Load the target module in isolation
spec = importlib.util.spec_from_file_location("__verify_target__", __TARGET_PATH__)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception as e:
    print(json.dumps({"fatal": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:]}))
    sys.exit(0)

# Build eval namespace: everything from the target, plus math/etc for eval'd example sources
ns = dict(mod.__dict__)

examples = __EXAMPLES_JSON__
results = []
for ex in examples:
    t0 = time.perf_counter()
    entry = {"source": ex["source"], "expected": ex["expected"], "kind": ex["kind"]}
    try:
        actual = eval(ex["source"], ns)
        entry["actual"] = repr(actual)
        entry["passed"] = _eq(actual, ex["expected"])
        entry["error"] = None
    except Exception as e:
        entry["actual"] = None
        entry["passed"] = False
        entry["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    entry["duration_ms"] = (time.perf_counter() - t0) * 1000
    results.append(entry)

print(json.dumps({"results": results}))
"""


def run_examples(
    code_path: Path,
    examples: list[Example],
    timeout_sec: float = 10.0,
    python_exe: str = sys.executable,
) -> VerificationReport:
    """Run examples in a subprocess, return structured report.

    Subprocess isolation prevents a misbehaving implementation from polluting
    the caller's interpreter state or blowing the stack. Subprocess timeout
    caps the total runtime — useful when a buggy solution has an infinite loop.
    """
    import time

    if not examples:
        return VerificationReport(entry_point=None, total=0, passed=0)

    # Build runner script
    examples_json = json.dumps([
        {"source": ex.source, "expected": ex.expected, "kind": ex.kind}
        for ex in examples
    ])
    runner = (
        _RUNNER_TEMPLATE
        .replace("__TARGET_PATH__", repr(str(code_path.resolve())))
        .replace("__EXAMPLES_JSON__", examples_json)
    )

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [python_exe, "-I", "-c", runner],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return VerificationReport(
            entry_point=examples[0].entry_point if examples else None,
            total=len(examples),
            passed=0,
            timed_out=True,
            wall_ms=(time.perf_counter() - t0) * 1000,
        )

    wall_ms = (time.perf_counter() - t0) * 1000

    stdout = (proc.stdout or "").strip()
    if not stdout:
        return VerificationReport(
            entry_point=examples[0].entry_point if examples else None,
            total=len(examples),
            passed=0,
            subprocess_error=f"no stdout from runner (exit={proc.returncode}); stderr={proc.stderr[-500:]}",
            wall_ms=wall_ms,
        )

    try:
        payload = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError as e:
        return VerificationReport(
            entry_point=examples[0].entry_point if examples else None,
            total=len(examples),
            passed=0,
            subprocess_error=f"runner emitted invalid JSON: {e}; tail={stdout[-300:]}",
            wall_ms=wall_ms,
        )

    if "fatal" in payload:
        return VerificationReport(
            entry_point=examples[0].entry_point if examples else None,
            total=len(examples),
            passed=0,
            subprocess_error=f"module load failed: {payload['fatal']}",
            wall_ms=wall_ms,
        )

    results = []
    for row, ex in zip(payload["results"], examples):
        results.append(ExampleResult(
            example=ex,
            passed=bool(row["passed"]),
            actual=row.get("actual"),
            error=row.get("error"),
            duration_ms=float(row.get("duration_ms", 0.0)),
        ))

    return VerificationReport(
        entry_point=examples[0].entry_point if examples else None,
        total=len(results),
        passed=sum(1 for r in results if r.passed),
        results=results,
        wall_ms=wall_ms,
    )


def verify_file(
    code_path: Path,
    entry_point: str | None = None,
    timeout_sec: float = 10.0,
) -> VerificationReport:
    """One-shot helper: find + run + return report."""
    code_path = Path(code_path)
    source = code_path.read_text(encoding="utf-8", errors="replace")
    examples = find_examples(source, entry_point=entry_point)
    return run_examples(code_path, examples, timeout_sec=timeout_sec)
