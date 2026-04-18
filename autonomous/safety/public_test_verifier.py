"""Public-test verification for competitive-programming tasks.

HumanEval's docstring examples are the pre-commit check that gave us +5pts on
that benchmark. The same play exists for competitive-programming benchmarks
(LiveCodeBench, parts of BigCodeBench) where each task ships `public_test_cases`
right in the prompt — but our agent never ran them against its own solution
before declaring done.

This module fills that gap with a tool the agent can call explicitly
(`run_public_tests`) AND a helper the reflector can call automatically when
it detects competitive-programming tasks.

Two test shapes are handled:

  stdin       - pipe `input` to the solution's stdin; expect `output` on stdout
  functional  - call Solution().<method>(args); expect `output` return value

Isolation: every test runs in a fresh `python -c` subprocess with a timeout
(trusted code we generated, so no extra sandboxing needed).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TestResult:
    idx: int
    passed: bool
    input_preview: str
    expected_preview: str
    actual_preview: str = ""
    error: str = ""
    wall_ms: float = 0.0


@dataclass
class PublicTestReport:
    total: int
    passed: int
    results: list[TestResult] = field(default_factory=list)
    testtype: str = "stdin"
    wall_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.total > 0 and self.passed == self.total

    def summary(self, max_failures: int = 3) -> str:
        if self.total == 0:
            return "[public-tests] No test cases provided. Nothing to verify."
        lines = [f"[public-tests] {self.passed}/{self.total} passed ({self.wall_ms:.0f}ms)"]
        failed = [r for r in self.results if not r.passed]
        if failed:
            lines.append(f"  {len(failed)} failed — showing first {min(len(failed), max_failures)}:")
            for r in failed[:max_failures]:
                inp = r.input_preview[:80]
                exp = r.expected_preview[:80]
                act = r.actual_preview[:80]
                lines.append(f"    test {r.idx}:")
                lines.append(f"      input:    {inp!r}")
                lines.append(f"      expected: {exp!r}")
                lines.append(f"      actual:   {act!r}" if not r.error else f"      error:    {r.error[:120]}")
            if len(failed) > max_failures:
                lines.append(f"    ... and {len(failed) - max_failures} more")
        return "\n".join(lines)


# ---- test runners ------------------------------------------------------------

_FUNCTIONAL_RUNNER = r"""
import sys, json
__SOLUTION_CODE__

_inputs = json.loads('''__INPUTS_JSON__''')
try:
    sol = Solution()
except Exception as e:
    print(json.dumps({'error': f'{type(e).__name__}: {e}'}), file=sys.stderr)
    sys.exit(1)

fn = getattr(sol, __METHOD_NAME__, None)
if fn is None:
    print(json.dumps({'error': f'method not found: {__METHOD_NAME__}'}), file=sys.stderr)
    sys.exit(1)

try:
    result = fn(*_inputs) if isinstance(_inputs, list) and __SPLAT__ else fn(_inputs)
    print(json.dumps({'result': result}, default=str))
except Exception as e:
    print(json.dumps({'error': f'{type(e).__name__}: {e}'}), file=sys.stderr)
    sys.exit(2)
"""


def _normalize(text: str) -> str:
    """Whitespace-tolerant comparison — matches competitive-programming grading conventions."""
    return "\n".join(line.rstrip() for line in (text or "").strip().split("\n"))


def _run_stdin_test(solution: str, stdin_input: str, timeout: float) -> tuple[bool, str, str]:
    """Return (ok, actual_stdout, error_reason)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", solution],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "", "timeout"
    except Exception as e:
        return False, "", f"exec error: {e}"

    if proc.returncode != 0:
        return False, proc.stdout or "", f"exit {proc.returncode}: {(proc.stderr or '')[-200:]}"
    return True, proc.stdout or "", ""


def _run_functional_test(
    solution: str, method_name: str, args_list: list, splat: bool, timeout: float
) -> tuple[bool, object, str]:
    runner = (
        _FUNCTIONAL_RUNNER
        .replace("__SOLUTION_CODE__", solution)
        .replace("__INPUTS_JSON__", json.dumps(args_list))
        .replace("__METHOD_NAME__", repr(method_name))
        .replace("__SPLAT__", "True" if splat else "False")
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, None, "timeout"
    except Exception as e:
        return False, None, f"exec error: {e}"

    if proc.returncode != 0:
        err = proc.stderr[-200:] if proc.stderr else ""
        return False, None, f"exit {proc.returncode}: {err}"
    try:
        return True, json.loads(proc.stdout.strip().splitlines()[-1])["result"], ""
    except (json.JSONDecodeError, IndexError, KeyError):
        return False, None, f"unparseable output: {proc.stdout[-200:]}"


def _extract_method_name(solution_code: str) -> str | None:
    m = re.search(r"class\s+Solution[^:]*:\s*\n(?:.|\n)*?def\s+(\w+)\s*\(self", solution_code)
    return m.group(1) if m else None


def _extract_arg_count(solution_code: str, method_name: str) -> int:
    pat = rf"def\s+{re.escape(method_name)}\s*\(([^)]*)\)"
    m = re.search(pat, solution_code)
    if not m:
        return 1
    params = [p.strip() for p in m.group(1).split(",") if p.strip()]
    params = [p for p in params if p.split(":")[0].strip() != "self"]
    return len(params)


# ---- public API --------------------------------------------------------------

def verify_public_tests(
    solution_code: str,
    tests: list[dict],
    per_test_timeout: float = 8.0,
) -> PublicTestReport:
    """Run a list of {input, output, testtype} cases against a solution.

    Returns a structured report.
    """
    report = PublicTestReport(total=len(tests), passed=0)
    if not tests:
        return report

    testtype = tests[0].get("testtype", "stdin")
    report.testtype = testtype
    method_name = None
    arg_count = 1
    splat = False

    if testtype == "functional":
        method_name = _extract_method_name(solution_code)
        if method_name is None:
            # Can't verify; return as if zero tests provided
            report.results = [
                TestResult(idx=0, passed=False,
                           input_preview=str(tests[0].get("input", ""))[:80],
                           expected_preview=str(tests[0].get("output", ""))[:80],
                           error="no Solution class / method found in code"),
            ]
            return report
        arg_count = _extract_arg_count(solution_code, method_name)

    start = time.perf_counter()
    for i, t in enumerate(tests):
        expected = t.get("output", "")
        raw_input = t.get("input", "")
        t0 = time.perf_counter()
        res = TestResult(
            idx=i, passed=False,
            input_preview=str(raw_input)[:80],
            expected_preview=str(expected)[:80],
        )

        if testtype == "stdin":
            ok, actual, err = _run_stdin_test(solution_code, raw_input, per_test_timeout)
            res.actual_preview = (actual or "")[:80]
            res.error = err
            if ok and _normalize(actual) == _normalize(expected):
                res.passed = True

        elif testtype == "functional":
            parsed = raw_input
            if isinstance(raw_input, str):
                try:
                    parsed = json.loads(raw_input)
                except json.JSONDecodeError:
                    parsed = raw_input
            # Decide splat vs single-arg based on method arity
            if arg_count == 1:
                inputs = [parsed]
                splat = False
            elif isinstance(parsed, list) and len(parsed) == arg_count:
                inputs = parsed
                splat = True
            else:
                inputs = [parsed]
                splat = False

            ok, actual, err = _run_functional_test(
                solution_code, method_name, inputs, splat, per_test_timeout
            )
            res.actual_preview = str(actual)[:80]
            res.error = err

            try:
                expected_parsed = json.loads(expected) if isinstance(expected, str) else expected
            except json.JSONDecodeError:
                expected_parsed = expected
            if ok and actual == expected_parsed:
                res.passed = True

        else:
            res.error = f"unknown testtype: {testtype}"

        res.wall_ms = (time.perf_counter() - t0) * 1000
        report.results.append(res)
        if res.passed:
            report.passed += 1

    report.wall_ms = (time.perf_counter() - start) * 1000
    return report
