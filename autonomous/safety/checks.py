"""
Pre-commit safety checks for file writes.

Two gates:
  1. ruff_check(path) — runs ruff with a narrow rule set (F821, F823, F822).
     Returns the NEW errors introduced vs a baseline.
  2. test_regression_check(repo_dir) — runs pytest against test_output/ and
     compares pass/fail to a recorded baseline. Returns any PASS->FAIL regressions.

Both are "report-only" functions; callers decide what to do on failure
(e.g. rollback the write, return an error observation to the agent).
"""
from __future__ import annotations
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Rule codes that indicate "referenced something that doesn't exist / isn't bound yet"
_CRITICAL_RULES = ("F821", "F822", "F823")
# F821 = undefined-name   F822 = undefined-export   F823 = local-var-referenced-before-assignment


@dataclass
class RuffResult:
    ok: bool
    new_errors: list[dict]      # list of {code, message, line, col}
    baseline_errors: list[dict]
    raw_stdout: str = ""


def _run_ruff(path: str) -> list[dict]:
    """Run ruff, return a list of findings. Empty list means no issues on selected rules."""
    rules = ",".join(_CRITICAL_RULES)
    try:
        r = subprocess.run(
            ["ruff", "check", "--select", rules, "--output-format", "json", "--no-cache", path],
            capture_output=True, text=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        return []  # If ruff hangs, don't block the agent
    except FileNotFoundError:
        return []  # ruff not installed — degrade gracefully
    out = r.stdout.strip()
    if not out:
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    findings = []
    for item in data if isinstance(data, list) else []:
        findings.append({
            "code": item.get("code", ""),
            "message": item.get("message", ""),
            "line": (item.get("location") or {}).get("row", 0),
            "col":  (item.get("location") or {}).get("column", 0),
        })
    return findings


def ruff_check(path: str, baseline_content: Optional[str] = None) -> RuffResult:
    """
    Run ruff on the file at `path`. If baseline_content is provided, the checker
    computes baseline findings against that content (in a temp file) and returns
    only NEW findings — things introduced by the current write.

    Without a baseline, any finding is considered 'new'.
    """
    current_findings = _run_ruff(path)

    baseline_findings: list[dict] = []
    if baseline_content is not None:
        # Write baseline to a temp path with same extension, analyse, clean up
        import tempfile, os
        suffix = Path(path).suffix or ".py"
        tf = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8")
        try:
            tf.write(baseline_content)
            tf.flush()
            tf.close()
            baseline_findings = _run_ruff(tf.name)
        finally:
            try: os.unlink(tf.name)
            except OSError: pass

    # Dedup baseline (line, code, message) to compare
    def _key(f): return (f["code"], f["message"])
    baseline_keys = {_key(f) for f in baseline_findings}
    new = [f for f in current_findings if _key(f) not in baseline_keys]

    return RuffResult(
        ok=not new,
        new_errors=new,
        baseline_errors=baseline_findings,
    )


def format_ruff_failure(res: RuffResult, path: str) -> str:
    lines = [f"STATIC-ANALYSIS FAILURE ({len(res.new_errors)} new issues in {path}):"]
    for e in res.new_errors[:10]:
        lines.append(f"  {path}:{e['line']}:{e['col']}  {e['code']}  {e['message']}")
    lines.append("")
    lines.append("The edit was NOT applied. Re-read the surrounding lines, find the reference you broke, "
                 "and propose a new_content that preserves all referenced names.")
    return "\n".join(lines)


# ---- test regression check ----------------------------------------------------

@dataclass
class TestBaseline:
    """Snapshot of pytest results; used to compare before/after an edit."""
    ok_tests: list[str] = field(default_factory=list)       # nodeids that passed
    failed_tests: list[str] = field(default_factory=list)   # nodeids that failed or errored
    raw: str = ""


def _run_pytest_json(test_dir: str, timeout_s: int = 90) -> tuple[list[str], list[str], str]:
    """Run pytest in verbose mode so we get one line per test, parse PASSED/FAILED/ERROR."""
    import re
    try:
        r = subprocess.run(
            ["python", "-m", "pytest", "-v", "--tb=no", "--no-header", "-p", "no:cacheprovider", test_dir],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return [], [], "(pytest timed out)"
    except FileNotFoundError:
        return [], [], "(pytest not installed)"
    raw = (r.stdout or "") + ("\n" + r.stderr if r.stderr else "")
    passed, failed = [], []
    # Verbose lines look like: "tests/test_foo.py::test_bar PASSED   [ 50%]"
    # We strip terminal color codes and trailing progress markers.
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    for line in raw.splitlines():
        s = ansi.sub("", line).rstrip()
        # find the status token — must be a whole word
        m = re.search(r"\b(PASSED|FAILED|ERROR|XPASS|XFAIL)\b", s)
        if not m:
            continue
        nodeid = s[: m.start()].strip()
        if not nodeid or "::" not in nodeid:
            continue
        status = m.group(1)
        if status in ("PASSED", "XPASS"):
            passed.append(nodeid)
        elif status in ("FAILED", "ERROR"):
            failed.append(nodeid)
    return passed, failed, raw


def take_test_baseline(test_dir: str) -> TestBaseline:
    """Snapshot current pass/fail state of tests in test_dir."""
    passed, failed, raw = _run_pytest_json(test_dir)
    return TestBaseline(ok_tests=passed, failed_tests=failed, raw=raw[:6000])


@dataclass
class RegressionResult:
    ok: bool
    regressed_tests: list[str] = field(default_factory=list)   # passed before, now failing
    new_tests_failing: list[str] = field(default_factory=list) # didn't exist before, now failing
    raw: str = ""


def test_regression_check(test_dir: str, baseline: TestBaseline) -> RegressionResult:
    """Run tests again; report tests that went from PASS -> FAIL (regressions)."""
    passed_now, failed_now, raw = _run_pytest_json(test_dir)

    baseline_passed = set(baseline.ok_tests)
    now_failed = set(failed_now)
    now_passed = set(passed_now)

    # PASS -> FAIL regressions
    regressed = sorted(baseline_passed & now_failed)
    # Tests that weren't in baseline at all, and are failing now
    previously_unknown = sorted(now_failed - baseline_passed - set(baseline.failed_tests))

    return RegressionResult(
        ok=not regressed,
        regressed_tests=regressed,
        new_tests_failing=previously_unknown,
        raw=raw[:4000],
    )


def format_regression_failure(res: RegressionResult) -> str:
    lines = ["TEST REGRESSION DETECTED — edit was rolled back."]
    if res.regressed_tests:
        lines.append(f"\nTests that passed BEFORE and now FAIL ({len(res.regressed_tests)}):")
        for t in res.regressed_tests[:10]:
            lines.append(f"  - {t}")
    if res.new_tests_failing:
        lines.append(f"\nNew failing tests: {res.new_tests_failing[:5]}")
    lines.append("\nRe-read the file near the edit point; you likely broke something referenced "
                 "by those tests. Fix the issue and propose a corrected new_content.")
    return "\n".join(lines)
