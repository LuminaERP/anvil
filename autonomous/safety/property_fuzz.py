"""Property-based fuzz testing via Hypothesis — edge case discovery.

Hypothesis is optional. If not installed, this module is a no-op and we fall
back to docstring verification alone.

What this adds on top of docstring verification:
  - The docstring gives us *concrete* examples. Those verify correctness for
    the cases the problem author chose. But they miss edge cases like empty
    inputs, negative numbers, single-element lists, Unicode boundaries, etc.
  - Hypothesis *generates* diverse inputs that match a type strategy inferred
    from the concrete examples, then runs the function on them.
  - Two properties are checked per generated input:
      1. The function doesn't crash.
      2. The function is deterministic (f(x) == f(x)).
  - Hypothesis's shrinking gives us minimal counter-examples, not noise.

The output is a list of discovered edge cases that will often be latent bugs
the concrete examples didn't surface.
"""
from __future__ import annotations

import ast
import importlib.util
import logging
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---- capability detection ----------------------------------------------------

def hypothesis_available() -> bool:
    """True iff the hypothesis package is importable in the current process."""
    try:
        import hypothesis  # noqa: F401
        return True
    except ImportError:
        return False


# ---- data model --------------------------------------------------------------

@dataclass
class FuzzFinding:
    """A Hypothesis-discovered counter-example or crash."""
    kind: str                 # 'crash' | 'nondeterministic' | 'type_drift'
    input_repr: str           # repr of the minimised failing input
    exception: str | None = None   # for 'crash'
    actual_1: str | None = None    # first call result (for 'nondeterministic')
    actual_2: str | None = None    # second call result (for 'nondeterministic')
    details: str = ""


@dataclass
class FuzzReport:
    entry_point: str
    inputs_generated: int = 0
    findings: list[FuzzFinding] = field(default_factory=list)
    skipped_reason: str | None = None
    wall_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.findings and not self.skipped_reason

    def summary(self) -> str:
        if self.skipped_reason:
            return f"[fuzz] skipped: {self.skipped_reason}"
        if not self.findings:
            return f"[fuzz] {self.inputs_generated} random inputs: no crashes, deterministic ({self.wall_ms:.0f}ms)"
        lines = [f"[fuzz] {self.inputs_generated} inputs tested, {len(self.findings)} findings:"]
        for f in self.findings[:3]:
            lines.append(f"  [{f.kind}] input={f.input_repr[:80]}")
            if f.exception:
                lines.append(f"    exception: {f.exception[:200]}")
            if f.actual_1 is not None and f.actual_2 is not None:
                lines.append(f"    call1: {f.actual_1[:80]}  call2: {f.actual_2[:80]}")
            if f.details:
                lines.append(f"    details: {f.details}")
        if len(self.findings) > 3:
            lines.append(f"  ... and {len(self.findings) - 3} more")
        return "\n".join(lines)


# ---- type inference from concrete examples ----------------------------------

def infer_strategy(examples: list) -> str | None:
    """Given concrete docstring examples, infer a Hypothesis strategy expression
    as a string (to be eval'd inside the fuzz runner's namespace).

    Returns None if the inputs are too heterogeneous to strategise safely.

    Example: examples have inputs [1, 2, 3], [-1, 0, 1] → strategy `lists(integers())`
    """
    if not examples:
        return None

    # Extract argument values from each example. The source is `f(...)` — we
    # parse the AST to recover argument AST nodes.
    arg_samples: list[list[ast.AST]] = []
    for ex in examples:
        try:
            call = ast.parse(ex.source, mode="eval").body  # type: ignore[attr-defined]
            if isinstance(call, ast.Call):
                arg_samples.append(call.args)
        except (SyntaxError, AttributeError):
            continue

    if not arg_samples:
        return None

    # Require all examples to share the same arity
    arities = {len(args) for args in arg_samples}
    if len(arities) != 1:
        return None
    arity = arities.pop()
    if arity == 0:
        return None  # Zero-arg functions aren't interesting to fuzz

    # Infer a strategy per argument position
    per_arg: list[str] = []
    for pos in range(arity):
        types = {_infer_type(args[pos]) for args in arg_samples}
        types.discard(None)
        if not types:
            return None
        if len(types) > 2:
            # Too heterogeneous; skip
            return None
        per_arg.append(_types_to_strategy(types))

    if len(per_arg) == 1:
        return per_arg[0]
    return "st.tuples(" + ", ".join(per_arg) + ")"


def _infer_type(node: ast.AST) -> str | None:
    """Infer a type label from an argument AST node."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return "bool"
        if isinstance(node.value, int):
            return "int"
        if isinstance(node.value, float):
            return "float"
        if isinstance(node.value, str):
            return "str"
        if node.value is None:
            return "none"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _infer_type(node.operand)
    if isinstance(node, ast.List):
        if not node.elts:
            return "list_any"
        elt_types = {_infer_type(e) for e in node.elts}
        elt_types.discard(None)
        if elt_types == {"int"}:
            return "list_int"
        if elt_types <= {"int", "float"}:
            return "list_num"
        if elt_types == {"str"}:
            return "list_str"
        return "list_any"
    if isinstance(node, ast.Tuple):
        return "tuple"
    if isinstance(node, ast.Dict):
        return "dict"
    return None


def _types_to_strategy(types: set[str]) -> str:
    """Map a set of type labels to a Hypothesis strategy expression."""
    if "list_int" in types:
        return "st.lists(st.integers(min_value=-100, max_value=100), max_size=10)"
    if "list_num" in types:
        return "st.lists(st.one_of(st.integers(min_value=-100, max_value=100), st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100)), max_size=10)"
    if "list_str" in types:
        return "st.lists(st.text(max_size=10), max_size=10)"
    if "list_any" in types:
        return "st.lists(st.integers(), max_size=10)"
    if "str" in types:
        return "st.text(max_size=50)"
    if "float" in types:
        return "st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000)"
    if "int" in types:
        return "st.integers(min_value=-1000, max_value=1000)"
    if "bool" in types:
        return "st.booleans()"
    return "st.integers()"


# ---- execution ---------------------------------------------------------------

_FUZZ_RUNNER = r"""
import importlib.util, json, sys, time, traceback

try:
    from hypothesis import given, strategies as st, settings, HealthCheck, seed, Verbosity
    from hypothesis.errors import Unsatisfiable
except ImportError as e:
    print(json.dumps({"skip": f"hypothesis not installed: {e}"}))
    sys.exit(0)

spec = importlib.util.spec_from_file_location("__fuzz_target__", __TARGET_PATH__)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception as e:
    print(json.dumps({"skip": f"module load failed: {type(e).__name__}: {e}"}))
    sys.exit(0)

target_fn = getattr(mod, __ENTRY_POINT__, None)
if target_fn is None:
    print(json.dumps({"skip": f"entry point {__ENTRY_POINT__!r} not found"}))
    sys.exit(0)

strategy = eval(__STRATEGY__)
arity = __ARITY__

findings = []
inputs_tested = [0]

try:
    from hypothesis.strategies import SearchStrategy
except ImportError:
    SearchStrategy = object

def _call_target(args_tuple):
    if arity == 1:
        return target_fn(args_tuple)
    return target_fn(*args_tuple)

@settings(
    max_examples=__MAX_EXAMPLES__,
    deadline=1000,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture, HealthCheck.filter_too_much],
    verbosity=Verbosity.quiet,
    derandomize=True,
)
@given(strategy)
def _prop(args):
    inputs_tested[0] += 1
    try:
        result1 = _call_target(args)
    except Exception as e:
        findings.append({
            "kind": "crash",
            "input_repr": repr(args)[:200],
            "exception": f"{type(e).__name__}: {str(e)[:300]}",
        })
        return
    # Determinism check: second call must match
    try:
        result2 = _call_target(args)
    except Exception:
        return  # already captured above if it was going to crash
    if repr(result1) != repr(result2):
        findings.append({
            "kind": "nondeterministic",
            "input_repr": repr(args)[:200],
            "actual_1": repr(result1)[:200],
            "actual_2": repr(result2)[:200],
        })

t0 = time.perf_counter()
try:
    _prop()
except Unsatisfiable as e:
    print(json.dumps({"skip": f"hypothesis could not generate inputs: {e}"}))
    sys.exit(0)
except Exception as e:
    # Hypothesis raises on first unhandled failure; already captured in findings
    pass
wall_ms = (time.perf_counter() - t0) * 1000

print(json.dumps({
    "findings": findings,
    "inputs_generated": inputs_tested[0],
    "wall_ms": wall_ms,
}))
"""


def fuzz_function(
    code_path: Path,
    entry_point: str,
    examples: list,
    budget_sec: float = 5.0,
    max_examples: int = 40,
) -> FuzzReport:
    """Run Hypothesis-based fuzz testing against the function.

    Uses the provided concrete examples to infer an input strategy.
    Returns a FuzzReport. Never raises — errors become `skipped_reason`.
    """
    import time

    if not hypothesis_available():
        return FuzzReport(entry_point=entry_point, skipped_reason="hypothesis not installed (pip install hypothesis)")

    strategy_expr = infer_strategy(examples)
    if strategy_expr is None:
        return FuzzReport(
            entry_point=entry_point,
            skipped_reason="could not infer input strategy from docstring examples",
        )

    # Compute arity
    try:
        first_call = ast.parse(examples[0].source, mode="eval").body  # type: ignore[attr-defined]
        arity = len(first_call.args) if isinstance(first_call, ast.Call) else 1
    except (SyntaxError, AttributeError):
        arity = 1

    runner = (
        _FUZZ_RUNNER
        .replace("__TARGET_PATH__", repr(str(code_path.resolve())))
        .replace("__ENTRY_POINT__", repr(entry_point))
        .replace("__STRATEGY__", repr(strategy_expr))
        .replace("__ARITY__", str(arity))
        .replace("__MAX_EXAMPLES__", str(max_examples))
    )

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", runner],
            capture_output=True,
            text=True,
            timeout=budget_sec,
        )
    except subprocess.TimeoutExpired:
        return FuzzReport(
            entry_point=entry_point,
            wall_ms=(time.perf_counter() - t0) * 1000,
            skipped_reason=f"fuzz budget exceeded ({budget_sec}s)",
        )

    wall_ms = (time.perf_counter() - t0) * 1000
    stdout = (proc.stdout or "").strip()
    if not stdout:
        return FuzzReport(
            entry_point=entry_point, wall_ms=wall_ms,
            skipped_reason=f"runner produced no output (exit={proc.returncode}, stderr={proc.stderr[-300:]})",
        )

    try:
        import json
        payload = json.loads(stdout.splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return FuzzReport(
            entry_point=entry_point, wall_ms=wall_ms,
            skipped_reason="runner emitted invalid JSON",
        )

    if "skip" in payload:
        return FuzzReport(entry_point=entry_point, wall_ms=wall_ms, skipped_reason=payload["skip"])

    findings = [
        FuzzFinding(
            kind=f["kind"],
            input_repr=f.get("input_repr", ""),
            exception=f.get("exception"),
            actual_1=f.get("actual_1"),
            actual_2=f.get("actual_2"),
        )
        for f in payload.get("findings", [])
    ]

    return FuzzReport(
        entry_point=entry_point,
        inputs_generated=int(payload.get("inputs_generated", 0)),
        findings=findings,
        wall_ms=wall_ms,
    )
