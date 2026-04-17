"""
Reflector node — Reflexion pattern.

Runs after a batch of executor work. Reviews outcomes and:
  1. Writes 0-3 "lessons learned" that persist to memory.
  2. Decides: done? keep going? stuck (→ replan)?
  3. If everything succeeded and produced a clean artifact, can promote
     the action sequence to a Skill in the library.

Hardened with automatic docstring-example verification: if the executor wrote
Python files with docstring examples (>>>  or `f(x) == expected` patterns) in
this batch, we run those examples. A "done" verdict with failing examples gets
auto-overridden to "continue" with a structured failure report attached.
"""
from __future__ import annotations
import datetime
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from ..config import CONFIG
from ..state import AgentState, HistoryEvent, Lesson
from ..memory import Memory, SkillLibrary, Skill, SkillStep
from ..safety.doctest_check import find_examples, run_examples, VerificationReport

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are the Reflector — a STRICT quality gate on an autonomous coding agent.

You will be given the original goal, the subgoal tree with statuses and results, and recent tool calls.

Produce a JSON object:
{
  "verdict": "continue" | "done" | "stuck",
  "reason":  "one-sentence explanation",
  "completeness": {
    "target_count": int | null,
    "actual_count": int | null,
    "target_basis": "short quote from goal that expresses the count"
  },
  "lessons": [
    {"text": "concise GENERIC lesson for future sessions", "severity": "info|warn|error", "tags": ["tag1"]}
  ],
  "save_as_skill": null | {"name": "snake_case", "description": "...", "trigger_text": "..."}
}

COMPLETENESS CHECK — read the goal carefully and look for quantitative or bounded-list language:
  - "all N of ...", "every X", "each of ...": target_count = N (or count the items in the list)
  - "top 5", "three tests", "five bugs": target_count = that number
  - Explicit bulleted sub-requirements like "(a) do X, (b) do Y, (c) do Z": target_count = the number of bullets
  - No quantitative language: target_count = null (completeness irrelevant)

Then count what actually landed (from tool observations / subgoal results). Put that in actual_count.
If target_count > actual_count: verdict MUST be "continue" (not "done"). Reason should say what's missing.

QUALITY CHECKS — use "continue" (not "done") if ANY fail:
  1. Did the goal ACTUALLY succeed, or did the executor merely CLAIM success? Check tool outputs.
  2. If a FILE was written: does the content look complete? Stub `pass` bodies, TODO placeholders, or "I've done X" prose inside a file are NOT done.
  3. If a REPORT was written: does it cite concrete evidence (line numbers, URLs, tool observations)? Reports that say "sources: [Some Magazine 2023]" without a URL are hallucinated — NOT done.
  4. If TESTS were written: do they have real assertions or just `pass`/`...`? Stubs are NOT done.
  5. If a BUG was claimed fixed: was the fix actually written to the file, and was verification attempted (re-read or run test)? Otherwise NOT done.
  6. Does the executor's final text contain `<tool_call>` or `<function=` tags? That's a hallucinated tool-call as text — NOT done; treat as needing another execution pass.
  7. Did the ruff or regression gate fire (message contains "ROLLED BACK" or "STATIC-ANALYSIS FAILURE" or "TEST REGRESSION DETECTED")? That's an automatic "continue" — the edit was reverted.
  8. Did docstring verification fail (message contains "VERIFICATION FAILURE")? Automatic "continue" — code contradicts its own contract.

Verdict rules:
  - "done" only if the goal is genuinely satisfied per checks 1-8 above AND completeness target is met (if specified).
  - "stuck" if the SAME failure recurs OR two consecutive subgoals fail for related reasons. Triggers a replan.
  - "continue" in all other cases.

Lessons should be GENERIC and REUSABLE. "Always cite URLs in research briefs" > "the 2026 embroidery brief was missing URLs".
save_as_skill only when verdict is "done" AND the sequence is reusable for future similar goals.
Return ONLY the JSON. No prose, no fences."""


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _extract_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in:\n{text[:400]}")
    return json.loads(text[start : end + 1])


def _format_subgoals(subgoals) -> str:
    parts = []
    for s in subgoals:
        result_excerpt = (s.result or "")[:400]
        parts.append(f"[{s.id}] status={s.status} attempts={s.attempts} role={s.role}\n  desc: {s.description}\n  result: {result_excerpt}")
    return "\n\n".join(parts)


# ---- docstring-verification gate --------------------------------------------

_WRITE_TOOLS = {"write_file", "edit_file", "apply_patch", "pskit_write_file", "pskit_edit_file"}


def _recent_py_writes(state: AgentState) -> list[Path]:
    """Collect .py files written or edited in this executor batch.

    We inspect the most recent tool calls (not the entire history — the gate
    targets THIS batch's output, not earlier artefacts).
    """
    workspace = Path(os.environ.get("AGENT_WORKSPACE", "."))
    paths: list[Path] = []
    seen: set[Path] = set()

    for tc in state.get("last_tool_calls", []) or []:
        name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else None)
        args = getattr(tc, "args", None) or (tc.get("args", {}) if isinstance(tc, dict) else {})
        if name not in _WRITE_TOOLS:
            continue
        if not isinstance(args, dict):
            continue
        path_arg = args.get("path") or args.get("file_path") or args.get("target")
        if not path_arg:
            continue
        p = Path(path_arg)
        if not p.is_absolute():
            p = workspace / p
        if p.suffix != ".py":
            continue
        if p in seen:
            continue
        seen.add(p)
        if p.exists():
            paths.append(p)

    return paths


def _verify_written_code(state: AgentState) -> tuple[list[dict], list[HistoryEvent]]:
    """Run docstring verification on every .py file written in this batch.

    Returns (failures, events) where:
      - failures is a list of dicts describing files with failing examples
        (empty if all files pass or have no examples)
      - events are HistoryEvents to append to the state history
    """
    failures: list[dict] = []
    events: list[HistoryEvent] = []

    for path in _recent_py_writes(state):
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            logger.debug("doctest gate: could not read %s: %s", path, e)
            continue

        try:
            examples = find_examples(source)
        except Exception as e:
            logger.debug("doctest gate: parse failed for %s: %s", path, e)
            continue

        if not examples:
            continue

        try:
            report: VerificationReport = run_examples(path, examples, timeout_sec=10.0)
        except Exception as e:
            logger.warning("doctest gate: run_examples crashed for %s: %s", path, e)
            continue

        if report.ok:
            events.append(HistoryEvent(
                kind="verification",
                content=f"[verify] {path.name}: {report.passed}/{report.total} examples passed",
                data={"path": str(path), "total": report.total, "passed": report.passed},
                timestamp=_now(),
            ))
            continue

        # Verification failed — build structured failure payload
        failed_examples = [
            {
                "source": r.example.source,
                "expected": r.example.expected,
                "actual": r.actual,
                "error": r.error,
                "kind": r.example.kind,
            }
            for r in report.results if not r.passed
        ]
        failures.append({
            "path": str(path),
            "entry_point": report.entry_point,
            "total": report.total,
            "passed": report.passed,
            "failed_examples": failed_examples,
            "broken_code": source[:4000],
            "report_summary": report.summary(),
        })
        events.append(HistoryEvent(
            kind="verification",
            content=f"VERIFICATION FAILURE in {path.name}: {len(failed_examples)}/{report.total} examples failed",
            data={
                "path": str(path),
                "failed_count": len(failed_examples),
                "total": report.total,
                "summary": report.summary(max_failures=3),
            },
            timestamp=_now(),
        ))

    return failures, events


def reflector_node(state: AgentState) -> dict:
    fleet = CONFIG["fleet"]
    client = OpenAI(base_url=fleet.reflector.base_url, api_key="EMPTY")
    subgoals = state.get("subgoals") or []

    # --- pre-verdict: run docstring verification on this batch's writes ---
    verification_failures, verification_events = _verify_written_code(state)

    user = (
        f"ORIGINAL GOAL:\n{state['goal']}\n\n"
        f"SUBGOAL TREE:\n{_format_subgoals(subgoals)}\n\n"
        f"RECENT TOOL CALL COUNT: {len(state.get('last_tool_calls', []))}\n"
        f"ITERATION: {state.get('iterations', 0)}\n"
        f"CONSECUTIVE FAILURES: {state.get('consecutive_failures', 0)}"
    )
    if verification_failures:
        ver_block_lines = ["\nDOCSTRING VERIFICATION — FAILURES DETECTED:"]
        for vf in verification_failures:
            ver_block_lines.append(f"  - {vf['path']}: {len(vf['failed_examples'])}/{vf['total']} examples failed")
            ver_block_lines.append("    " + vf["report_summary"].replace("\n", "\n    "))
        user = user + "\n" + "\n".join(ver_block_lines)

    r = client.chat.completions.create(
        model=fleet.reflector.name,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user},
        ],
        max_tokens=fleet.reflector.max_tokens,
        temperature=fleet.reflector.temperature,
    )
    raw = r.choices[0].message.content or ""
    try:
        parsed = _extract_json(raw)
    except Exception as e:
        return {
            "status": "executing",
            "history": verification_events + [
                HistoryEvent(kind="reflection", content=f"parse failed: {e}", data={"raw": raw[:400]}, timestamp=_now())
            ],
        }

    verdict = parsed.get("verdict", "continue")
    reason = parsed.get("reason", "")

    # Enforce completeness override: if target_count > actual_count, block "done"
    completeness = parsed.get("completeness") or {}
    try:
        tc = completeness.get("target_count")
        ac = completeness.get("actual_count")
        if (tc is not None and ac is not None
                and isinstance(tc, (int, float)) and isinstance(ac, (int, float))
                and ac < tc and verdict == "done"):
            verdict = "continue"
            reason = (f"completeness override: goal requires {tc} items ({completeness.get('target_basis', '')}); "
                      f"only {ac} delivered. Need to finish the remaining {tc - ac}.")
    except Exception:
        pass

    # ENFORCE verification override: if any docstring-example verification failed, block "done"
    if verification_failures and verdict == "done":
        verdict = "continue"
        first = verification_failures[0]
        reason = (
            f"docstring verification override: {first['path']} has "
            f"{len(first['failed_examples'])}/{first['total']} failing examples. "
            f"The code contradicts its own docstring contract. Fix the implementation."
        )
    lesson_objs = [
        Lesson(text=l.get("text", ""), severity=l.get("severity", "info"), tags=list(l.get("tags") or []))
        for l in parsed.get("lessons", []) if l.get("text")
    ]

    mem = Memory()
    session_id = state.get("session_id", "unknown")
    for l in lesson_objs:
        try:
            mem.add_lesson(session_id, l)
        except Exception:
            pass

    # Maybe promote a skill
    sk = parsed.get("save_as_skill")
    skill_event = None
    if sk and verdict == "done":
        try:
            steps: list[SkillStep] = []
            for tc in state.get("last_tool_calls", []):
                if tc.error is None:
                    steps.append(SkillStep(tool=tc.name, args=tc.args, why=""))
            if steps:
                skill = Skill(
                    name=sk.get("name", "unnamed_skill"),
                    description=sk.get("description", ""),
                    trigger_text=sk.get("trigger_text", state["goal"]),
                    steps=steps,
                )
                SkillLibrary().save(skill)
                skill_event = HistoryEvent(kind="lesson", content=f"saved skill: {skill.name}", data={"skill_name": skill.name, "steps": len(steps)}, timestamp=_now())
        except Exception as e:
            skill_event = HistoryEvent(kind="lesson", content=f"skill save failed: {e}", timestamp=_now())

    new_failures = state.get("consecutive_failures", 0)
    if verdict == "stuck":
        new_failures += 1
    elif verdict == "done":
        new_failures = 0

    events = list(verification_events)
    events.append(HistoryEvent(
        kind="reflection",
        content=f"[{verdict}] {reason}",
        data={"verdict": verdict, "lessons_added": len(lesson_objs), "verification_failures": len(verification_failures)},
        timestamp=_now(),
    ))
    for l in lesson_objs:
        events.append(HistoryEvent(kind="lesson", content=l.text, data={"severity": l.severity, "tags": l.tags}, timestamp=_now()))
    if skill_event:
        events.append(skill_event)

    next_status: str
    if verdict == "done":
        next_status = "done"
    elif verdict == "stuck":
        next_status = "replanning"
    else:
        from .executor import _pick_next_subgoal
        next_status = "executing" if _pick_next_subgoal(subgoals) is not None else "done"

    out: dict[str, Any] = {
        "status": next_status,
        "stuck": verdict == "stuck",
        "consecutive_failures": new_failures,
        "lessons": lesson_objs,
        "history": events,
    }
    # Expose verification failures on the state so retrospective can mine them
    if verification_failures:
        out["verification_failures"] = verification_failures

    if next_status == "done":
        out["final_answer"] = _synthesize_final(state, subgoals, reason)
    return out


def _synthesize_final(state: AgentState, subgoals, reason: str) -> str:
    lines = [f"Goal: {state['goal']}", f"Verdict: done ({reason})", "", "Subgoal outcomes:"]
    for s in subgoals:
        lines.append(f"  [{s.id}] {s.status}: {s.description}")
        if s.result:
            lines.append(f"      → {s.result[:300]}")
    return "\n".join(lines)
