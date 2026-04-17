"""
Reflector node — Reflexion pattern.

Runs after a batch of executor work. Reviews outcomes and:
  1. Writes 0-3 "lessons learned" that persist to memory.
  2. Decides: done? keep going? stuck (→ replan)?
  3. If everything succeeded and produced a clean artifact, can promote
     the action sequence to a Skill in the library.
"""
from __future__ import annotations
import datetime
import json
import re
from typing import Any

from openai import OpenAI

from ..config import CONFIG
from ..state import AgentState, HistoryEvent, Lesson
from ..memory import Memory, SkillLibrary, Skill, SkillStep


_SYSTEM_PROMPT = """You are the Reflector — a STRICT quality gate on an autonomous coding agent.

You will be given the original goal, the subgoal tree with statuses and results, and recent tool calls.

Produce a JSON object:
{
  "verdict": "continue" | "done" | "stuck",
  "reason":  "one-sentence explanation",
  "completeness": {
    "target_count": int | null,        // if goal specifies a count, put it here; else null
    "actual_count": int | null,        // how many of the target were actually completed
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

Verdict rules:
  - "done" only if the goal is genuinely satisfied per checks 1-7 above AND completeness target is met (if specified).
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


def reflector_node(state: AgentState) -> dict:
    fleet = CONFIG["fleet"]
    client = OpenAI(base_url=fleet.reflector.base_url, api_key="EMPTY")
    subgoals = state.get("subgoals") or []

    user = (
        f"ORIGINAL GOAL:\n{state['goal']}\n\n"
        f"SUBGOAL TREE:\n{_format_subgoals(subgoals)}\n\n"
        f"RECENT TOOL CALL COUNT: {len(state.get('last_tool_calls', []))}\n"
        f"ITERATION: {state.get('iterations', 0)}\n"
        f"CONSECUTIVE FAILURES: {state.get('consecutive_failures', 0)}"
    )

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
        # Conservative fallback: continue, no lessons
        return {
            "status": "executing",
            "history": [HistoryEvent(kind="reflection", content=f"parse failed: {e}", data={"raw": raw[:400]}, timestamp=_now())],
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
    lesson_objs = [
        Lesson(text=l.get("text", ""), severity=l.get("severity", "info"), tags=list(l.get("tags") or []))
        for l in parsed.get("lessons", []) if l.get("text")
    ]

    # Persist lessons
    mem = Memory()
    session_id = state.get("session_id", "unknown")
    for l in lesson_objs:
        try:
            mem.add_lesson(session_id, l)
        except Exception:
            pass  # don't derail the agent on memory errors

    # Maybe promote a skill
    sk = parsed.get("save_as_skill")
    skill_event = None
    if sk and verdict == "done":
        try:
            # Build Skill from successful subgoals + their tool call patterns
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

    events = [
        HistoryEvent(
            kind="reflection",
            content=f"[{verdict}] {reason}",
            data={"verdict": verdict, "lessons_added": len(lesson_objs)},
            timestamp=_now(),
        )
    ]
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
        # Still work pending — if subgoals remain, go back to executor; else also "done" by exhaustion
        from .executor import _pick_next_subgoal
        next_status = "executing" if _pick_next_subgoal(subgoals) is not None else "done"

    out: dict[str, Any] = {
        "status": next_status,
        "stuck": verdict == "stuck",
        "consecutive_failures": new_failures,
        "lessons": lesson_objs,
        "history": events,
    }
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
