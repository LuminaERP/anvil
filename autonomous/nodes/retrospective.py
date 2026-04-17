"""
Retrospective — post-session "could I have done this in 50% fewer steps?" pass.

Reads the full history of a completed session, generates:
  1. A condensed skill (if patterns emerged) for the skill library
  2. An "optimization lesson" stored in memory: "next time, do X directly instead of Y then Z"

This is the Hermes-Agent-style learning loop: after every successful task,
the agent retrospectively tries to compress what it did.
"""
from __future__ import annotations
import json
import re
from typing import Optional

from openai import OpenAI

from ..config import CONFIG
from ..memory import Memory, SkillLibrary, Skill, SkillStep
from ..state import Lesson


RETRO_PROMPT = """You are the Retrospective Critic of an autonomous agent.

You just completed a session. Given the trace below (plan, tool calls, reflections, final outcome), ask:

  1. Did this succeed? If not, what should be tried differently?
  2. Could this have been done in FEWER STEPS? Which tool calls were redundant, exploratory, or dead ends?
  3. Is there a REUSABLE PATTERN here that future similar goals would benefit from?

Produce a JSON object:
{
  "compressed_steps": [
    {"tool": "tool_name", "args": {...}, "why": "one-line purpose"}
  ],
  "optimization_lesson": "specific, actionable, generic — e.g. 'for code reviews, grep the imports first before reading files'",
  "save_skill": null | {
    "name": "snake_case",
    "description": "what it does",
    "trigger_text": "natural-language phrase that would match similar future goals"
  }
}

Rules:
  - compressed_steps is the MINIMAL sequence that would have achieved the same outcome. Omit the exploration, keep the essence.
  - optimization_lesson goes into memory and helps future planners. Make it GENERIC and REUSABLE, not task-specific.
  - Only propose save_skill if the pattern is clearly reusable AND the session succeeded.
  - Return ONLY the JSON. No prose, no fences."""


def _extract_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON")
    return json.loads(text[start : end + 1])


def _format_trace(events: list[dict]) -> str:
    lines = []
    for e in events[:200]:   # cap to avoid blowing context
        kind = e.get("kind", "?")
        sub = e.get("subgoal_id") or ""
        c = (e.get("content") or "")[:240]
        lines.append(f"[{kind}]{' sg='+sub if sub else ''} {c}")
    return "\n".join(lines)


def run_retrospective(session_id: str, goal: str, final_status: str) -> dict:
    """
    Produces {optimization_lesson, compressed_steps, save_skill}.
    Persists the lesson + optional skill into memory as a side effect.
    """
    mem = Memory()
    events = mem.events_for_session(session_id)
    trace = _format_trace(events)

    user_content = (
        f"SESSION ID: {session_id}\n"
        f"GOAL: {goal}\n"
        f"OUTCOME: {final_status}\n\n"
        f"TRACE:\n{trace}"
    )

    fleet = CONFIG["fleet"]
    client = OpenAI(base_url=fleet.reflector.base_url, api_key="EMPTY")
    r = client.chat.completions.create(
        model=fleet.reflector.name,
        messages=[
            {"role": "system", "content": RETRO_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=1500,
        temperature=0.1,
    )
    raw = r.choices[0].message.content or ""
    try:
        parsed = _extract_json(raw)
    except Exception as e:
        return {"error": str(e), "raw": raw[:400]}

    # Persist the optimization lesson
    opt = parsed.get("optimization_lesson", "").strip()
    if opt:
        try:
            mem.add_lesson(session_id, Lesson(
                text=opt,
                severity="info",
                tags=["retrospective", "optimization"],
            ))
        except Exception:
            pass

    # Maybe save the compressed skill
    sk = parsed.get("save_skill")
    steps_in = parsed.get("compressed_steps") or []
    if sk and steps_in and final_status == "done":
        try:
            skill = Skill(
                name=sk.get("name", "unnamed"),
                description=sk.get("description", ""),
                trigger_text=sk.get("trigger_text", goal),
                steps=[SkillStep(tool=s.get("tool", ""), args=s.get("args", {}), why=s.get("why", "")) for s in steps_in],
            )
            SkillLibrary().save(skill)
            parsed["skill_saved"] = skill.name
        except Exception as e:
            parsed["skill_save_error"] = str(e)

    return parsed
