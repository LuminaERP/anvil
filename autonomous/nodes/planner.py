"""
Planner node — STEP-style hierarchical subgoal decomposition.

Input:  state.goal, state.memory_context (prior lessons / similar sessions)
Output: state.subgoals, state.current_subgoal_idx=0, status="executing"

On replan (stuck==True), produces a revised subgoal tree that incorporates
lessons from the current session's failures.
"""
from __future__ import annotations
import json
import re
import datetime
from typing import Any

from openai import OpenAI

from ..config import CONFIG
from ..state import AgentState, Subgoal, HistoryEvent
from ..memory import Memory
from ..memory.skills import SkillLibrary
from .doc_context import build_doc_context
from .inventory import build_inventory, format_inventory_for_prompt


_SYSTEM_PROMPT = """You are the Planner of an autonomous coding agent.

Given a GOAL, produce a JSON array of SUBGOALS. Each subgoal is one of:
- {"id": "1", "role": "executor", "description": "...", "depends_on": []}
- {"id": "2", "role": "worker",   "description": "...", "depends_on": ["1"]}

GROUND TRUTH: A "WORKSPACE GROUND TRUTH" block below lists the REAL function, class, and test names that exist. Every name you write in a subgoal MUST appear in those lists or in the user's original goal. Do NOT invent function names. If the user goal references a function name, verify it against the ground-truth list before accepting it.

CRITICAL role assignment rules:
- "executor" has ACCESS TO TOOLS: read_file, write_file, list_dir, glob_files, run_bash, grep, web_search, web_fetch.
  Use "executor" WHENEVER the subgoal needs to: read files, list directories, search code, run commands, write code, fetch URLs, or verify any fact about the real system.
- "worker" has NO TOOLS. It only generates TEXT from what you put in the instruction.
  Use "worker" ONLY for pure text transforms: summarize/combine/rewrite content you've ALREADY gathered via executor subgoals.
- When in doubt, use "executor". A worker that needs tools will hallucinate and fail.

Planning rules:
1. Prefer FEW, POWERFUL subgoals (1-4). A single executor with good instructions often beats five small ones.
2. Each description is ONE concrete outcome, not a category. Use line numbers, file paths, exact commands.
3. Use "worker" ONLY when the inputs it needs are already produced by earlier "executor" subgoals AND no further tool use is required. Most plans have 0-1 worker subgoals.
4. Populate depends_on so independent work can parallelize.
5. For a goal like "analyze X, fix Y, and write docs", one executor that does all three is usually better than three separate ones.
6. Return ONLY the JSON array. No prose, no markdown fences.

Examples:

Goal: "Review foo.py for bugs and write a summary report to report.md"
CORRECT:
[{"id":"1","role":"executor","description":"Read foo.py, identify bugs with line-number citations, and write a prioritized report to /workspace/report.md. Verify the file was written.","depends_on":[]}]
WRONG (worker can't read files):
[{"id":"1","role":"worker","description":"List files","depends_on":[]},{"id":"2","role":"worker","description":"Read foo.py","depends_on":["1"]}]

Goal: "Research LangGraph vs CrewAI and summarize the differences in a blog post"
CORRECT:
[
  {"id":"1","role":"executor","description":"Use web_search + web_fetch to gather 3-5 high-quality sources comparing LangGraph and CrewAI (2026). Return a dossier of key quotes with URLs.","depends_on":[]},
  {"id":"2","role":"worker","description":"Using the dossier from subgoal 1, write a 400-word blog post comparing the two frameworks for a technical audience.","depends_on":["1"]}
]"""


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _clean_json(s: str) -> str:
    """Remove trailing commas, quote bare keys — tolerate common LLM mistakes."""
    # Strip trailing commas before ] or }
    s = re.sub(r",(\s*[\]}])", r"\1", s)
    # Convert single-quoted JSON-ish strings to double-quoted where safe
    # (skip lines with double quotes already present).
    return s


def _parse_subgoals(text: str) -> list[dict]:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON array found in planner output:\n{text[:500]}")
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Retry with whitespace/trailing-comma cleanup
        return json.loads(_clean_json(candidate))


def _format_memory_context(lessons: list[dict], similar_skills: list[dict]) -> str:
    """Format recalled lessons/patterns/skills for the planner prompt.

    Lessons split into two flavours:
      - success patterns (text starts with '[PATTERN]') — transferable techniques
        mined from solved tasks; planner should actively consider them
      - regular lessons — failure-mode advice ("don't X", "always Y")
    Rendered in separate sections so the planner treats them differently.
    """
    patterns = []
    other_lessons = []
    for l in lessons:
        text = l.get("text", "")
        if text.startswith("[PATTERN]"):
            patterns.append(l)
        else:
            other_lessons.append(l)

    parts = []
    if patterns:
        parts.append("PROVEN TECHNIQUES FROM SIMILAR SOLVED PROBLEMS (consider using):")
        for p in patterns:
            text = p.get("text", "")
            src = p.get("source", "")
            marker = "🧠" if src == "shared" else "📎"
            parts.append(f"  {marker} {text[:400]}")

    if other_lessons:
        parts.append("\nLESSONS FROM PRIOR SESSIONS (avoid these mistakes):")
        for l in other_lessons:
            parts.append(f"  - [{l['severity']}] {l['text']}")

    if similar_skills:
        parts.append("\nRELEVANT SKILLS (prior successful tool-call sequences):")
        for s in similar_skills:
            parts.append(f"  - {s['name']} (used {s['success_count']}x): {s['description']}")

    return "\n".join(parts) if parts else "(no prior experience with similar tasks)"


def plan_node(state: AgentState) -> dict:
    """Called at session start AND when reflector flags stuck==True.

    Wrapped in an `invoke_agent anvil.planner` span so nested LLM calls +
    memory lookups roll up under one trace row in Grafana / Tempo.
    """
    from .. import telemetry as _tel
    goal = state["goal"]
    session_id = state.get("session_id", "") or ""
    cycle = state.get("iterations", 0) if isinstance(state, dict) else 0

    with _tel.agent_span("anvil.planner", session_id=session_id, cycle=cycle, node="planner") as _span:
        try:
            _span.set_attribute("anvil.goal", (goal or "")[:300])
        except Exception:
            pass
        return _plan_node_inner(state)


def _plan_node_inner(state: AgentState) -> dict:
    """Actual planner body — extracted so the telemetry wrapper can stay thin."""
    goal = state["goal"]
    fleet = CONFIG["fleet"]
    client = OpenAI(base_url=fleet.planner.base_url, api_key="EMPTY")

    # Pull prior lessons (semantic-similar to the goal) + relevant skills
    mem = Memory()
    skills = SkillLibrary()
    recalled = mem.recall_lessons(goal, k=6)
    similar = skills.recall(goal, k=3)
    memory_context = _format_memory_context(recalled, similar)

    prior_lessons_in_session = state.get("lessons", [])
    is_replan = state.get("stuck", False) or bool(prior_lessons_in_session)

    # Pre-fetch library docs + file previews for libraries/files mentioned in the goal
    doc_ctx = ""
    try:
        doc_ctx = build_doc_context(goal)
    except Exception as e:
        doc_ctx = f"(doc_context unavailable: {e})"

    # Ground truth inventory so the planner can't reference hallucinated functions
    inv_block = ""
    try:
        inv = build_inventory(str(CONFIG["paths"].workspace))
        inv_block = format_inventory_for_prompt(inv, max_funcs=30)
    except Exception as e:
        inv_block = f"(inventory unavailable: {e})"

    user_content = [f"GOAL:\n{goal}", f"\n\n{memory_context}"]
    if inv_block:
        user_content.append(f"\n\n{inv_block}")
    if doc_ctx:
        user_content.append(f"\n\n{doc_ctx}")
    if is_replan:
        in_session = "\n".join(f"  - {l.text}" for l in prior_lessons_in_session[-5:])
        user_content.append(
            f"\n\nTHIS IS A REPLAN. The previous attempt got stuck. Recent lessons from THIS session:\n{in_session}\n\nProduce a REVISED plan that addresses these issues."
        )

    r = client.chat.completions.create(
        model=fleet.planner.name,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": "\n".join(user_content)},
        ],
        max_tokens=fleet.planner.max_tokens,
        temperature=fleet.planner.temperature,
    )
    raw = r.choices[0].message.content or ""
    try:
        parsed = _parse_subgoals(raw)
    except Exception as e:
        return {
            "status": "failed",
            "error": f"planner JSON parse failed: {e}\nRAW:\n{raw[:600]}",
            "history": [HistoryEvent(kind="plan", content=f"FAILED: {e}", data={"raw": raw}, timestamp=_now())],
        }

    subgoals = [
        Subgoal(
            id=str(sg.get("id", i + 1)),
            description=sg["description"],
            depends_on=list(sg.get("depends_on", [])),
            role=sg.get("role", "executor"),
        )
        for i, sg in enumerate(parsed)
    ]

    event = HistoryEvent(
        kind="replan" if is_replan else "plan",
        content=f"{len(subgoals)} subgoals: " + "; ".join(f"{s.id}:{s.description[:60]}" for s in subgoals),
        data={"subgoals": [{"id": s.id, "desc": s.description, "role": s.role, "depends_on": s.depends_on} for s in subgoals],
              "recalled_lessons": [l["text"] for l in recalled],
              "matched_skills": [s["name"] for s in similar]},
        timestamp=_now(),
    )

    return {
        "subgoals": subgoals,
        "current_subgoal_idx": 0,
        "status": "executing",
        "stuck": False,
        "memory_context": memory_context,
        "iterations": state.get("iterations", 0) + 1,
        "history": [event],
    }
