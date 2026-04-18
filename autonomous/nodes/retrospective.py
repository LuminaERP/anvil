"""
Retrospective — post-session "could I have done this in 50% fewer steps?" pass.

Reads the full history of a completed session, generates:
  1. A condensed skill (if patterns emerged) for the skill library
  2. An "optimization lesson" stored in memory
  3. Transferable verification-failure lessons (new) — any docstring verification
     that fired during the session gets abstracted into a general pattern and
     saved to memory keyed by semantic similarity to the docstring

This is the Hermes-Agent-style learning loop: after every task (success or
failure), the agent retrospectively compresses what it did AND what class of
mistake it made, so future sessions don't repeat them.
"""
from __future__ import annotations
import json
import logging
import re
from typing import Any, Optional

from openai import OpenAI

from ..config import CONFIG
from ..memory import Memory, SkillLibrary, Skill, SkillStep
from ..memory.verification_lessons import VerificationFailure, save_lesson
from ..memory.success_patterns import SuccessContext, save_pattern
from ..state import Lesson

logger = logging.getLogger(__name__)


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
    for e in events[:200]:
        kind = e.get("kind", "?")
        sub = e.get("subgoal_id") or ""
        c = (e.get("content") or "")[:240]
        lines.append(f"[{kind}]{' sg='+sub if sub else ''} {c}")
    return "\n".join(lines)


# ---- verification-failure mining --------------------------------------------

def _collect_verification_failures_from_events(events: list[dict]) -> list[VerificationFailure]:
    """Reconstruct VerificationFailure objects from session events.

    The reflector emits one `kind=verification` event per .py file it checked,
    with the failure payload in `data`. We group by path and produce a
    VerificationFailure per distinct entry point that failed.
    """
    failures: list[VerificationFailure] = []
    # We need both the verification event (has failed count) and the source
    # code at time of failure. Source code isn't in events — we read from disk
    # at retrospective time as best-effort. If the file has since been
    # overwritten with a fixed version, we skip to avoid a misleading lesson.

    from pathlib import Path
    seen_paths: set[str] = set()

    for e in events:
        if e.get("kind") != "verification":
            continue
        data = e.get("data") or {}
        if "failed_count" not in data:
            continue
        if data.get("failed_count", 0) <= 0:
            continue

        path = data.get("path")
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)

        # Read current file state
        try:
            code = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        # Extract docstring from the current file
        try:
            import ast as _ast
            tree = _ast.parse(code)
        except SyntaxError:
            continue

        # Find the first function with a docstring
        entry_point = None
        docstring = ""
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                d = _ast.get_docstring(node)
                if d:
                    entry_point = node.name
                    docstring = d
                    break
        if not entry_point:
            continue

        # Summary has the failure details; we reconstruct failed_examples as best-effort
        # The original failed_examples list isn't in the event (only the summary string),
        # so we parse it back out of the summary. Coarse but works.
        summary = data.get("summary", "")
        failed_examples = _parse_summary_failures(summary)

        failures.append(VerificationFailure(
            entry_point=entry_point,
            docstring=docstring,
            failed_examples=failed_examples,
            broken_code=code[:3000],
        ))

    return failures


_FAILURE_LINE_PATTERN = re.compile(
    r"^\s{4}>>> (?P<source>.+?)\n"
    r"^\s{8}expected: (?P<expected>.+?)\n"
    r"^\s{8}actual:\s+(?P<actual>.+?)(?:  error: (?P<error>.+?))?$",
    re.MULTILINE,
)


def _parse_summary_failures(summary: str) -> list[dict]:
    out = []
    for m in _FAILURE_LINE_PATTERN.finditer(summary):
        out.append({
            "source": m.group("source").strip(),
            "expected": m.group("expected").strip(),
            "actual": (m.group("actual") or "").strip(),
            "error": (m.group("error") or "").strip() or None,
            "kind": ">>>",
        })
    return out


def _mine_verification_lessons(events: list[dict], mem: Memory) -> list[dict]:
    """For each verification failure in the session, extract a transferable
    lesson via the supervisor LLM and persist it to memory.

    Returns a list of saved-lesson descriptors for telemetry.
    """
    fleet = CONFIG["fleet"]
    client = OpenAI(base_url=fleet.supervisor.base_url, api_key="EMPTY") if hasattr(fleet, "supervisor") else \
             OpenAI(base_url=fleet.reflector.base_url, api_key="EMPTY")

    model_name = getattr(fleet.supervisor, "name", None) if hasattr(fleet, "supervisor") else fleet.reflector.name

    saved: list[dict] = []
    for failure in _collect_verification_failures_from_events(events):
        try:
            lesson = save_lesson(failure, mem, client, model=model_name)
            if lesson:
                saved.append({
                    "pattern": lesson.pattern,
                    "confidence": lesson.confidence,
                    "entry_point": failure.entry_point,
                    "trigger_keywords": lesson.trigger_keywords,
                })
        except Exception as e:
            logger.warning("verification lesson extraction failed for %s: %s", failure.entry_point, e)

    return saved


# ---- retrospective entrypoint -----------------------------------------------

def run_retrospective(session_id: str, goal: str, final_status: str) -> dict:
    """
    Produces {optimization_lesson, compressed_steps, save_skill, verification_lessons}.
    Persists lessons + optional skill into memory as a side effect.
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
        parsed = {"error": str(e), "raw": raw[:400]}

    # Persist the optimization lesson — publish to shared pool so siblings see it
    opt = parsed.get("optimization_lesson", "").strip() if isinstance(parsed, dict) else ""
    if opt:
        try:
            if hasattr(mem, "publish_lesson"):
                mem.publish_lesson(
                    session_id=session_id,
                    lesson=Lesson(
                        text=opt,
                        severity="info",
                        tags=["retrospective", "optimization"],
                    ),
                    confidence="medium",  # retrospective optimisations are medium-confidence
                )
            else:
                mem.add_lesson(session_id, Lesson(
                    text=opt, severity="info",
                    tags=["retrospective", "optimization"],
                ))
        except Exception as e:
            logger.debug("failed to persist optimization_lesson: %s", e)

    # Maybe save the compressed skill
    sk = parsed.get("save_skill") if isinstance(parsed, dict) else None
    steps_in = parsed.get("compressed_steps", []) if isinstance(parsed, dict) else []
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

    # Mine verification failures into transferable lessons
    try:
        verification_lessons = _mine_verification_lessons(events, mem)
        if verification_lessons:
            parsed["verification_lessons"] = verification_lessons
            logger.info("retrospective saved %d verification lessons", len(verification_lessons))
    except Exception as e:
        logger.warning("verification lesson mining failed: %s", e)

    # Mine passing sessions for transferable algorithmic patterns
    if final_status == "done":
        try:
            saved = _mine_success_patterns(events, mem, goal)
            if saved:
                parsed["success_patterns"] = saved
                logger.info("retrospective saved %d success patterns", len(saved))
        except Exception as e:
            logger.warning("success pattern mining failed: %s", e)

    return parsed


def _mine_success_patterns(events: list[dict], mem: Memory, goal: str) -> list[dict]:
    """Collect code files the agent successfully wrote in this session and
    extract algorithmic patterns via the supervisor LLM. Publish each to the
    shared pool so siblings on similar problems can recall them.
    """
    from pathlib import Path
    import ast as _ast

    # Find .py files written this session. Same heuristic as reflector.
    write_events = [
        e for e in events
        if e.get("kind") == "tool_call"
        and "write_file" in (e.get("content") or "")
        and ".py" in (e.get("content") or "")
    ]
    # Also check the verification events — a 'verification: N/N passed'
    # indicates the file passed its contract.
    verify_ok_paths: set[str] = set()
    for e in events:
        if e.get("kind") == "verification":
            data = e.get("data") or {}
            if data.get("total", 0) > 0 and data.get("passed", 0) == data.get("total", 0):
                if data.get("path"):
                    verify_ok_paths.add(str(data["path"]))

    # Fall back to scanning workspace if no verification signals
    workspace = Path(CONFIG["paths"].workspace)
    candidate_files: list[Path] = []
    if verify_ok_paths:
        for p in verify_ok_paths:
            fp = Path(p)
            if fp.exists() and fp.suffix == ".py":
                candidate_files.append(fp)
    else:
        # Pick up to 5 recent .py files in workspace / test_output
        for pattern in ["solution.py", "*.py"]:
            for fp in workspace.rglob(pattern):
                if fp.is_file() and ".git" not in str(fp) and "__pycache__" not in str(fp):
                    candidate_files.append(fp)
                if len(candidate_files) >= 5:
                    break
            if candidate_files:
                break

    if not candidate_files:
        return []

    fleet = CONFIG["fleet"]
    client = OpenAI(base_url=fleet.supervisor.base_url, api_key="EMPTY") if hasattr(fleet, "supervisor") else \
             OpenAI(base_url=fleet.reflector.base_url, api_key="EMPTY")
    model_name = getattr(fleet.supervisor, "name", None) if hasattr(fleet, "supervisor") else fleet.reflector.name

    saved = []
    for fp in candidate_files[:3]:  # cap to 3 per session so we don't explode LLM cost
        try:
            code = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(code) < 60 or len(code) > 8000:
            continue  # too trivial or too big
        # Parseable?
        try:
            _ast.parse(code)
        except SyntaxError:
            continue

        success = SuccessContext(
            task_id=f"{fp.stem}",
            problem_description=goal[:3000],
            working_code=code,
            domain_hint="coding/python",
            test_outcome_summary=f"verified at {fp.name}" if str(fp) in verify_ok_paths else "",
        )
        try:
            pat = save_pattern(success, mem, client, model=model_name)
        except Exception as e:
            logger.debug("pattern extract error for %s: %s", fp.name, e)
            pat = None
        if pat:
            saved.append({
                "name": pat.name,
                "source_file": str(fp),
                "confidence": pat.confidence,
            })
    return saved
