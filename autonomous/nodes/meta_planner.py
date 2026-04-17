"""
Meta-planner — intrinsic goal generation (IMGEP-style).

When the daemon is between goals, this node proposes the NEXT goal based on:
  - The project's current state (README, git status, recent changes)
  - Prior session outcomes + lessons (memory)
  - The skill library (what we already know how to do)

Scoring heuristic: prefer goals with high expected LEARNING PROGRESS —
  - Not trivial (already solved)
  - Not impossible (no relevant skills)
  - Addresses a gap flagged by recent lessons
"""
from __future__ import annotations
import json
import re
from typing import Optional

from openai import OpenAI

from ..config import CONFIG
from ..memory import Memory, SkillLibrary
from .inventory import (
    build_inventory, format_inventory_for_prompt, validate_goal_against_inventory
)


META_PROMPT = """You are the Meta-Planner of an autonomous coding agent operating in a loop.

You have just finished a session. Propose the NEXT GOAL for yourself.

**READ THE WORKSPACE GROUND TRUTH BLOCK CAREFULLY.** It lists the actual function and class names that exist. You MUST ONLY propose goals that reference names from those lists. Do NOT invent function names like `step4_train_model` or `step5_evaluate_model` if they're not in the ground truth — those are hallucinations and will be rejected.

Choose a goal that addresses a concrete, observable gap:
  - Functions in "MISSING DOCSTRINGS" → add a docstring
  - Bare `except:` lines → narrow the handler
  - Functions listed but lacking tests in "EXISTING TESTS" → write a test
  - Modified files in GIT STATUS → review/stabilize them
  - Pattern that emerged in LESSONS → apply it

Rules:
  - Every function name in your goal MUST appear in the ground-truth lists.
  - Every file path in your goal MUST be a real file from the source_files / test_files lists.
  - If nothing meaningful is left to do, say exactly: `goal: "IDLE - all visible improvements are complete"` and we'll pause.
  - Goal must be CONCRETE and BOUNDED (completable in 5-20 agent steps).

Avoid:
  - Duplicates of recent completed goals (the AVOID list below)
  - Anything requiring credentials you don't have

Return ONLY a JSON object:
{
  "goal": "The concrete next goal, one paragraph.",
  "rationale": "Why this now? What will you learn?",
  "estimated_steps": 8,
  "referenced_names": ["names_from_ground_truth_used_in_this_goal"]
}"""


def _read_excerpt(path: str, max_chars: int = 2000) -> str:
    try:
        from pathlib import Path
        p = Path(path)
        if not p.exists() or not p.is_file():
            return ""
        text = p.read_text(encoding="utf-8", errors="replace")
        return text[:max_chars]
    except Exception:
        return ""


def _git_status(workspace: str) -> str:
    import subprocess
    try:
        r = subprocess.run(["git", "-C", workspace, "status", "--short"], capture_output=True, text=True, timeout=5)
        return r.stdout[:1500]
    except Exception:
        return ""


def _git_recent(workspace: str, n: int = 10) -> str:
    import subprocess
    try:
        r = subprocess.run(["git", "-C", workspace, "log", f"-{n}", "--oneline"], capture_output=True, text=True, timeout=5)
        return r.stdout[:2000]
    except Exception:
        return ""


def _extract_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object")
    return json.loads(text[start : end + 1])


def propose_next_goal(
    workspace: str | None = None,
    backlog: list[str] | None = None,
    avoid: list[str] | None = None,
) -> dict:
    """
    Returns {"goal": str, "rationale": str, "estimated_steps": int}.
    Raises on parse failure.
    """
    ws = workspace or str(CONFIG["paths"].workspace)
    mem = Memory()
    skills = SkillLibrary()

    recent_sessions = mem.recent_sessions(10)
    session_lines = [f"- [{s['outcome'] or 'running'}] {s['goal'][:100]}" for s in recent_sessions]

    # Collect recent lessons: query with a broad term + recent session goals
    lessons = []
    for s in recent_sessions[:3]:
        for l in mem.recall_lessons(s["goal"], k=3):
            lessons.append(l["text"])
    # Dedup
    seen, lesson_lines = set(), []
    for l in lessons:
        if l not in seen:
            lesson_lines.append(f"- {l}")
            seen.add(l)
    lesson_lines = lesson_lines[:10]

    skill_lines = [f"- {r['name']} (used {r['success_count']}x): {r['description']}" for r in skills.list_all()[:10]]

    readme = _read_excerpt(f"{ws}/README.md", 1500)
    gstatus = _git_status(ws)
    glog = _git_recent(ws, 10)

    backlog_lines = [f"- {b}" for b in (backlog or [])]
    avoid_lines = [f"- {a}" for a in (avoid or [])]

    # ── Ground truth: scan the workspace for real functions/classes/tests ──
    inv = build_inventory(ws)
    inventory_block = format_inventory_for_prompt(inv, max_funcs=40)

    # ── Semantic dedup of avoid list ──
    # Embed recent goals; the LLM sees the avoid list but we also enforce a hard
    # reject-loop if it proposes something semantically identical.
    avoid_texts: list[str] = list(avoid or [])
    avoid_embeddings = None
    if avoid_texts:
        try:
            import numpy as np
            from ..memory import Memory as _M
            emb = _M.embedder()
            avoid_embeddings = emb.encode(avoid_texts, normalize_embeddings=True)
        except Exception:
            avoid_embeddings = None

    def _too_similar(new_goal: str, threshold: float = 0.82) -> tuple[bool, float, str]:
        if avoid_embeddings is None or not avoid_texts:
            return False, 0.0, ""
        try:
            import numpy as np
            from ..memory import Memory as _M
            q = _M.embedder().encode([new_goal], normalize_embeddings=True)[0]
            sims = avoid_embeddings @ q
            i = int(np.argmax(sims))
            return bool(sims[i] >= threshold), float(sims[i]), avoid_texts[i]
        except Exception:
            return False, 0.0, ""

    ctx_parts = [
        f"WORKSPACE: {ws}",
        inventory_block,
        f"\nREADME (excerpt):\n{readme}" if readme else "",
        f"\nPRIOR SESSIONS ({len(recent_sessions)}):\n" + "\n".join(session_lines) if session_lines else "",
        f"\nLESSONS:\n" + "\n".join(lesson_lines) if lesson_lines else "",
        f"\nSKILLS:\n" + "\n".join(skill_lines) if skill_lines else "",
        f"\nBACKLOG:\n" + "\n".join(backlog_lines) if backlog_lines else "",
        f"\nAVOID (recently completed — do NOT re-propose anything semantically similar):\n" + "\n".join(avoid_lines) if avoid_lines else "",
    ]
    ctx = "\n".join(p for p in ctx_parts if p)

    fleet = CONFIG["fleet"]
    client = OpenAI(base_url=fleet.planner.base_url, api_key="EMPTY")

    # ── Validation+retry loop ──
    last_error = ""
    for attempt in range(3):
        user_msg = ctx
        if last_error:
            user_msg = f"{ctx}\n\nPREVIOUS ATTEMPT REJECTED:\n{last_error}\n\nTry again with a valid proposal."

        r = client.chat.completions.create(
            model=fleet.planner.name,
            messages=[
                {"role": "system", "content": META_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=1024,
            temperature=0.4 if attempt == 0 else 0.2,  # lower temp on retry
        )
        raw = r.choices[0].message.content or ""
        try:
            parsed = _extract_json(raw)
        except Exception as e:
            last_error = f"invalid JSON: {e}"
            continue

        goal_text = (parsed.get("goal") or "").strip()
        if not goal_text or goal_text.upper().startswith("IDLE"):
            return {"goal": "", "rationale": parsed.get("rationale", ""), "idle": True}

        # A: Validate against inventory
        v = validate_goal_against_inventory(goal_text, inv)
        if not v.ok:
            last_error = v.message
            continue

        # C: Semantic dedup
        sim_hit, score, matched = _too_similar(goal_text)
        if sim_hit:
            last_error = (f"goal is semantically a duplicate of a recent one (cosine={score:.2f}): "
                          f"'{matched[:120]}...'. Propose something substantively different.")
            continue

        # OK
        parsed["_validation"] = {"attempt": attempt + 1, "inventory_size": len(inv.functions)}
        return parsed

    # All retries failed — fall back to a deterministic suggestion
    return _deterministic_fallback(inv, avoid_texts, last_error)


def _deterministic_fallback(inv, avoid_texts: list[str], last_error: str) -> dict:
    """If the LLM keeps hallucinating, generate a grounded goal from the inventory directly."""
    # Priority 1: bare excepts
    if inv.bare_excepts:
        bx = inv.bare_excepts[0]
        goal = (f"Replace the bare `except:` at {bx.module}:{bx.lineno} with a narrow exception handler that "
                f"logs context. Read a 6-line window around that line first, use edit_file to replace "
                f"ONLY the bare except line. Verify file still parses.")
        return {"goal": goal, "rationale": "deterministic fallback: bare-except fix", "estimated_steps": 5,
                "_validation": {"fallback": True, "last_error": last_error}}

    # Priority 2: functions missing docstrings
    if inv.missing_docstrings:
        key = inv.missing_docstrings[0]
        module, fn = key.split("::", 1)
        goal = (f"Add a Google-style docstring to `{fn}` in {module}. Read the function body first to "
                f"understand what it does, then use edit_file to replace the def line + insert docstring. "
                f"Do not change any logic. Verify syntax after.")
        return {"goal": goal, "rationale": "deterministic fallback: add docstring", "estimated_steps": 4,
                "_validation": {"fallback": True, "last_error": last_error}}

    # Priority 3: functions without return type hints
    if inv.missing_return_hints:
        key = inv.missing_return_hints[0]
        module, fn = key.split("::", 1)
        goal = (f"Add a return type hint to `{fn}` in {module}. Read the function body first to determine "
                f"what it actually returns, then use edit_file to update only the def line with the hint.")
        return {"goal": goal, "rationale": "deterministic fallback: add return hint", "estimated_steps": 3,
                "_validation": {"fallback": True, "last_error": last_error}}

    return {"goal": "", "rationale": f"idle — nothing concrete to do (last LLM error: {last_error})",
            "idle": True, "_validation": {"fallback": True}}
