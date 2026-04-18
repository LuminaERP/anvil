"""Mine transferable algorithmic patterns from successful solves.

Counterpart to verification_lessons.py: that module turns FAILURES into
"don't make this mistake" lessons; this module turns SUCCESSES into
"when you see a problem like X, consider technique Y" patterns.

Why this matters: LCB / BCB / SWE-bench problems cluster around a finite
set of techniques (XOR prefix sums, binary search on the answer, monotonic
stack, topological sort, DP on trees, sliding window invariants, ...).
A pass on task A using technique X should inform task B's planner that
technique X has worked on a similar-shape problem.

We save these as high-confidence shared-pool entries tagged with trigger
keywords, so the planner's recall step surfaces them for similar future
problems.

Public API:
    extract_pattern(success: SuccessContext, client, model) -> Pattern | None
    save_pattern(success, memory, client, model) -> Pattern | None
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SuccessContext:
    """Context from a passing solve worth mining for a pattern."""
    task_id: str
    problem_description: str           # the prompt the agent received
    working_code: str                  # the solution that passed
    domain_hint: str = ""              # 'coding/python', 'coding/stdin', 'ERP/sql', etc.
    test_outcome_summary: str = ""     # e.g. 'passed 3/3 public + 40/40 private'
    difficulty: str = "unknown"        # 'easy' | 'medium' | 'hard' | arbitrary label


@dataclass
class Pattern:
    name: str                          # short: 'xor_prefix_sum'
    technique: str                     # 1-2 sentence description
    when_to_apply: str                 # 1-sentence trigger phrase
    example_snippet: str = ""          # minimal illustrative code
    trigger_keywords: list[str] = field(default_factory=list)
    confidence: str = "medium"         # we'll upgrade via ref_count in shared pool
    source_task: str = ""              # for debug / attribution


# ---- LLM prompt --------------------------------------------------------------

_EXTRACT_PROMPT = """You are a senior engineer teaching an autonomous coding agent.
The agent just SOLVED a problem; your job is to extract the transferable technique
so the agent recognises it on a future similar problem.

CONTEXT:

Problem:
{problem}

Working solution:
```python
{code}
```

{test_summary}
Domain: {domain}

Extract a SINGLE transferable technique-level pattern. The pattern should:
  - name a general technique (e.g. "monotonic stack for next-greater", "XOR
    prefix sum for subarray equals K", "binary search on answer")
  - describe WHEN the technique applies, not just what this specific code does
  - avoid mentioning the concrete variable names or problem values from THIS task
  - be applicable across multiple similar future problems, not unique to this one

If the solution is too trivial to abstract (e.g. one-liner, string formatting,
off-the-shelf library call), return {{"pattern": null}}.

Return STRICT JSON:
{{
  "pattern": "<snake_case short name, 2-4 words>",
  "technique": "<1-2 sentences describing the method>",
  "when_to_apply": "<1 sentence; the recognition signal>",
  "example_snippet": "<5-12 lines of minimal illustrative code in Python>",
  "trigger_keywords": ["<5-10 words likely to appear in problem statements where this technique applies>"],
  "confidence": "low | medium | high"
}}

Return ONLY the JSON object. No prose, no fence.
"""


# ---- extract + save ---------------------------------------------------------

def extract_pattern(
    success: SuccessContext,
    client,
    model: str = "supervisor",
    temperature: float = 0.2,
) -> Pattern | None:
    """Ask the supervisor to name the technique behind a successful solve."""
    if not success.working_code.strip():
        return None

    # Keep the prompt tight — problem can be long, code we truncate aggressively
    prompt = _EXTRACT_PROMPT.format(
        problem=success.problem_description[:2500],
        code=success.working_code[:2000],
        test_summary=f"Test outcome: {success.test_outcome_summary}\n" if success.test_outcome_summary else "",
        domain=success.domain_hint or "coding/python",
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You turn concrete solved code into transferable algorithmic patterns."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=500,
        )
    except Exception as e:
        logger.warning("pattern extraction LLM call failed: %s", e)
        return None

    raw = (response.choices[0].message.content or "").strip()
    raw = _strip_json_fence(raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.debug("pattern JSON parse failed: %s — raw=%s", e, raw[:300])
        return None

    if not parsed.get("pattern"):
        return None

    return Pattern(
        name=str(parsed["pattern"]).strip()[:80],
        technique=str(parsed.get("technique", "")).strip()[:500],
        when_to_apply=str(parsed.get("when_to_apply", "")).strip()[:300],
        example_snippet=str(parsed.get("example_snippet", "")).strip()[:1200],
        trigger_keywords=[str(k).lower() for k in parsed.get("trigger_keywords", [])][:12],
        confidence=str(parsed.get("confidence", "medium")).lower(),
        source_task=success.task_id,
    )


def save_pattern(
    success: SuccessContext,
    memory,
    client,
    model: str = "supervisor",
) -> Pattern | None:
    """Full pipeline: extract pattern → store in shared memory.

    Patterns go into the SAME shared_lessons table as failure lessons but with
    a structured body that the planner can recognise at recall time. We prefix
    the text with '[PATTERN]' so planner prompts can visually distinguish.
    """
    pat = extract_pattern(success, client, model=model)
    if pat is None:
        return None

    # Build the indexable text so semantic-similar problems pull this up
    indexable = (
        f"[PATTERN] {pat.name} :: {pat.technique} :: "
        f"trigger:{', '.join(pat.trigger_keywords)} :: "
        f"when:{pat.when_to_apply}"
    )

    from ..state import Lesson as LessonState
    lesson = LessonState(
        text=indexable,
        severity="info",
        tags=list(pat.trigger_keywords) + ["success_pattern", pat.name],
    )

    try:
        if hasattr(memory, "publish_lesson"):
            memory.publish_lesson(
                session_id=success.task_id,
                lesson=lesson,
                confidence=pat.confidence,
                task_id=success.task_id,
            )
        elif hasattr(memory, "add_lesson"):
            memory.add_lesson(success.task_id, lesson)
    except Exception as e:
        logger.warning("pattern save failed: %s", e)
        return None

    logger.info("saved success pattern: %s (confidence=%s)", pat.name, pat.confidence)
    return pat


def recall_patterns_for_problem(problem_description: str, memory, k: int = 4) -> list[dict]:
    """Retrieve previously-saved patterns relevant to a new problem.

    Returns dicts with keys: name, technique, when_to_apply, confidence, distance.
    Empty list if none found or memory layer doesn't support recall.
    """
    try:
        rows = memory.recall_lessons(problem_description, k=k * 3)
    except AttributeError:
        return []
    except Exception as e:
        logger.debug("pattern recall failed: %s", e)
        return []

    out = []
    for r in rows:
        text = r.get("text") if isinstance(r, dict) else getattr(r, "text", "")
        if not isinstance(text, str) or not text.startswith("[PATTERN]"):
            continue
        parsed = _parse_indexable(text)
        if not parsed:
            continue
        out.append({
            **parsed,
            "distance": r.get("distance") if isinstance(r, dict) else None,
            "source": r.get("source") if isinstance(r, dict) else None,
        })
        if len(out) >= k:
            break
    return out


# ---- helpers ----------------------------------------------------------------

_PATTERN_LINE = re.compile(
    r"^\[PATTERN\]\s*"
    r"(?P<name>[^:]+?)\s*::\s*"
    r"(?P<technique>.+?)\s*::\s*"
    r"trigger:(?P<triggers>[^:]*?)\s*::\s*"
    r"when:(?P<when>.+)$"
)


def _parse_indexable(text: str) -> dict | None:
    m = _PATTERN_LINE.match(text.strip())
    if not m:
        return None
    return {
        "name": m.group("name").strip(),
        "technique": m.group("technique").strip(),
        "when_to_apply": m.group("when").strip(),
        "trigger_keywords": [t.strip() for t in m.group("triggers").split(",") if t.strip()],
    }


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()
