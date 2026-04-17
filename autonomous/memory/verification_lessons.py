"""Mine verification failures into transferable lessons stored in semantic memory.

When docstring verification catches a real bug, we have three valuable artifacts:
  1. The docstring (describes the problem domain)
  2. The failing example (concrete case where the bug manifests)
  3. The broken code the agent wrote (symptom of the misunderstanding)

A naive memory system would save "here's what went wrong in task HumanEval/108".
That's not transferable — it only helps if the agent sees HumanEval/108 again.

A better memory system extracts the GENERAL pattern:
  "When summing digits of a signed integer, the first digit should inherit
  the sign of the number. ABS-THEN-SUM loses the sign contribution."

This module asks the supervisor LLM to perform that abstraction, keyed on
semantic embedding of the docstring so retrieval is similarity-based.

Public API:
    extract_lesson(failure: VerificationFailure, client: OpenAI | None) -> Lesson | None
    save_lesson(failure, memory, client) -> bool
    recall_lessons_for_docstring(docstring: str, memory, k: int) -> list[Lesson]
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VerificationFailure:
    """Context needed to distill a lesson from a verification failure."""
    entry_point: str
    docstring: str
    failed_examples: list[dict]       # [{"source": ..., "expected": ..., "actual": ..., "error": ...}]
    broken_code: str                  # the function body that failed
    iteration_count: int = 1          # how many times the agent tried before giving up


@dataclass
class Lesson:
    pattern: str                      # one-line pattern name (for grep-like triggers)
    rule: str                         # 1-2 sentence rule
    trigger_keywords: list[str]       # words likely to appear in future problems of this class
    source_failure: str = ""          # reference to the originating failure (debug)
    confidence: str = "medium"        # 'low' | 'medium' | 'high'


# ---- LLM-based lesson extraction ---------------------------------------------

_EXTRACTION_PROMPT = """A coding agent failed docstring verification on a Python function.
Your job is to extract a SINGLE transferable lesson that will help the agent
avoid the same class of mistake on *future* problems with similar shape.

CONTEXT:

Function: {entry_point}

Docstring:
{docstring}

Failing examples (what the function should have returned vs what the agent's code did):
{failure_table}

Code the agent wrote (that has the bug):
```python
{broken_code}
```

Return STRICT JSON with these keys:
{{
  "pattern": "<one-line name for this class of bug, e.g. 'signed-digit sum of negatives'>",
  "rule": "<1-2 sentence rule describing WHAT to do (positive framing, not 'don't X')>",
  "trigger_keywords": ["<5-8 words likely to appear in similar future problems>"],
  "confidence": "low | medium | high"
}}

Rules:
- The pattern should generalise — don't mention specific variable names, function names, or test values
- The rule must be actionable — tell a future agent what to do, not just what the mistake was
- Trigger keywords should enable recall: if a future docstring contains several of these words,
  this lesson should probably be considered
- Confidence high = this bug pattern is common and the fix is well-established
- Confidence low = this might be a one-off or the failure cause is unclear
- If the failure does not suggest a generalisable lesson (e.g. the bug is a typo or a
  single-character mistake with no transferable pattern), return
  {{"pattern": null, "rule": "", "trigger_keywords": [], "confidence": "low"}}

Return ONLY the JSON object. No prose, no markdown fence.
"""


def extract_lesson(
    failure: VerificationFailure,
    client,
    model: str = "supervisor",
    temperature: float = 0.2,
) -> Lesson | None:
    """Call the supervisor LLM to abstract a lesson from a concrete failure.

    Returns None if the model says no generalisable lesson applies.
    """
    failure_table = _format_failures_for_prompt(failure.failed_examples)

    prompt = _EXTRACTION_PROMPT.format(
        entry_point=failure.entry_point,
        docstring=failure.docstring[:2000],
        failure_table=failure_table[:1500],
        broken_code=failure.broken_code[:2000],
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a senior software engineer who mentors coding agents by turning specific bugs into general lessons."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=400,
        )
    except Exception as e:
        logger.warning("lesson extraction LLM call failed: %s", e)
        return None

    raw = (response.choices[0].message.content or "").strip()
    raw = _strip_json_fence(raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("lesson JSON parse failed: %s — raw=%s", e, raw[:300])
        return None

    if not parsed.get("pattern"):
        return None  # No generalisable lesson

    return Lesson(
        pattern=str(parsed["pattern"]).strip()[:120],
        rule=str(parsed.get("rule", "")).strip()[:400],
        trigger_keywords=[str(k).lower() for k in parsed.get("trigger_keywords", [])][:10],
        confidence=str(parsed.get("confidence", "medium")).lower(),
        source_failure=failure.entry_point,
    )


def save_lesson(
    failure: VerificationFailure,
    memory,
    client,
    model: str = "supervisor",
) -> Lesson | None:
    """Full pipeline: extract lesson from failure, persist it, return it.

    `memory` must be an autonomous.memory.episodic.Memory instance. If the memory
    save fails or the model doesn't provide a generalisable lesson, returns None.
    """
    lesson = extract_lesson(failure, client, model=model)
    if lesson is None:
        logger.info("lesson extraction: no generalisable pattern for %s", failure.entry_point)
        return None

    # Build a memory entry. We key on the docstring + lesson pattern so that
    # future docstrings with similar content recall this lesson.
    indexable_text = f"{lesson.pattern} :: {lesson.rule} :: keywords:{', '.join(lesson.trigger_keywords)}"
    snippet_text = f"DOCSTRING:\n{failure.docstring[:600]}"

    try:
        # Try the structured Memory.save_lesson signature first; fall back to whatever
        # signature is actually present.
        _try_save(memory, indexable_text, snippet_text, lesson, failure)
    except Exception as e:
        logger.warning("lesson save failed: %s", e)
        return None

    logger.info("saved verification lesson: %s (%s)", lesson.pattern, lesson.confidence)
    return lesson


def recall_lessons_for_docstring(docstring: str, memory, k: int = 3) -> list[Lesson]:
    """Retrieve semantically similar lessons for a new docstring."""
    # Memory.recall_lessons returns raw records — shape depends on Anvil's memory layer.
    # We adapt on the caller side.
    try:
        raw = memory.recall_lessons(docstring, k=k)
    except AttributeError:
        # Older memory API variant
        raw = memory.recall(docstring, k=k)
    except Exception as e:
        logger.warning("recall_lessons failed: %s", e)
        return []

    out = []
    for r in raw:
        text = r.get("text") if isinstance(r, dict) else getattr(r, "text", "")
        if "::" in text:
            pattern, _, tail = text.partition(" :: ")
            rule, _, kw_part = tail.partition(" :: keywords:")
            keywords = [w.strip() for w in kw_part.split(",") if w.strip()]
            out.append(Lesson(
                pattern=pattern.strip(),
                rule=rule.strip(),
                trigger_keywords=keywords,
            ))
    return out


# ---- helpers -----------------------------------------------------------------

def _format_failures_for_prompt(failed_examples: list[dict]) -> str:
    lines = []
    for i, ex in enumerate(failed_examples[:5], 1):
        src = ex.get("source", "<unknown>")[:100]
        exp = ex.get("expected", "<unknown>")[:80]
        act = ex.get("actual") or ex.get("error") or "<no output>"
        lines.append(f"  {i}. {src}")
        lines.append(f"     expected: {exp}")
        lines.append(f"     got:      {str(act)[:80]}")
    if len(failed_examples) > 5:
        lines.append(f"  ... and {len(failed_examples) - 5} more failures")
    return "\n".join(lines)


def _strip_json_fence(text: str) -> str:
    """Remove ```json fences around the JSON object, if present."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _try_save(memory, indexable_text: str, snippet_text: str, lesson: Lesson, failure: VerificationFailure) -> None:
    """Adapt to whatever `save_lesson`/`save_memory` signature the Memory class exposes."""
    # Pattern 1: Memory.save_lesson(text, severity, context, ...)
    if hasattr(memory, "save_lesson"):
        try:
            memory.save_lesson(
                text=indexable_text,
                severity={"high": "high", "medium": "medium", "low": "low"}.get(lesson.confidence, "medium"),
                context=snippet_text,
                tags=lesson.trigger_keywords,
            )
            return
        except TypeError:
            # Fallback to minimal signature
            memory.save_lesson(text=indexable_text, severity=lesson.confidence)
            return
    # Pattern 2: Memory.add(text, ...)
    if hasattr(memory, "add"):
        memory.add(text=indexable_text, metadata={"source": "verification_failure", "lesson_pattern": lesson.pattern})
        return
    raise RuntimeError("Memory object has neither save_lesson() nor add()")
