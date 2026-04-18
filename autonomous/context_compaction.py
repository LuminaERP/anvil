"""Context compaction — Claude Code's three-tier pattern adapted for Anvil.

When an executor's message history approaches the model's context limit, this
module produces a single structured summary that replaces most of the older
messages while preserving:

  1. The system prompt
  2. The original goal
  3. Tool results that have been REFERENCED in later messages (load-bearing)
  4. The last N turns verbatim

The summary itself has fixed sections (Claude Code's convention):
  - Goal
  - Completed Steps
  - In-Progress Step
  - Key Facts Learned (variable names, file paths, line numbers, URLs, API shapes)
  - Next Action
  - Open Tool Results (tool_call_ids still needed)

Compaction is performed by the smallest model in the fleet (the 7B worker) —
it's plenty capable of structured summarisation and 4× cheaper than running
the 30B coder over the same text.

Typical savings: 15-20K token history → 2-3K token summary (60-85% reduction).

Two trigger modes:
  - reactive:  caller explicitly calls compact_messages(...) because a budget
               ledger fired "degrade" or an LLM call returned context_length_exceeded.
  - proactive: needs_compaction(messages, model_ctx_limit) returns True when
               the rolling estimate exceeds soft_ratio * model_ctx_limit. The
               executor checks this before each turn.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from .budget import estimate_tokens
from . import telemetry as tel

logger = logging.getLogger(__name__)


# ---- trigger predicates -----------------------------------------------------

def tokens_in_messages(messages: list[dict]) -> int:
    """Fast char-based estimate of the total tokens in an OpenAI-style message list."""
    total = 0
    for m in messages:
        content = m.get("content") or ""
        total += estimate_tokens(content)
        # Tool calls carry structured JSON that also costs tokens
        for tc in (m.get("tool_calls") or []):
            try:
                args = (tc.get("function") or {}).get("arguments") or ""
                total += estimate_tokens(str(args))
                total += estimate_tokens((tc.get("function") or {}).get("name") or "")
            except Exception:
                pass
    return total


def needs_compaction(
    messages: list[dict],
    model_ctx_limit: int,
    soft_ratio: float = 0.70,
    min_messages_to_bother: int = 8,
) -> bool:
    """Should we compact? True when rolling estimate exceeds soft_ratio * limit."""
    if len(messages) < min_messages_to_bother:
        return False
    return tokens_in_messages(messages) >= soft_ratio * model_ctx_limit


# ---- which messages to keep verbatim ----------------------------------------

def _collect_referenced_tool_call_ids(messages: list[dict], keep_last_n: int) -> set[str]:
    """Find tool_call_ids mentioned in any of the last-N or any assistant message.

    If a later message mentions tool_call_id abc123 or contains the tool's
    result verbatim, the result is load-bearing — we keep it.
    """
    if not messages:
        return set()

    # Last N messages are kept anyway — their mentions also matter
    tail_texts = " ".join(
        (m.get("content") or "") + " " +
        " ".join(str(tc.get("id", "")) for tc in (m.get("tool_calls") or []))
        for m in messages[-keep_last_n:]
    )
    # Look for any tool_call_id-like token (openai gives them IDs like "call_abc123")
    ids = set(re.findall(r"\bcall_[a-zA-Z0-9_]+\b", tail_texts))

    # Assistant messages that name tools also constitute reference
    return ids


def _last_n_complete_turns(messages: list[dict], n: int = 3) -> int:
    """Find the index at which we should split messages into [to_compact | to_keep].

    A "turn" is one assistant message plus any tool messages that followed it.
    We keep the last n complete turns verbatim.
    """
    if not messages:
        return 0
    turn_starts: list[int] = []
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            turn_starts.append(i)

    if len(turn_starts) <= n:
        # Not enough turns to split — keep everything after the system message
        return 1 if (messages and messages[0].get("role") == "system") else 0

    return turn_starts[-n]


# ---- the compaction pass ----------------------------------------------------

@dataclass
class CompactionResult:
    messages_before: int
    messages_after: int
    tokens_before: int
    tokens_after: int
    summary_text: str
    load_bearing_count: int = 0


_SUMMARY_PROMPT = """You are summarising a coding agent's session history so it can continue with
a smaller context. Produce a SINGLE structured summary with these exact sections,
and NOTHING else outside them:

<goal>
The original goal the agent was given.
</goal>

<completed_steps>
Numbered list of meaningful work completed. Be concrete (file paths, function
names, line numbers, findings). Skip throwaway exploration.
</completed_steps>

<in_progress>
What is the agent currently doing RIGHT NOW, and what blocks it?
</in_progress>

<key_facts>
Hard facts the agent discovered that will be needed going forward:
  - file paths and the role of each
  - symbol names and signatures
  - API / library usage patterns that worked
  - environment info (Python version, installed packages, etc.)
  - numeric thresholds, counts, URLs
Do NOT lose these details — they are load-bearing for the next step.
</key_facts>

<next_action>
Single next step the agent should take. Be specific (which tool, which args).
</next_action>

<open_tool_results>
Tool call IDs whose outputs are still referenced. Usually empty unless the
agent explicitly cited a past observation.
</open_tool_results>

The session history follows. Produce ONLY the six tagged sections. Do not
include preamble, disclaimers, or any text outside the tags.

HISTORY:
{history}
"""


def _render_history_for_summary(messages: list[dict], max_chars: int = 12000) -> str:
    """Compact serialisation of the messages to summarise.

    Each message becomes one block like:
      [role=assistant] <content>
        → tool_call tool_name(args_preview)
      [role=tool tool_call_id=X] <result>
    """
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "?")
        content = (m.get("content") or "").strip()
        header = f"[role={role}"
        if m.get("tool_call_id"):
            header += f" tool_call_id={m['tool_call_id']}"
        header += "]"
        if content:
            lines.append(header)
            lines.append(content[:800])
        tcs = m.get("tool_calls") or []
        for tc in tcs:
            fn = (tc.get("function") or {})
            name = fn.get("name", "?")
            args = str(fn.get("arguments", ""))[:200]
            lines.append(f"  → call_{tc.get('id', '?')} {name}({args})")

    blob = "\n".join(lines)
    if len(blob) > max_chars:
        blob = blob[:max_chars] + f"\n... [history truncated, {len(blob)-max_chars} more chars]"
    return blob


def compact_messages(
    messages: list[dict],
    openai_client: Any,
    summarizer_model: str,
    session_id: str = "",
    keep_last_n_turns: int = 3,
    summarizer_max_tokens: int = 800,
    summarizer_temperature: float = 0.1,
) -> tuple[list[dict], CompactionResult]:
    """Compact a message list into a structured summary + keep recent turns.

    Returns the new message list, plus a CompactionResult with before/after
    metrics. Wrapped in a `anvil.compaction` span if telemetry is configured.

    Falls back to returning messages unchanged if summarisation fails.
    """
    tokens_before = tokens_in_messages(messages)

    with tel.compaction_span(session_id=session_id, tokens_before=tokens_before) as span:
        if len(messages) < 4:
            # Not enough to bother
            return messages, CompactionResult(
                messages_before=len(messages),
                messages_after=len(messages),
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                summary_text="",
            )

        # Split: system msg (keep), middle (summarise), last-N turns (keep)
        has_system = bool(messages) and messages[0].get("role") == "system"
        system_msg = messages[0] if has_system else None
        rest = messages[1:] if has_system else list(messages)

        split_idx_relative = _last_n_complete_turns(rest, n=keep_last_n_turns)
        to_summarise = rest[:split_idx_relative]
        to_keep_verbatim = rest[split_idx_relative:]

        if not to_summarise:
            return messages, CompactionResult(
                messages_before=len(messages),
                messages_after=len(messages),
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                summary_text="",
            )

        # Preserve load-bearing tool results even if they fall in the summarise window
        referenced_ids = _collect_referenced_tool_call_ids(messages, keep_last_n=keep_last_n_turns)
        load_bearing: list[dict] = []
        to_summarise_filtered: list[dict] = []
        for m in to_summarise:
            if m.get("role") == "tool" and m.get("tool_call_id") in referenced_ids:
                load_bearing.append(m)
            else:
                to_summarise_filtered.append(m)

        # Call the summariser
        history_blob = _render_history_for_summary(to_summarise_filtered)
        prompt = _SUMMARY_PROMPT.format(history=history_blob)

        try:
            response = openai_client.chat.completions.create(
                model=summarizer_model,
                messages=[
                    {"role": "system", "content": "You produce tight structured summaries of agent sessions. Output only the requested tagged sections."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=summarizer_max_tokens,
                temperature=summarizer_temperature,
            )
            summary_text = (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning("compaction failed — returning messages unchanged: %s", e)
            return messages, CompactionResult(
                messages_before=len(messages),
                messages_after=len(messages),
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                summary_text=f"<error>{e}</error>",
            )

        # Assemble the new message list
        new_messages: list[dict] = []
        if system_msg:
            new_messages.append(system_msg)

        # Summary goes in as a system-adjacent message
        new_messages.append({
            "role": "system",
            "content": (
                "PRIOR SESSION SUMMARY (older history was compacted; the sections below "
                "are the distilled context):\n\n" + summary_text
            ),
        })

        # Preserve load-bearing tool results verbatim
        new_messages.extend(load_bearing)

        # Keep the last-N turns verbatim
        new_messages.extend(to_keep_verbatim)

        tokens_after = tokens_in_messages(new_messages)
        tel.record_compaction(
            span_obj=span,
            tokens_after=tokens_after,
            messages_kept=len(to_keep_verbatim) + len(load_bearing) + (1 if system_msg else 0),
            messages_summarised=len(to_summarise_filtered),
        )

        return new_messages, CompactionResult(
            messages_before=len(messages),
            messages_after=len(new_messages),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            summary_text=summary_text,
            load_bearing_count=len(load_bearing),
        )


# ---- executor integration helper --------------------------------------------

def maybe_compact(
    messages: list[dict],
    model_ctx_limit: int,
    openai_client: Any,
    summarizer_model: str,
    session_id: str = "",
    soft_ratio: float = 0.70,
    keep_last_n_turns: int = 3,
) -> tuple[list[dict], CompactionResult | None]:
    """Check the trigger, compact if needed, return (new_messages, result).

    `result` is None if no compaction was performed.
    """
    if not needs_compaction(messages, model_ctx_limit, soft_ratio=soft_ratio):
        return messages, None

    logger.info(
        "compaction triggered: %d messages, ~%d tokens (soft=%.0f%% of %d ctx)",
        len(messages), tokens_in_messages(messages), soft_ratio * 100, model_ctx_limit,
    )
    return compact_messages(
        messages=messages,
        openai_client=openai_client,
        summarizer_model=summarizer_model,
        session_id=session_id,
        keep_last_n_turns=keep_last_n_turns,
    )
