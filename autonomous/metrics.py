"""Session-level metrics: token accounting, cost estimation, error taxonomy.

Every LLM call records its `response.usage` here. The retrospective reads the
accumulator for cost-per-task reporting, and the reflector can warn the agent
when budgets get tight.

Cost estimates are best-effort — they're correct for the listed models at the
rates published at the time of writing, and a no-op for anything we haven't
configured. Local-fleet usage is billed as $0.0.

Error taxonomy: a small classifier that maps raw error strings into
actionable categories. Used by retrospective for failure aggregation.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any


# ---- model pricing (USD per 1M tokens, input/output) ------------------------
# Local fleet = $0. Add commercial models here if/when they get used.

_PRICING: dict[str, tuple[float, float]] = {
    # Local/self-hosted — free
    "supervisor":   (0.0, 0.0),
    "coder":        (0.0, 0.0),
    "reviewer":     (0.0, 0.0),
    "worker":       (0.0, 0.0),
    # Cloud APIs (if routed to later)
    "claude-sonnet-4-6":     (3.0, 15.0),
    "claude-opus-4-7":       (15.0, 75.0),
    "gpt-5":                 (2.5, 10.0),
    "gpt-5-mini":            (0.5, 1.5),
}


@dataclass
class TokenUsage:
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    node: str = ""  # 'planner' | 'executor' | 'reflector' | 'reviewer' | 'retrospective'
    ts: float = 0.0


@dataclass
class SessionMetrics:
    session_id: str = ""
    calls: list[TokenUsage] = field(default_factory=list)
    by_node: dict[str, dict[str, float]] = field(default_factory=dict)
    errors_by_category: dict[str, int] = field(default_factory=dict)
    start_ts: float = 0.0

    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    def total_cost(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    def summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "total_calls": len(self.calls),
            "total_tokens": self.total_tokens(),
            "total_cost_usd": round(self.total_cost(), 4),
            "by_node": self.by_node,
            "errors_by_category": dict(self.errors_by_category),
            "wall_sec": round(time.time() - self.start_ts, 1) if self.start_ts else 0.0,
        }


class MetricsCollector:
    """Thread-safe session metrics accumulator.

    Intended usage:
        from autonomous.metrics import METRICS
        METRICS.record_llm_call(session_id="abc", model="coder",
                                prompt_tokens=..., completion_tokens=...,
                                node="executor")
    """

    def __init__(self):
        self._sessions: dict[str, SessionMetrics] = {}
        self._lock = threading.Lock()

    def _get(self, session_id: str) -> SessionMetrics:
        with self._lock:
            sm = self._sessions.get(session_id)
            if sm is None:
                sm = SessionMetrics(session_id=session_id, start_ts=time.time())
                self._sessions[session_id] = sm
            return sm

    def record_llm_call(
        self,
        session_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        node: str = "",
    ) -> None:
        if not session_id:
            return
        total = prompt_tokens + completion_tokens
        in_rate, out_rate = _PRICING.get(model, (0.0, 0.0))
        cost = (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000

        usage = TokenUsage(
            model=model, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, total_tokens=total,
            cost_usd=cost, node=node, ts=time.time(),
        )
        with self._lock:
            sm = self._get(session_id)
            sm.calls.append(usage)
            node_stats = sm.by_node.setdefault(node or "unknown", {
                "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0, "cost_usd": 0.0,
            })
            node_stats["calls"] += 1
            node_stats["prompt_tokens"] += prompt_tokens
            node_stats["completion_tokens"] += completion_tokens
            node_stats["total_tokens"] += total
            node_stats["cost_usd"] += cost

    def record_error(self, session_id: str, error_text: str) -> None:
        cat = classify_error(error_text)
        sm = self._get(session_id)
        with self._lock:
            sm.errors_by_category[cat] = sm.errors_by_category.get(cat, 0) + 1

    def session_summary(self, session_id: str) -> dict:
        return self._get(session_id).summary()

    def reset_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


METRICS = MetricsCollector()


# ---- error classification ---------------------------------------------------

_ERROR_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("CONTEXT_OVERFLOW",   re.compile(r"maximum context length|context_length_exceeded|input token.* exceed", re.I)),
    ("RATE_LIMIT",         re.compile(r"rate ?limit|too many requests|429|quota", re.I)),
    ("TIMEOUT",            re.compile(r"\btimeout\b|timed out|timeout.expired", re.I)),
    ("TURN_EXHAUSTED",     re.compile(r"executor exhausted|turn (?:cap|limit)|max_turns", re.I)),
    ("JSON_PARSE",         re.compile(r"json.*(?:decode|parse)|invalid json|expecting value", re.I)),
    ("SCHEMA_VALIDATION",  re.compile(r"validation error for |pydantic|expected .* got", re.I)),
    ("HALLUCINATED_TOOL",  re.compile(r"unknown tool|no such tool|tool .* not found", re.I)),
    ("FILE_NOT_FOUND",     re.compile(r"\bfilenotfounderror\b|no such file", re.I)),
    ("SYNTAX_ERROR",       re.compile(r"syntaxerror|indentationerror", re.I)),
    ("TYPE_ERROR",         re.compile(r"\btypeerror\b", re.I)),
    ("ATTRIBUTE_ERROR",    re.compile(r"\battributeerror\b", re.I)),
    ("VALUE_ERROR",        re.compile(r"\bvalueerror\b", re.I)),
    ("MODULE_NOT_FOUND",   re.compile(r"modulenotfounderror|no module named", re.I)),
    ("AUTH",               re.compile(r"\bauth|unauthorised|unauthorized|403|401", re.I)),
    ("PATH_REJECTED",      re.compile(r"path.*(?:outside|rejected|not allowed)", re.I)),
]


def classify_error(message: str | None) -> str:
    """Map an error string to a taxonomy bucket. Returns 'OTHER' if no pattern matches."""
    if not message:
        return "UNKNOWN"
    for name, pat in _ERROR_PATTERNS:
        if pat.search(message):
            return name
    return "OTHER"


# ---- token extraction helper ------------------------------------------------

def extract_usage(response: Any) -> tuple[int, int]:
    """Pull (prompt_tokens, completion_tokens) from an OpenAI-style response.

    Tolerates missing/None fields. Returns (0, 0) if usage is unavailable.
    """
    try:
        usage = getattr(response, "usage", None) or response.get("usage", {})
        if usage is None:
            return 0, 0
        p = getattr(usage, "prompt_tokens", None)
        c = getattr(usage, "completion_tokens", None)
        if p is None and isinstance(usage, dict):
            p = usage.get("prompt_tokens", 0)
            c = usage.get("completion_tokens", 0)
        return int(p or 0), int(c or 0)
    except Exception:
        return 0, 0
