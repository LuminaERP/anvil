"""Token + cost budget enforcement for agent sessions.

Pattern (from the SWE-agent / Aider / Claude Code convergence):

  pre-call:    estimate cost of the pending prompt. If estimate + spent > hard
               cap, either graceful-degrade (trigger compaction, switch to
               cheaper model) or raise BudgetExceeded.
  post-call:   reconcile using the provider's real usage numbers.

Budgets compose: you can set a session-level ledger and still have
per-node sub-budgets that report up. For now the BudgetLedger is flat and
per-session; add sub-budgets when needed.

Cost is computed from a vendored pricing snapshot (see PRICING_TABLE below).
Local vLLM calls cost $0 — they're recorded for token accounting but do not
count against monetary caps.

Key design choice: the ledger is read/written via METRICS (metrics.py), so
there's a single source of truth for "what has this session spent". This
module owns the ENFORCEMENT; metrics.py owns the AGGREGATION.
"""
from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass, field
from typing import Any

from .metrics import METRICS

logger = logging.getLogger(__name__)


# ---- pricing snapshot --------------------------------------------------------
# Keep in sync with LiteLLM's model_prices_and_context_window.json (refresh
# weekly). Entries are (input_cost_per_million, output_cost_per_million).
# Local/self-hosted models = $0.

PRICING_TABLE: dict[str, tuple[float, float]] = {
    # Local fleet
    "supervisor":   (0.0, 0.0),
    "coder":        (0.0, 0.0),
    "reviewer":     (0.0, 0.0),
    "worker":       (0.0, 0.0),
    "planner":      (0.0, 0.0),
    "executor":     (0.0, 0.0),
    # Cloud APIs
    "claude-opus-4-7":       (15.0, 75.0),
    "claude-sonnet-4-6":     (3.0,  15.0),
    "claude-haiku-4-5":      (0.25, 1.25),
    "gpt-5":                 (2.5,  10.0),
    "gpt-5-mini":            (0.5,  1.5),
    "gpt-5-nano":            (0.05, 0.40),
    "gemini-2.5-pro":        (2.0,  8.0),
    "deepseek-v3.1":         (0.27, 1.10),
}


def price_for(model: str) -> tuple[float, float]:
    """Return (input_price_per_m, output_price_per_m). Unknown → (0, 0)."""
    return PRICING_TABLE.get(model, (0.0, 0.0))


# ---- token estimation --------------------------------------------------------

# Research finding: len(text)/3.5 is measurably closer than /4 for English code+prose
# on BPE tokenizers (Qwen, Llama, Claude all cluster around 3.5-3.7 chars/token).
CHARS_PER_TOKEN_ENGLISH = 3.5


def estimate_tokens(text: str) -> int:
    """Fast char-based estimate. Use for pre-call budget checks where speed matters.

    For accurate post-call accounting, use the provider's `usage` block.
    """
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN_ENGLISH))


def estimate_call_cost(
    model: str,
    prompt_text: str,
    max_output_tokens: int = 4096,
) -> tuple[int, int, float]:
    """Pre-call estimate: (prompt_tokens, max_output_tokens, estimated_usd)."""
    p = estimate_tokens(prompt_text)
    in_rate, out_rate = price_for(model)
    cost = (p * in_rate + max_output_tokens * out_rate) / 1_000_000
    return p, max_output_tokens, cost


# ---- the ledger --------------------------------------------------------------

@dataclass
class BudgetLedger:
    """Per-session budget enforcement.

    Settings (any can be None to disable that check):
      hard_cap_usd      — hard dollar cap. Pre-call check fails if projected cost > cap.
      hard_cap_tokens   — total tokens (input+output summed across all calls).
      soft_ratio        — fraction of the hard cap at which graceful_degrade fires.
      allow_degrade     — if True, over-soft-limit triggers compaction attempts
                          via the registered compactor instead of failing immediately.
    """
    session_id: str
    hard_cap_usd: float | None = None
    hard_cap_tokens: int | None = None
    soft_ratio: float = 0.70
    allow_degrade: bool = True

    # Mutable state (not frozen)
    _degrade_triggered: bool = False
    _last_status: str = "ok"   # ok | soft_warn | degraded | exceeded

    @property
    def spent_usd(self) -> float:
        return METRICS.session_summary(self.session_id).get("total_cost_usd", 0.0)

    @property
    def spent_tokens(self) -> int:
        return METRICS.session_summary(self.session_id).get("total_tokens", 0)

    def check_pre_call(
        self,
        model: str,
        prompt_text: str,
        max_output_tokens: int = 4096,
    ) -> "BudgetDecision":
        """Returns a BudgetDecision — caller inspects .action."""
        p_tok, out_tok, cost_est = estimate_call_cost(model, prompt_text, max_output_tokens)
        projected_usd = self.spent_usd + cost_est
        projected_tokens = self.spent_tokens + p_tok + out_tok

        usd_exceeded = (self.hard_cap_usd is not None and projected_usd > self.hard_cap_usd)
        tok_exceeded = (self.hard_cap_tokens is not None and projected_tokens > self.hard_cap_tokens)

        if usd_exceeded or tok_exceeded:
            reason = []
            if usd_exceeded:
                reason.append(f"${projected_usd:.2f} > hard cap ${self.hard_cap_usd:.2f}")
            if tok_exceeded:
                reason.append(f"{projected_tokens:,} tokens > hard cap {self.hard_cap_tokens:,}")
            self._last_status = "exceeded"
            return BudgetDecision(
                action="stop",
                reason="budget exceeded: " + "; ".join(reason),
                estimated_cost_usd=cost_est,
                estimated_tokens=p_tok + out_tok,
                spent_usd=self.spent_usd,
                spent_tokens=self.spent_tokens,
            )

        # Soft-limit check
        soft_usd = (self.hard_cap_usd or math.inf) * self.soft_ratio
        soft_tokens = (self.hard_cap_tokens or math.inf) * self.soft_ratio
        over_soft = (self.spent_usd > soft_usd) or (self.spent_tokens > soft_tokens)

        if over_soft and self.allow_degrade and not self._degrade_triggered:
            self._degrade_triggered = True
            self._last_status = "degraded"
            return BudgetDecision(
                action="degrade",
                reason=(f"soft limit crossed — spent ${self.spent_usd:.2f} / "
                        f"{self.spent_tokens:,} tokens; triggering compaction"),
                estimated_cost_usd=cost_est,
                estimated_tokens=p_tok + out_tok,
                spent_usd=self.spent_usd,
                spent_tokens=self.spent_tokens,
            )

        if over_soft:
            self._last_status = "soft_warn"
        else:
            self._last_status = "ok"

        return BudgetDecision(
            action="proceed",
            reason="",
            estimated_cost_usd=cost_est,
            estimated_tokens=p_tok + out_tok,
            spent_usd=self.spent_usd,
            spent_tokens=self.spent_tokens,
        )

    def record_call(self, model: str, prompt_tokens: int, completion_tokens: int, node: str = "") -> None:
        """Post-call reconciliation using provider's real usage."""
        METRICS.record_llm_call(
            session_id=self.session_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            node=node,
        )

    def status(self) -> str:
        return self._last_status

    def reset_degrade_trigger(self) -> None:
        """Call after a successful compaction so soft-limit can retrigger later."""
        self._degrade_triggered = False


@dataclass
class BudgetDecision:
    """Result of a pre-call budget check."""
    action: str  # "proceed" | "degrade" | "stop"
    reason: str
    estimated_cost_usd: float
    estimated_tokens: int
    spent_usd: float
    spent_tokens: int


class BudgetExceeded(Exception):
    """Raised by nodes when a BudgetDecision says stop and no degradation path is available."""

    def __init__(self, decision: BudgetDecision):
        self.decision = decision
        super().__init__(decision.reason)


# ---- registry of ledgers per session ----------------------------------------

_LEDGERS: dict[str, BudgetLedger] = {}
_LOCK = threading.Lock()


def get_ledger(session_id: str) -> BudgetLedger:
    """Return the ledger for a session, creating one from env defaults on first access."""
    with _LOCK:
        if session_id in _LEDGERS:
            return _LEDGERS[session_id]

        # Env-driven defaults — Anvil daemons / benchmarks set these for runs
        hard_usd = os.environ.get("AGENT_MAX_COST_USD")
        hard_tok = os.environ.get("AGENT_MAX_TOKENS_PER_SESSION")
        soft_ratio = float(os.environ.get("AGENT_BUDGET_SOFT_RATIO", "0.7"))
        allow_degrade = os.environ.get("AGENT_BUDGET_ALLOW_DEGRADE", "1").lower() in ("1", "true", "yes")

        ledger = BudgetLedger(
            session_id=session_id,
            hard_cap_usd=float(hard_usd) if hard_usd else None,
            hard_cap_tokens=int(hard_tok) if hard_tok else None,
            soft_ratio=soft_ratio,
            allow_degrade=allow_degrade,
        )
        _LEDGERS[session_id] = ledger
        return ledger


def reset_ledger(session_id: str) -> None:
    with _LOCK:
        _LEDGERS.pop(session_id, None)
