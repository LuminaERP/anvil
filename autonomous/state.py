"""
Shared state that flows through the LangGraph state machine.

Design notes:
- Lists are append-only across nodes; use reducers (operator.add) so parallel
  branches don't clobber each other's writes.
- The "history" is a trace of everything the agent did — its own audit log.
- "current_subgoal_idx" is an int pointer into subgoals; -1 means not yet planning.
"""
from __future__ import annotations
from typing import TypedDict, Annotated, Literal, Optional
import operator
from dataclasses import dataclass, field, asdict


# ---- Domain objects --------------------------------------------------------

@dataclass
class Subgoal:
    id: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    # Assigned role: "executor" for complex reasoning, "worker" for simple parallel work
    role: Literal["executor", "worker"] = "executor"
    status: Literal["pending", "running", "done", "failed", "skipped"] = "pending"
    result: Optional[str] = None
    attempts: int = 0


@dataclass
class ToolCall:
    name: str                          # filesystem.read_file, shell.run_bash, etc.
    args: dict
    result: Optional[str] = None
    error: Optional[str] = None
    duration_ms: int = 0


@dataclass
class HistoryEvent:
    """One atomic thing the agent did. Goes to both state.history and audit.jsonl."""
    kind: Literal["plan", "tool_call", "observation", "reflection", "lesson", "replan", "memory_recall", "subgoal_start", "subgoal_end"]
    subgoal_id: Optional[str] = None
    content: str = ""                  # human-readable summary
    data: dict = field(default_factory=dict)  # structured payload
    timestamp: str = ""                # ISO8601


@dataclass
class Lesson:
    """Reflexion output: something we learned that should persist across sessions."""
    text: str
    severity: Literal["info", "warn", "error"] = "info"
    subgoal_id: Optional[str] = None
    tags: list[str] = field(default_factory=list)


# ---- LangGraph state ------------------------------------------------------

def _append(a: list, b: list) -> list:
    """Reducer for append-only lists. Equivalent to operator.add but explicit."""
    return (a or []) + (b or [])


class _TestBaselineHolder:
    """Process-global singleton for the current session's test baseline.
    Tools read this via SESSION_TEST_BASELINE.get(); daemon/main set it at session start."""
    _value = None

    def get(self):
        return self._value

    def set(self, value) -> None:
        self._value = value

    def clear(self) -> None:
        self._value = None


SESSION_TEST_BASELINE = _TestBaselineHolder()


class AgentState(TypedDict, total=False):
    # Input
    goal: str
    session_id: str                    # for memory scoping + checkpointing

    # Planning output
    subgoals: list[Subgoal]
    current_subgoal_idx: int

    # Per-turn scratch
    last_tool_calls: Annotated[list[ToolCall], _append]
    last_observation: str              # last tool output fed into next planner turn

    # Reflection state
    lessons: Annotated[list[Lesson], _append]
    consecutive_failures: int
    stuck: bool                        # reflector flips this to trigger replan

    # Full trace (audit)
    history: Annotated[list[HistoryEvent], _append]

    # Memory injection — retrieved from prior sessions at plan time
    memory_context: str

    # Terminal
    final_answer: Optional[str]
    status: Literal["planning", "executing", "reflecting", "replanning", "done", "failed", "needs_approval"]
    iterations: int                    # incremented each planner/executor cycle
    error: Optional[str]
