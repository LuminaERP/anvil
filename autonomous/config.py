"""
Runtime configuration. All endpoints, budgets, and safety knobs live here.
Override via environment variables; no secrets in the file.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelEndpoint:
    name: str                 # served-model-name on the vLLM server
    base_url: str             # OpenAI-compatible base URL ending in /v1
    max_tokens: int = 2048
    temperature: float = 0.2


@dataclass(frozen=True)
class Fleet:
    # Planner: decomposes goal into subgoal tree. Needs good instruction-following.
    planner: ModelEndpoint = field(
        default_factory=lambda: ModelEndpoint(
            name="supervisor",
            base_url=os.environ.get("PLANNER_URL", "http://localhost:8000/v1"),
            max_tokens=2048,
            temperature=0.1,
        )
    )
    # Executor: the ReAct agent. Needs strong tool use + code reasoning.
    executor: ModelEndpoint = field(
        default_factory=lambda: ModelEndpoint(
            name="coder",
            base_url=os.environ.get("EXECUTOR_URL", "http://localhost:8001/v1"),
            max_tokens=4096,
            temperature=0.2,
        )
    )
    # Reflector: critiques action batches, writes lessons to memory.
    reflector: ModelEndpoint = field(
        default_factory=lambda: ModelEndpoint(
            name="reviewer",
            base_url=os.environ.get("REFLECTOR_URL", "http://localhost:8002/v1"),
            max_tokens=1024,
            temperature=0.0,
        )
    )
    # Worker pool: small, concurrent, for independent subgoal fan-out.
    worker: ModelEndpoint = field(
        default_factory=lambda: ModelEndpoint(
            name="worker",
            base_url=os.environ.get("WORKER_URL", "http://localhost:8003/v1"),
            max_tokens=2048,
            temperature=0.3,
        )
    )


@dataclass(frozen=True)
class Budget:
    """Hard limits to prevent runaway agents."""
    max_iterations: int = 25            # total plan/act/reflect cycles
    max_tool_calls_per_step: int = 8    # per-executor-turn tool-call cap
    max_parallel_workers: int = 4       # concurrency on worker pool
    executor_max_turns: int = 18        # inner ReAct loop cap (default if no difficulty tag)
    max_context_chars: int = 40_000     # truncate long tool outputs

    # Difficulty-aware turn caps: easy tasks finish fast, hard ones get room
    # to iterate. Consumers call resolve_turn_cap(difficulty) rather than
    # reading executor_max_turns directly.
    turn_caps_by_difficulty: dict = field(default_factory=lambda: {
        "trivial": 6,
        "easy":    10,
        "medium":  18,
        "hard":    28,
        "extreme": 40,
    })


def resolve_turn_cap(difficulty: str | None, default: int = 18) -> int:
    """Look up the turn cap for a difficulty tag; fall back to default."""
    caps = {
        "trivial": 6, "easy": 10, "medium": 18, "hard": 28, "extreme": 40,
    }
    if difficulty is None:
        return default
    key = str(difficulty).strip().lower()
    return caps.get(key, default)


@dataclass(frozen=True)
class Paths:
    workspace: Path = field(default_factory=lambda: Path(os.environ.get("AGENT_WORKSPACE", "/workspace")))
    data_dir:  Path = field(default_factory=lambda: Path(os.environ.get("AGENT_DATA", "/workspace/swarm/autonomous/data")))

    @property
    def memory_db(self) -> Path:
        return self.data_dir / "memory.sqlite"

    @property
    def checkpoint_db(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    @property
    def skill_db(self) -> Path:
        return self.data_dir / "skills.sqlite"

    @property
    def audit_log(self) -> Path:
        return self.data_dir / "audit.jsonl"


@dataclass(frozen=True)
class Safety:
    """Command whitelisting + human-approval gates."""
    # Always-allowed shell commands (prefixes). Anything not on this list prompts the human.
    shell_allowlist: tuple[str, ...] = (
        "ls", "cat", "head", "tail", "wc", "file", "find", "pwd", "echo",
        "grep", "rg", "ast-grep", "tree",
        "python", "python3", "pytest", "pip show", "pip list",
        "git status", "git diff", "git log", "git show", "git branch",
        "flutter analyze", "flutter test",
        "node -v", "npm ls", "npm test",
        "curl -s", "curl -sS", "curl -I",
    )
    # Explicit block list — these ALWAYS require human approval even if a prefix matches allowlist.
    shell_denylist: tuple[str, ...] = (
        "rm ", "rm\t", "rm-", "rmdir", "mv /", "dd ",
        "git push", "git reset --hard", "git clean", "git branch -D",
        "sudo", "su ", "chmod -R", "chown -R",
        "kill ", "pkill", "killall",
        "curl -X DELETE", "curl -X PUT", "curl -X POST",
        ":(){", "mkfs", "> /dev",
    )
    # Write ops (write_file, etc.) always require confirmation unless --yolo
    require_approval_for_writes: bool = True
    # If True, just log + proceed; if False (default), bail when approval denied
    approve_default: bool = False


CONFIG = {
    "fleet": Fleet(),
    "budget": Budget(),
    "paths": Paths(),
    "safety": Safety(),
}
