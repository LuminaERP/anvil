"""Base class every benchmark adapter inherits from.

An adapter knows:
  1. How to load its dataset (local files or HuggingFace)
  2. How to translate a task row into an Anvil goal + workspace layout
  3. How to extract the prediction (patch/code/tool-call) once Anvil is done
  4. What format the grader expects

The runner below takes an adapter instance and does the plumbing.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class Task:
    """Benchmark task, normalised across suites."""

    id: str
    payload: dict[str, Any]         # raw row from the dataset
    workspace_seed: dict[str, str] = field(default_factory=dict)  # filename -> contents
    goal: str = ""                  # prompt Anvil will receive
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    """Anvil's output for one task, in the shape the benchmark's grader expects."""

    task_id: str
    prediction: dict[str, Any]     # grader-specific payload
    anvil_stats: dict[str, Any] = field(default_factory=dict)  # cycles, turns, tokens, etc.
    error: str | None = None


class BenchmarkAdapter(ABC):
    """Subclasses implement one method per phase."""

    name: str = "<unnamed>"
    default_max_cycles: int = 5
    default_timeout_sec: int = 600

    # ---- phase 1: dataset ----
    @abstractmethod
    def fetch(self, cache_dir: Path) -> None:
        """Download / clone the dataset into cache_dir. Idempotent."""

    @abstractmethod
    def load_tasks(self, cache_dir: Path, selector: str = "all") -> Iterator[Task]:
        """Yield Task objects. `selector` is either 'all' or a comma-separated ID list."""

    # ---- phase 2: prepare per-task workspace ----
    def seed_workspace(self, task: Task, workspace: Path) -> None:
        """Populate task.workspace_seed files into workspace/. Default impl covers the simple case."""
        workspace.mkdir(parents=True, exist_ok=True)
        for rel_path, contents in task.workspace_seed.items():
            target = workspace / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(contents, encoding="utf-8")

    # ---- phase 3: extract prediction ----
    @abstractmethod
    def extract_prediction(self, task: Task, workspace: Path, run_stats: dict[str, Any]) -> Prediction:
        """Read final workspace state and translate into grader format."""

    # ---- phase 4: scoring ----
    @abstractmethod
    def evaluate(self, predictions_path: Path, cache_dir: Path, out_dir: Path) -> dict[str, Any]:
        """Run the benchmark's official grader. Return summary dict (pass_rate, per-task results)."""
