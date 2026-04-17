"""Anvil invocation wrapper for benchmark runs.

Responsibilities:
  - per-task workspace isolation (AGENT_WORKSPACE + AGENT_DATA env vars)
  - subprocess timeout
  - capture wall-clock + stdout/stderr
  - parallel orchestration (thread pool, bounded by --workers)
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

from .adapter import BenchmarkAdapter, Task, Prediction


@dataclass
class RunResult:
    task_id: str
    success: bool
    wall_sec: float
    cycles_used: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: str | None = None


class AnvilRunner:
    """Runs Anvil in one-shot mode against an isolated workspace.

    Each call spawns `python -m autonomous.main --yolo <goal>` in a subprocess
    with AGENT_WORKSPACE and AGENT_DATA pointing into a task-specific directory,
    so nothing leaks between tasks.
    """

    def __init__(
        self,
        anvil_root: Path,
        sandbox_root: Path,
        max_cycles: int = 5,
        timeout_sec: int = 600,
        python_exe: str = sys.executable,
        extra_env: dict[str, str] | None = None,
        shared_memory: bool = True,
    ) -> None:
        self.anvil_root = anvil_root.resolve()
        self.sandbox_root = sandbox_root.resolve()
        self.max_cycles = max_cycles
        self.timeout_sec = timeout_sec
        self.python_exe = python_exe
        self.extra_env = extra_env or {}
        self.sandbox_root.mkdir(parents=True, exist_ok=True)

        # Shared memory pool: cross-task lesson propagation within a batch.
        # Each task has its own AGENT_DATA (isolation), but all tasks in the
        # batch share AGENT_SHARED_DATA so lessons from completed tasks show
        # up on siblings' next reflection cycle.
        if shared_memory:
            self.shared_dir = self.sandbox_root / "_shared_memory"
            self.shared_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.shared_dir = None

    def task_workspace(self, task_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in task_id)
        return self.sandbox_root / safe

    def run(self, adapter: BenchmarkAdapter, task: Task) -> tuple[RunResult, Path]:
        workspace = self.task_workspace(task.id)
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True)
        data_dir = workspace / ".anvil"
        data_dir.mkdir()

        adapter.seed_workspace(task, workspace)

        env = os.environ.copy()
        env["AGENT_WORKSPACE"] = str(workspace)
        env["AGENT_DATA"] = str(data_dir)
        env["AGENT_YOLO"] = "1"
        env["PYTHONPATH"] = f"{self.anvil_root}{os.pathsep}{env.get('PYTHONPATH', '')}"
        if self.shared_dir is not None:
            env["AGENT_SHARED_DATA"] = str(self.shared_dir)
        env.update(self.extra_env)

        # autonomous.main is one-shot — runs one graph traversal. The max_cycles
        # concept lives in the daemon; for benchmarks we cap work via timeout
        # and per-subgoal turn limits (configured in env before main runs).
        env["EXECUTOR_MAX_TURNS"] = str(max(6, self.max_cycles * 3))
        cmd = [
            self.python_exe, "-m", "autonomous.main",
            "--yolo",
            task.goal,
        ]

        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(workspace),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                check=False,
            )
            wall = time.time() - t0
            return RunResult(
                task_id=task.id,
                success=proc.returncode == 0,
                wall_sec=wall,
                stdout_tail=proc.stdout[-4000:] if proc.stdout else "",
                stderr_tail=proc.stderr[-2000:] if proc.stderr else "",
                cycles_used=_count_cycles(proc.stdout),
                error=None if proc.returncode == 0 else f"exit {proc.returncode}",
            ), workspace
        except subprocess.TimeoutExpired as e:
            wall = time.time() - t0
            return RunResult(
                task_id=task.id,
                success=False,
                wall_sec=wall,
                stdout_tail=(e.stdout or "")[-4000:] if e.stdout else "",
                stderr_tail=(e.stderr or "")[-2000:] if e.stderr else "",
                error=f"TIMEOUT after {self.timeout_sec}s",
            ), workspace
        except Exception as e:
            wall = time.time() - t0
            return RunResult(task_id=task.id, success=False, wall_sec=wall, error=str(e)), workspace

    def run_many(
        self,
        adapter: BenchmarkAdapter,
        tasks: list[Task],
        workers: int = 1,
        on_complete: Callable[[RunResult, Prediction], None] | None = None,
    ) -> list[tuple[RunResult, Prediction]]:
        """Run a batch of tasks, returning (RunResult, Prediction) for each."""
        results: list[tuple[RunResult, Prediction]] = []

        def _one(task: Task) -> tuple[RunResult, Prediction]:
            run_res, workspace = self.run(adapter, task)
            stats = {
                "wall_sec": run_res.wall_sec,
                "cycles_used": run_res.cycles_used,
                "success": run_res.success,
            }
            try:
                pred = adapter.extract_prediction(task, workspace, stats)
            except Exception as e:
                pred = Prediction(task_id=task.id, prediction={}, error=f"extract failed: {e}")
            pred.anvil_stats = stats
            return run_res, pred

        if workers <= 1:
            for t in tasks:
                pair = _one(t)
                results.append(pair)
                if on_complete:
                    on_complete(*pair)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_one, t): t for t in tasks}
                for fut in as_completed(futures):
                    pair = fut.result()
                    results.append(pair)
                    if on_complete:
                        on_complete(*pair)
        return results


def _count_cycles(stdout: str) -> int | None:
    if not stdout:
        return None
    return stdout.count("[daemon] CYCLE ") or stdout.count("[cycle] ")
