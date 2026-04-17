"""BigCodeBench adapter.

Dataset: HF `bigcode/bigcodebench`. 1,140 Python tasks that exercise real
libraries (pandas, numpy, requests, regex, crypto, etc.). Each task:

  task_id, complete_prompt, instruct_prompt, canonical_solution, code_prompt,
  test (hidden), entry_point, doc_struct, libs

Two sub-tasks per benchmark:
  - "complete": function-signature-driven completion (like HumanEval)
  - "instruct": natural-language instruction → full function

We produce both kinds of predictions and let the grader score.

Grader: `pip install bigcodebench`, then `bigcodebench.evaluate` runs the hidden
tests inside a Docker sandbox.
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

from ..common.adapter import BenchmarkAdapter, Task, Prediction


DATASET_ID = "bigcode/bigcodebench"


class BigCodeBenchAdapter(BenchmarkAdapter):
    name = "bigcodebench"
    default_max_cycles = 4
    default_timeout_sec = 300

    def __init__(self, subset: str = "complete") -> None:
        """subset: 'complete' or 'instruct' — which prompt style to use."""
        assert subset in ("complete", "instruct")
        self.subset = subset

    # ------- fetch -------
    def fetch(self, cache_dir: Path) -> None:
        out = cache_dir / "bigcodebench.jsonl"
        if out.exists():
            return
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:
            raise RuntimeError("pip install datasets") from e
        ds = load_dataset(DATASET_ID, split="v0.1.4")
        with open(out, "w", encoding="utf-8") as fh:
            for row in ds:
                fh.write(json.dumps(dict(row), default=str) + "\n")

    # ------- load -------
    def load_tasks(self, cache_dir: Path, selector: str = "all") -> Iterator[Task]:
        src = cache_dir / "bigcodebench.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"run --fetch first: {src}")

        selected = None if selector == "all" else set(selector.split(","))
        prompt_key = "complete_prompt" if self.subset == "complete" else "instruct_prompt"

        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = row["task_id"]
            if selected is not None and task_id not in selected:
                continue

            prompt = row.get(prompt_key) or row.get("complete_prompt") or ""
            entry_point = row.get("entry_point", "")

            if self.subset == "complete":
                goal = (
                    f"Complete the Python function. Write the FULL function (signature + body) "
                    f"to ./solution.py. Do not add examples, __main__ blocks, or prints. "
                    f"Imports stay at the top. The entry point is `{entry_point}`.\n\n"
                    f"```python\n{prompt}\n```"
                )
            else:
                goal = (
                    f"Implement the requested Python function. Write the entire function "
                    f"(signature + body + any necessary imports) to ./solution.py. The entry "
                    f"point is `{entry_point}`. Include only what the instruction asks for — "
                    f"no example usage, no prints.\n\n"
                    f"INSTRUCTION:\n{prompt}"
                )

            yield Task(
                id=task_id,
                payload=row,
                workspace_seed={"problem.py": prompt},
                goal=goal,
                metadata={"entry_point": entry_point, "subset": self.subset},
            )

    # ------- extract -------
    def extract_prediction(self, task: Task, workspace: Path, run_stats: dict[str, Any]) -> Prediction:
        sol = workspace / "solution.py"
        if not sol.exists():
            return Prediction(task_id=task.id, prediction={"task_id": task.id, "solution": ""},
                              error="no solution.py")

        content = sol.read_text(encoding="utf-8", errors="replace")
        return Prediction(
            task_id=task.id,
            prediction={"task_id": task.id, "solution": content},
            error=None if content.strip() else "empty solution",
        )

    # ------- evaluate -------
    def evaluate(self, predictions_path: Path, cache_dir: Path, out_dir: Path) -> dict[str, Any]:
        samples_path = out_dir / "samples.jsonl"
        with open(predictions_path, encoding="utf-8") as src, \
             open(samples_path, "w", encoding="utf-8") as dst:
            for line in src:
                if not line.strip():
                    continue
                row = json.loads(line)
                pred = row.get("prediction") or {}
                dst.write(json.dumps({
                    "task_id": pred.get("task_id") or row["task_id"],
                    "solution": pred.get("solution", ""),
                }) + "\n")

        # Official grader — runs in Docker
        proc = subprocess.run(
            [
                sys.executable, "-m", "bigcodebench.evaluate",
                "--split", "complete" if self.subset == "complete" else "instruct",
                "--subset", "full",
                "--samples", str(samples_path),
            ],
            capture_output=True, text=True,
        )
        return {
            "name": f"bigcodebench_{self.subset}",
            "grader_stdout_tail": proc.stdout[-3000:],
            "grader_stderr_tail": proc.stderr[-1500:],
            "returncode": proc.returncode,
        }
