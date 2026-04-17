"""HumanEval adapter.

Dataset: https://github.com/openai/human-eval — 164 problems, each has a Python
function signature + docstring, you produce the body, their harness runs hidden
tests against it.

Input row:
  {
    "task_id": "HumanEval/0",
    "prompt": "def has_close_elements(...):\n    ...",
    "entry_point": "has_close_elements",
    "canonical_solution": "<reference>",
    "test": "<test harness>"
  }

Output expected by grader (one line per task):
  {"task_id": "HumanEval/0", "completion": "<just the body, no signature>"}
"""
from __future__ import annotations
import json
import gzip
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

from ..common.adapter import BenchmarkAdapter, Task, Prediction


HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"


class HumanEvalAdapter(BenchmarkAdapter):
    name = "humaneval"
    default_max_cycles = 3
    default_timeout_sec = 180

    # ------- fetch -------
    def fetch(self, cache_dir: Path) -> None:
        jsonl_path = cache_dir / "HumanEval.jsonl"
        gz_path = cache_dir / "HumanEval.jsonl.gz"
        if jsonl_path.exists():
            return
        _download(HUMANEVAL_URL, gz_path)
        with gzip.open(gz_path, "rb") as src, open(jsonl_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        gz_path.unlink()

    # ------- load -------
    def load_tasks(self, cache_dir: Path, selector: str = "all") -> Iterator[Task]:
        jsonl_path = cache_dir / "HumanEval.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(f"dataset not fetched — run with --fetch first: {jsonl_path}")

        selected = None if selector == "all" else set(selector.split(","))
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                task_id = row["task_id"]
                if selected is not None and task_id not in selected:
                    continue

                prompt = row["prompt"]
                entry_point = row["entry_point"]

                goal = (
                    f"Complete the Python function `{entry_point}` described by the "
                    f"docstring below. Write the implementation to ./solution.py. "
                    f"The file must contain the ENTIRE function (signature + body) so "
                    f"it imports cleanly. Do not write any other code — no examples, "
                    f"no __main__ block, no print statements. Match the docstring exactly.\n\n"
                    f"```python\n{prompt}\n```"
                )

                yield Task(
                    id=task_id,
                    payload=row,
                    workspace_seed={"problem.py": prompt},
                    goal=goal,
                    metadata={"entry_point": entry_point},
                )

    # ------- extract -------
    def extract_prediction(self, task: Task, workspace: Path, run_stats: dict[str, Any]) -> Prediction:
        sol_path = workspace / "solution.py"
        if not sol_path.exists():
            return Prediction(task_id=task.id, prediction={"completion": ""},
                              error="no solution.py produced")

        content = sol_path.read_text(encoding="utf-8", errors="replace")
        completion = _extract_body(content, task.payload["prompt"], task.metadata["entry_point"])

        return Prediction(
            task_id=task.id,
            prediction={"task_id": task.id, "completion": completion},
            error=None if completion else "empty completion",
        )

    # ------- evaluate -------
    def evaluate(self, predictions_path: Path, cache_dir: Path, out_dir: Path) -> dict[str, Any]:
        """Use the `human-eval` pip package to run reference tests."""
        samples_path = out_dir / "samples.jsonl"
        _convert_predictions_to_samples(predictions_path, samples_path)

        # Official grader
        proc = subprocess.run(
            [sys.executable, "-m", "human_eval.evaluate_functional_correctness", str(samples_path)],
            capture_output=True, text=True, cwd=str(out_dir),
        )
        if proc.returncode != 0:
            return {
                "error": "grader failed",
                "stdout": proc.stdout[-2000:],
                "stderr": proc.stderr[-2000:],
            }

        # Grader writes <samples>.jsonl_results.jsonl alongside input
        results_path = Path(str(samples_path) + "_results.jsonl")
        passed = 0
        total = 0
        per_task = {}
        if results_path.exists():
            for line in results_path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                total += 1
                if row.get("passed"):
                    passed += 1
                per_task[row["task_id"]] = {"passed": row.get("passed"), "result": row.get("result", "")}

        return {
            "name": "humaneval",
            "total": total,
            "passed": passed,
            "pass_rate": f"{100 * passed / total:.1f}%" if total else "0/0",
            "grader_stdout": proc.stdout[-500:],
            "per_task_sample": dict(list(per_task.items())[:5]),
        }


# ---- helpers ----

def _download(url: str, dest: Path) -> None:
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)


def _extract_body(solution: str, prompt: str, entry_point: str) -> str:
    """Isolate just the body (below the signature) from whatever the agent wrote.

    The grader concatenates prompt + completion, so completion must be just the
    body — not another copy of the signature. We try hard to handle agents that
    produced either pattern.
    """
    # If the solution already starts with a full function (signature + body), strip
    # everything up to and including the signature line, return the body.
    signature_line = f"def {entry_point}("
    idx = solution.find(signature_line)
    if idx >= 0:
        # Find end of signature (first line ending with ':')
        after_sig = solution[idx:]
        colon = after_sig.find(":\n")
        if colon > 0:
            body_plus = after_sig[colon + 2:]
            # Body only — stop before any second top-level definition
            body_lines = []
            for line in body_plus.split("\n"):
                stripped = line.lstrip()
                # Hitting a top-level def/class/etc means we're past this function
                if line and not line.startswith((" ", "\t")) and stripped and not stripped.startswith("#"):
                    break
                body_lines.append(line)
            return "\n".join(body_lines).rstrip() + "\n"

    # Fallback: assume the whole file IS the body
    return solution


def _convert_predictions_to_samples(predictions_path: Path, samples_path: Path) -> None:
    """Shape the jsonl so `human_eval.evaluate_functional_correctness` accepts it."""
    with open(predictions_path, encoding="utf-8") as src, \
         open(samples_path, "w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            pred = row.get("prediction") or {}
            task_id = pred.get("task_id") or row["task_id"]
            completion = pred.get("completion", "")
            dst.write(json.dumps({"task_id": task_id, "completion": completion}) + "\n")
