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
                    f"docstring below. Write the implementation to the file named "
                    f"exactly `solution.py` in the current working directory (NOT "
                    f"/workspace/solution.py — use the relative path `solution.py`). "
                    f"The file must contain the ENTIRE function (signature + body), plus "
                    f"any imports the implementation needs (at the top of the file), so "
                    f"it imports cleanly. Do not write any other code — no examples, "
                    f"no __main__ block, no print statements.\n\n"
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
        entry_point = task.metadata["entry_point"]
        sol_path = _locate_solution(workspace, entry_point)
        if sol_path is None:
            return Prediction(task_id=task.id, prediction={"task_id": task.id, "completion": ""},
                              error="no solution.py produced (checked workspace + *.py)")

        content = sol_path.read_text(encoding="utf-8", errors="replace")
        completion = _extract_body(content, task.payload["prompt"], entry_point)

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

        # Official grader — use absolute path so cwd doesn't matter
        samples_abs = samples_path.resolve()
        proc = subprocess.run(
            [sys.executable, "-m", "human_eval.evaluate_functional_correctness", str(samples_abs)],
            capture_output=True, text=True, cwd=str(out_dir.resolve()),
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
    """Isolate completion text that, concatenated after the prompt, produces a
    valid Python program.

    The HumanEval grader executes `prompt + completion`. The prompt already
    contains the function signature + docstring. We want the completion to be:

      - any imports the solution added at module level (converted to imports
        INSIDE the function so they land under the signature when concatenated)
      - the function body (lines below the `def ...:` line)

    We stop before any second top-level `def`/`class` to exclude extra helpers
    the model may have added below.
    """
    # Extract module-level imports (lines starting with 'import' or 'from')
    # that appear ABOVE the target signature line. These would be lost if we
    # only took the body, so we re-emit them as indented imports inside the
    # function.
    signature_line = f"def {entry_point}("
    idx = solution.find(signature_line)

    imports_to_inject: list[str] = []
    if idx > 0:
        preamble = solution[:idx]
        for line in preamble.split("\n"):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                imports_to_inject.append("    " + stripped)

    if idx >= 0:
        after_sig = solution[idx:]
        colon = after_sig.find(":\n")
        if colon > 0:
            body_plus = after_sig[colon + 2:]
            body_lines = []
            for line in body_plus.split("\n"):
                stripped = line.lstrip()
                # Hitting a top-level def/class/etc means we're past this function
                if line and not line.startswith((" ", "\t")) and stripped and not stripped.startswith("#"):
                    break
                body_lines.append(line)
            body = "\n".join(body_lines).rstrip() + "\n"
            if imports_to_inject:
                body = "\n".join(imports_to_inject) + "\n" + body
            return body

    # Fallback: assume the whole file IS the body
    return solution


def _locate_solution(workspace: Path, entry_point: str) -> Path | None:
    """Find the solution file across common places the agent might write to.

    1. <workspace>/solution.py                (the intended location)
    2. <workspace>/*.py containing `def <entry_point>(`   (any stray filename)
    3. /workspace/solution.py                 (legacy location — only as fallback
       when the task workspace has nothing; race-prone but better than nothing)
    """
    primary = workspace / "solution.py"
    if primary.exists():
        return primary

    # Scan for any .py file in workspace that defines the entry point
    for candidate in workspace.glob("*.py"):
        try:
            text = candidate.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if f"def {entry_point}(" in text:
            return candidate

    return None


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
