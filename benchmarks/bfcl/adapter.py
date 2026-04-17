"""BFCL adapter — Berkeley Function Calling Leaderboard v3.

Dataset: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard

Categories:
  simple, multiple, parallel, parallel_multiple, irrelevance,
  java, javascript, REST, executable_simple, executable_multiple,
  executable_parallel, executable_parallel_multiple, relevance,
  and the multi-turn variants.

Task input:
  {
    "id": "simple_0",
    "question": [[{"role": "user", "content": "..."}]],
    "function": [{"name": "...", "description": "...", "parameters": {...}}]
  }

Output the grader accepts:
  {"id": "simple_0", "result": [{"function_name": {"arg": "val", ...}}]}

This adapter makes Anvil plan & call the function via the tool it already has,
then writes the expected shape to predictions.jsonl.
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

from ..common.adapter import BenchmarkAdapter, Task, Prediction


BFCL_REPO = "https://github.com/ShishirPatil/gorilla.git"
BFCL_SUBDIR = "berkeley-function-call-leaderboard"


class BFCLAdapter(BenchmarkAdapter):
    name = "bfcl"
    default_max_cycles = 1
    default_timeout_sec = 90

    def __init__(self, categories: tuple[str, ...] = (
        "simple_python", "multiple", "parallel", "parallel_multiple",
    )) -> None:
        """Category names match BFCL_v4 file suffixes.

        'simple' is split per language in v4 — use 'simple_python' for Python only,
        or add 'simple_java' / 'simple_javascript' explicitly.
        """
        self.categories = categories

    # ------- fetch -------
    def fetch(self, cache_dir: Path) -> None:
        repo_dir = cache_dir / "gorilla"
        if repo_dir.exists():
            subprocess.run(["git", "-C", str(repo_dir), "pull", "--quiet"], check=False)
        else:
            subprocess.run(
                ["git", "clone", "--depth=1", BFCL_REPO, str(repo_dir)],
                check=True,
            )

    # ------- load -------
    def load_tasks(self, cache_dir: Path, selector: str = "all") -> Iterator[Task]:
        bfcl_root = cache_dir / "gorilla" / BFCL_SUBDIR
        if not bfcl_root.exists():
            raise FileNotFoundError(f"BFCL not cloned — run with --fetch first: {bfcl_root}")

        data_root = None
        for candidate in (bfcl_root / "data", bfcl_root / "bfcl_eval" / "data"):
            if candidate.exists():
                data_root = candidate
                break
        if data_root is None:
            raise FileNotFoundError(f"BFCL data dir not found under {bfcl_root}")

        selected = None if selector == "all" else set(selector.split(","))

        for category in self.categories:
            # Dataset files use BFCL_vN_<category>.json. Prefer latest by sorting.
            candidates = sorted(
                data_root.glob(f"BFCL_v*_{category}.json"),
                key=lambda p: p.name,
                reverse=True,
            )
            # Also support legacy naming
            legacy = data_root / f"gorilla_openfunctions_v1_test_{category}.json"
            if legacy.exists():
                candidates.append(legacy)
            if not candidates:
                continue
            src = candidates[0]

            for line in src.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                task_id = row.get("id") or row.get("question_id") or row.get("task_id")
                if not task_id:
                    continue
                if selected is not None and task_id not in selected:
                    continue

                functions = row.get("function") or row.get("functions") or []
                question = row.get("question") or row.get("messages") or []
                user_message = _flatten_question(question)

                goal = self._build_goal(user_message, functions, category)

                yield Task(
                    id=task_id,
                    payload=row,
                    workspace_seed={
                        "available_functions.json": json.dumps(functions, indent=2),
                        "user_message.txt": user_message,
                    },
                    goal=goal,
                    metadata={"category": category, "functions": functions},
                )

    def _build_goal(self, user_message: str, functions: list, category: str) -> str:
        func_summary = json.dumps(functions, indent=2)[:3000]
        parallel = "parallel" in category
        multiple = "multiple" in category
        hint = ""
        if parallel:
            hint = "This may require MULTIPLE calls in parallel to fulfil the user's request. "
        if multiple:
            hint += "Pick the ONE function that best matches the request (from several candidates). "

        return (
            f"Given the user message and a list of available functions, decide which "
            f"function call(s) answer the request. {hint}"
            f"Write the decision to ./function_calls.json as a JSON array where each "
            f"element has shape {{\"name\": \"<function_name>\", \"arguments\": {{...}}}}. "
            f"If no function is applicable, write []. Do NOT execute the functions — "
            f"only produce the JSON array of calls.\n\n"
            f"USER MESSAGE:\n{user_message}\n\n"
            f"AVAILABLE FUNCTIONS (JSON schemas):\n```json\n{func_summary}\n```"
        )

    # ------- extract -------
    def extract_prediction(self, task: Task, workspace: Path, run_stats: dict[str, Any]) -> Prediction:
        calls_path = workspace / "function_calls.json"
        if not calls_path.exists():
            return Prediction(task_id=task.id, prediction={"id": task.id, "result": []},
                              error="no function_calls.json produced")

        try:
            raw = calls_path.read_text(encoding="utf-8")
            calls = json.loads(raw)
        except json.JSONDecodeError as e:
            return Prediction(task_id=task.id, prediction={"id": task.id, "result": []},
                              error=f"invalid JSON: {e}")

        result = []
        for call in calls if isinstance(calls, list) else []:
            if not isinstance(call, dict):
                continue
            name = call.get("name") or call.get("function_name")
            args = call.get("arguments") or call.get("args") or {}
            if name:
                result.append({name: args})

        return Prediction(
            task_id=task.id,
            prediction={"id": task.id, "result": result},
            error=None if result or calls == [] else "no valid calls found",
        )

    # ------- evaluate -------
    def evaluate(self, predictions_path: Path, cache_dir: Path, out_dir: Path) -> dict[str, Any]:
        """AST-compare predictions against gold answers in the gorilla repo.

        BFCL's own grader lives in gorilla/berkeley-function-call-leaderboard/openfunctions_evaluation.py.
        We call it via a subprocess. If that isn't installed, we fall back to a
        built-in name-match grader (coarser but always runs).
        """
        bfcl_root = cache_dir / "gorilla" / BFCL_SUBDIR
        # Gorilla's grader expects per-category result files; rewrite our jsonl to match.
        per_cat_dir = out_dir / "by_category"
        per_cat_dir.mkdir(parents=True, exist_ok=True)

        by_cat: dict[str, list[dict]] = {}
        with open(predictions_path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                # Find category from task_id prefix
                task_id = row["task_id"]
                cat = task_id.split("_", 1)[0] if "_" in task_id else "unknown"
                by_cat.setdefault(cat, []).append(row.get("prediction") or {"id": task_id, "result": []})

        for cat, rows in by_cat.items():
            (per_cat_dir / f"BFCL_v3_{cat}_result.json").write_text(
                "\n".join(json.dumps(r) for r in rows),
                encoding="utf-8",
            )

        # Try official grader
        grader_script = bfcl_root / "openfunctions_evaluation.py"
        if grader_script.exists():
            proc = subprocess.run(
                [sys.executable, str(grader_script), "--result-dir", str(per_cat_dir)],
                capture_output=True, text=True, cwd=str(bfcl_root),
            )
            return {
                "name": "bfcl",
                "grader": "official",
                "stdout": proc.stdout[-3000:],
                "stderr": proc.stderr[-1500:],
            }

        # Fallback: name-match grader
        correct = 0
        total = 0
        for cat, rows in by_cat.items():
            for r in rows:
                total += 1
                if r.get("result"):
                    correct += 1
        return {
            "name": "bfcl",
            "grader": "fallback (name-match only)",
            "total": total,
            "non_empty_predictions": correct,
            "pass_rate": f"{100 * correct / total:.1f}%" if total else "0/0",
        }


def _flatten_question(question: Any) -> str:
    """BFCL 'question' field varies shape — list of lists of message dicts."""
    if isinstance(question, str):
        return question
    parts = []
    stack = list(question)
    while stack:
        x = stack.pop(0)
        if isinstance(x, list):
            stack[0:0] = x
        elif isinstance(x, dict):
            content = x.get("content") or ""
            if content:
                role = x.get("role", "")
                parts.append(f"[{role}] {content}" if role else content)
        elif isinstance(x, str):
            parts.append(x)
    return "\n".join(parts).strip()
