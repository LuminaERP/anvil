"""LiveCodeBench adapter.

Dataset: HF `livecodebench/code_generation_lite` — competitive-programming tasks
scraped monthly from LeetCode, AtCoder, Codeforces. Contamination-resistant
because new problems drop every month.

Each problem:
  question_id, question_content, platform, question_title, difficulty,
  starter_code, public_test_cases, private_test_cases (base64-encoded pickle),
  metadata, contest_date

Prediction format the grader wants:
  {"question_id": "...", "code_list": ["<solution code>"]}

We call the upstream grader as a subprocess to compute pass rates.
"""
from __future__ import annotations
import base64
import json
import pickle
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any, Iterator

from ..common.adapter import BenchmarkAdapter, Task, Prediction


DATASET_ID = "livecodebench/code_generation_lite"
LCB_REPO = "https://github.com/LiveCodeBench/LiveCodeBench.git"


class LiveCodeBenchAdapter(BenchmarkAdapter):
    name = "livecodebench"
    default_max_cycles = 4
    default_timeout_sec = 300

    # Map friendly names to the JSONL files on HF
    RELEASE_FILES = {
        "release_v1": "test.jsonl",
        "release_v2": "test2.jsonl",
        "release_v3": "test3.jsonl",
        "release_v4": "test4.jsonl",
        "release_v5": "test5.jsonl",
        "release_v6": "test6.jsonl",
        "release_latest": "test6.jsonl",
    }

    def __init__(self, release: str = "release_latest") -> None:
        if release not in self.RELEASE_FILES:
            raise ValueError(
                f"unknown release {release!r}; valid: {list(self.RELEASE_FILES)}"
            )
        self.release = release

    # ------- fetch -------
    def fetch(self, cache_dir: Path) -> None:
        out = cache_dir / "livecodebench.jsonl"
        if not out.exists():
            # LiveCodeBench's HF dataset uses a loading script; recent versions of
            # `datasets` disable those by default. Prefer direct parquet download.
            self._fetch_via_parquet(cache_dir, out)

        # Also clone the grader repo for evaluation
        grader = cache_dir / "LiveCodeBench"
        if not grader.exists():
            subprocess.run(["git", "clone", "--depth=1", LCB_REPO, str(grader)], check=False)

    def _fetch_via_parquet(self, cache_dir: Path, out: Path) -> None:
        """Download the release's JSONL file directly from HF hub."""
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except ImportError as e:
            raise RuntimeError("pip install huggingface_hub") from e

        filename = self.RELEASE_FILES[self.release]
        local = hf_hub_download(repo_id=DATASET_ID, filename=filename, repo_type="dataset")
        # The LCB JSONL is already in our target format; copy it.
        import shutil
        shutil.copy(local, out)

    # ------- load -------
    def load_tasks(self, cache_dir: Path, selector: str = "all") -> Iterator[Task]:
        src = cache_dir / "livecodebench.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"run --fetch first: {src}")

        selected = None if selector == "all" else set(selector.split(","))

        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row["question_id"]
            if selected is not None and qid not in selected:
                continue

            content = row.get("question_content", "")
            starter = row.get("starter_code", "") or ""
            # Public test cases are small + visible; they help the agent self-check
            public_tests = row.get("public_test_cases") or "[]"
            try:
                public_tests_parsed = json.loads(public_tests) if isinstance(public_tests, str) else public_tests
            except json.JSONDecodeError:
                public_tests_parsed = []
            examples = _format_public_tests(public_tests_parsed)

            # Infer task style from public tests. stdin → AtCoder/Codeforces style
            # (read input, print output); functional → LeetCode class Solution style.
            testtype = "stdin"
            if public_tests_parsed:
                testtype = public_tests_parsed[0].get("testtype", "stdin")

            verify_nudge = (
                "\n\nSELF-VERIFICATION (mandatory before declaring done):\n"
                "After writing solution.py, call `run_public_tests(solution_path='solution.py')` "
                "to verify against the public test cases (already saved to ./public_tests.json by "
                "the harness). If any public test fails, fix the solution and re-verify. Only "
                "declare done when ALL public tests pass."
            )

            if testtype == "functional":
                goal = (
                    f"Solve this LeetCode-style problem in Python. Write the full "
                    f"solution to ./solution.py — define `class Solution` with the method "
                    f"signature from the starter code. Do not add a __main__ block, stdin "
                    f"reading, print statements, or any code outside the class. Your "
                    f"solution will be graded by instantiating Solution() and calling the method.\n\n"
                    f"PROBLEM:\n{content}\n\n"
                    + (f"STARTER CODE:\n```python\n{starter}\n```\n\n" if starter else "")
                    + (f"PUBLIC EXAMPLES:\n{examples}\n" if examples else "")
                    + verify_nudge
                )
            else:
                goal = (
                    f"Solve this competitive-programming problem in Python. Write the full "
                    f"solution to ./solution.py as a TOP-LEVEL script that reads input from "
                    f"stdin (via `input()` or `sys.stdin.read()`) and prints the answer(s) to "
                    f"stdout. Do NOT define a class or a function you forget to call — the "
                    f"grader runs the script and compares printed output to the expected "
                    f"output. Do not add extra prints, debug output, or trailing characters.\n\n"
                    f"PROBLEM:\n{content}\n\n"
                    + (f"PUBLIC EXAMPLES (input → expected output):\n{examples}\n" if examples else "")
                    + verify_nudge
                )

            # Seed the public tests so the agent can call run_public_tests
            seed_files = {
                "problem.md": content,
                "public_tests.json": json.dumps(public_tests_parsed, indent=2),
            }
            if starter:
                seed_files["starter.py"] = starter

            yield Task(
                id=qid,
                payload=row,
                workspace_seed=seed_files,
                goal=goal,
                metadata={
                    "platform": row.get("platform"),
                    "difficulty": row.get("difficulty"),
                    "contest_date": row.get("contest_date"),
                },
            )

    # ------- extract -------
    def extract_prediction(self, task: Task, workspace: Path, run_stats: dict[str, Any]) -> Prediction:
        sol = workspace / "solution.py"
        if not sol.exists():
            return Prediction(task_id=task.id, prediction={"question_id": task.id, "code_list": [""]},
                              error="no solution.py")

        code = sol.read_text(encoding="utf-8", errors="replace")
        return Prediction(
            task_id=task.id,
            prediction={"question_id": task.id, "code_list": [code]},
            error=None if code.strip() else "empty solution",
        )

    # ------- evaluate -------
    def evaluate(self, predictions_path: Path, cache_dir: Path, out_dir: Path) -> dict[str, Any]:
        grader = cache_dir / "LiveCodeBench"
        if not grader.exists():
            return {"error": "LiveCodeBench grader repo not cloned"}

        # LiveCodeBench's runner expects a flat predictions list
        samples_path = out_dir / "samples.json"
        rows = []
        with open(predictions_path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                pred = row.get("prediction") or {}
                rows.append({
                    "question_id": pred.get("question_id") or row["task_id"],
                    "code_list": pred.get("code_list", [""]),
                })
        samples_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

        # Grader invocation
        proc = subprocess.run(
            [
                sys.executable, "-m", "lcb_runner.runner.main",
                "--model", "anvil",
                "--release_version", self.release,
                "--custom_output_file", str(samples_path),
                "--evaluate",
            ],
            capture_output=True, text=True, cwd=str(grader),
        )
        return {
            "name": "livecodebench",
            "grader_stdout_tail": proc.stdout[-3000:],
            "grader_stderr_tail": proc.stderr[-1500:],
            "returncode": proc.returncode,
        }


def _format_public_tests(public_tests: list) -> str:
    """Public tests are [(input, output), ...] or [{"input":..., "output":...}, ...]."""
    if not public_tests:
        return ""
    lines = []
    for i, t in enumerate(public_tests[:3]):
        if isinstance(t, dict):
            inp = t.get("input", "")
            out = t.get("output", "")
        else:
            inp = t[0] if len(t) > 0 else ""
            out = t[1] if len(t) > 1 else ""
        lines.append(f"Example {i + 1}:\n  Input:  {inp}\n  Output: {out}")
    return "\n\n".join(lines)
