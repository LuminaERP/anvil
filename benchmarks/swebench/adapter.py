"""SWE-bench adapter — Verified / Lite / Multimodal.

Dataset: HF `princeton-nlp/SWE-bench_Verified` (500 tasks), `SWE-bench_Lite` (300),
or the full 2,294 instances. Each task:

  instance_id         e.g. 'astropy__astropy-7746'
  repo                e.g. 'astropy/astropy'
  base_commit         SHA to check out
  problem_statement   the GitHub issue body
  hints_text          comments from the issue thread
  test_patch          hidden — what tests must pass/change
  FAIL_TO_PASS        hidden — test IDs that start failing and must pass
  PASS_TO_PASS        hidden — tests that must keep passing
  version             repo version tag

Anvil gets: repo, base_commit, problem_statement (+ hints). Produces: unified
diff that resolves the issue.

Grader: `pip install swebench`, then `swebench.harness.run_evaluation` spins up
a per-task Docker, applies the patch, runs the hidden test suite, compares
pre-patch vs post-patch test outcomes.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

from ..common.adapter import BenchmarkAdapter, Task, Prediction


DATASETS = {
    "lite":       "princeton-nlp/SWE-bench_Lite",
    "verified":   "princeton-nlp/SWE-bench_Verified",
    "full":       "princeton-nlp/SWE-bench",
}


class SWEBenchAdapter(BenchmarkAdapter):
    name = "swebench"
    default_max_cycles = 8
    default_timeout_sec = 1800  # 30 min per issue — SWE-bench tasks are slow

    def __init__(self, split: str = "lite") -> None:
        """split: 'lite', 'verified', or 'full'."""
        assert split in DATASETS, f"unknown split: {split}"
        self.split = split
        self.dataset_id = DATASETS[split]

    # ------- fetch -------
    def fetch(self, cache_dir: Path) -> None:
        out = cache_dir / f"swebench_{self.split}.jsonl"
        if out.exists():
            return
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:
            raise RuntimeError("pip install datasets swebench") from e
        ds = load_dataset(self.dataset_id, split="test")
        with open(out, "w", encoding="utf-8") as fh:
            for row in ds:
                fh.write(json.dumps(dict(row), default=str) + "\n")

    # ------- load -------
    def load_tasks(self, cache_dir: Path, selector: str = "all") -> Iterator[Task]:
        src = cache_dir / f"swebench_{self.split}.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"run --fetch first: {src}")

        selected = None if selector == "all" else set(selector.split(","))

        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            iid = row["instance_id"]
            if selected is not None and iid not in selected:
                continue

            repo = row["repo"]
            base = row["base_commit"]
            statement = row["problem_statement"]
            hints = row.get("hints_text", "") or ""

            goal = (
                f"Fix the GitHub issue below by editing files in this repo. The repo is "
                f"already cloned in the current workspace at commit {base[:12]}.\n\n"
                f"REPOSITORY: {repo}\n"
                f"COMMIT: {base}\n\n"
                f"ISSUE:\n{statement}\n\n"
                + (f"THREAD HINTS:\n{hints[:2000]}\n\n" if hints else "")
                + "WORKFLOW:\n"
                "1. Read the repo structure with list_dir + grep to find the relevant files\n"
                "2. Read the current implementation — never guess\n"
                "3. Apply the MINIMAL change that resolves the issue\n"
                "4. Do not change unrelated code. Do not add new tests unless the issue asks for it.\n"
                "5. When done, run the test suite to verify no regressions.\n"
            )

            yield Task(
                id=iid,
                payload=row,
                workspace_seed={},  # workspace seeded below via git clone
                goal=goal,
                metadata={
                    "repo": repo,
                    "base_commit": base,
                    "version": row.get("version"),
                    "FAIL_TO_PASS": row.get("FAIL_TO_PASS"),
                    "PASS_TO_PASS": row.get("PASS_TO_PASS"),
                },
            )

    # ------- seed workspace -------
    def seed_workspace(self, task: Task, workspace: Path) -> None:
        """Clone the repo at the correct commit into workspace/."""
        workspace.mkdir(parents=True, exist_ok=True)
        repo = task.metadata["repo"]
        base = task.metadata["base_commit"]

        # Use git clone + checkout; supports --reference caches later if desired
        subprocess.run(
            ["git", "clone", f"https://github.com/{repo}.git", str(workspace)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(workspace), "checkout", base],
            check=True, capture_output=True,
        )

        # Mark the commit so we can diff cleanly
        subprocess.run(
            ["git", "-C", str(workspace), "tag", "-f", "anvil-base", base],
            check=False, capture_output=True,
        )

    # ------- extract -------
    def extract_prediction(self, task: Task, workspace: Path, run_stats: dict[str, Any]) -> Prediction:
        # Produce unified diff of any changes Anvil made
        result = subprocess.run(
            ["git", "-C", str(workspace), "diff", "anvil-base", "HEAD", "--"],
            capture_output=True, text=True,
        )

        # Also include untracked files as diffs against /dev/null
        untracked_proc = subprocess.run(
            ["git", "-C", str(workspace), "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True,
        )
        untracked = untracked_proc.stdout.strip().splitlines()
        untracked_diff_parts = []
        for rel in untracked:
            full = workspace / rel
            if full.is_file() and full.stat().st_size < 200_000:
                try:
                    content = full.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                untracked_diff_parts.append(
                    f"diff --git a/{rel} b/{rel}\n"
                    f"new file mode 100644\n"
                    f"--- /dev/null\n"
                    f"+++ b/{rel}\n"
                    + "".join(f"+{ln}\n" for ln in content.splitlines())
                )

        patch = result.stdout + "\n".join(untracked_diff_parts)

        return Prediction(
            task_id=task.id,
            prediction={
                "instance_id": task.id,
                "model_patch": patch,
                "model_name_or_path": "anvil-v0.1",
            },
            error=None if patch.strip() else "no diff produced (no changes)",
        )

    # ------- evaluate -------
    def evaluate(self, predictions_path: Path, cache_dir: Path, out_dir: Path) -> dict[str, Any]:
        # Reshape jsonl to a JSON list of predictions (what the SWE-bench harness expects)
        flat_predictions = []
        with open(predictions_path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                pred = row.get("prediction") or {}
                if pred.get("instance_id") and pred.get("model_patch"):
                    flat_predictions.append({
                        "instance_id": pred["instance_id"],
                        "model_patch": pred["model_patch"],
                        "model_name_or_path": pred.get("model_name_or_path", "anvil-v0.1"),
                    })

        flat_path = out_dir / "swebench_predictions.json"
        flat_path.write_text(json.dumps(flat_predictions, indent=2), encoding="utf-8")

        # Run the official harness
        proc = subprocess.run(
            [
                sys.executable, "-m", "swebench.harness.run_evaluation",
                "--dataset_name", self.dataset_id,
                "--predictions_path", str(flat_path),
                "--max_workers", os.environ.get("SWEBENCH_WORKERS", "4"),
                "--run_id", f"anvil-{self.split}",
            ],
            capture_output=True, text=True,
        )

        # Harness writes summary JSON alongside — try to find it
        summary_files = list(Path.cwd().glob(f"anvil-{self.split}*/*.json"))
        resolved = 0
        total = len(flat_predictions)
        if summary_files:
            for sf in summary_files:
                try:
                    data = json.loads(sf.read_text())
                    if "resolved_instances" in data:
                        resolved = len(data["resolved_instances"])
                        break
                except (json.JSONDecodeError, KeyError):
                    continue

        return {
            "name": f"swebench_{self.split}",
            "total_predictions": total,
            "resolved": resolved,
            "pass_rate": f"{100 * resolved / total:.1f}%" if total else "0/0",
            "grader_stdout_tail": proc.stdout[-3000:],
            "grader_stderr_tail": proc.stderr[-1500:],
            "returncode": proc.returncode,
        }
