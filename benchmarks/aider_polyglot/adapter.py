"""Aider Polyglot adapter — 225 Exercism problems across 6 languages.

Dataset: https://github.com/Aider-AI/polyglot-benchmark
Each problem is a directory with:
  - instructions (the task description)
  - example-solution files (starter code)
  - test file(s) (what must pass)
  - `.meta/config.json` naming solution/test files

Task shape per language:
  - C++:  solution in .cpp, tests compile + run
  - Go:   solution in .go,  tests via `go test`
  - Java: solution in .java, tests via `gradle test` or `mvn`
  - JS:   solution in .js,   tests via `npm test`
  - Python: solution in .py, tests via `pytest`
  - Rust:  solution in .rs,  tests via `cargo test`

Evaluation requires each language's toolchain to be installed — we use a Docker
image that has all six, so the evaluator is portable.
"""
from __future__ import annotations
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

from ..common.adapter import BenchmarkAdapter, Task, Prediction


POLYGLOT_REPO = "https://github.com/Aider-AI/polyglot-benchmark.git"


class AiderPolyglotAdapter(BenchmarkAdapter):
    name = "aider_polyglot"
    default_max_cycles = 3
    default_timeout_sec = 300

    LANG_TO_EXT = {
        "cpp":    ".cpp",
        "go":     ".go",
        "java":   ".java",
        "javascript": ".js",
        "python": ".py",
        "rust":   ".rs",
    }

    LANG_TEST_CMD = {
        "cpp":    ["bash", "-c", "cmake -S . -B build -G Ninja && cmake --build build && ./build/tests"],
        "go":     ["go", "test", "./..."],
        "java":   ["./gradlew", "test", "--no-daemon"],
        "javascript": ["bash", "-c", "npm install --no-audit --silent && npm test --silent"],
        "python": [sys.executable, "-m", "pytest", "-q"],
        "rust":   ["cargo", "test", "--offline"],
    }

    # ------- fetch -------
    def fetch(self, cache_dir: Path) -> None:
        repo = cache_dir / "polyglot-benchmark"
        if repo.exists():
            subprocess.run(["git", "-C", str(repo), "pull", "--quiet"], check=False)
        else:
            subprocess.run(["git", "clone", "--depth=1", POLYGLOT_REPO, str(repo)], check=True)

    # ------- load -------
    def load_tasks(self, cache_dir: Path, selector: str = "all") -> Iterator[Task]:
        root = cache_dir / "polyglot-benchmark"
        if not root.exists():
            raise FileNotFoundError(f"polyglot-benchmark not cloned — run with --fetch first")

        selected = None if selector == "all" else set(selector.split(","))

        # Each language has one subdir; problems sit under <lang>/exercises/practice/<problem>/
        for lang_dir in sorted(root.iterdir()):
            if not lang_dir.is_dir() or lang_dir.name.startswith("."):
                continue
            lang = lang_dir.name
            exercises_root = lang_dir / "exercises" / "practice"
            if not exercises_root.exists():
                continue

            for problem_dir in sorted(exercises_root.iterdir()):
                if not problem_dir.is_dir():
                    continue
                task_id = f"{lang}/{problem_dir.name}"
                if selected is not None and task_id not in selected:
                    continue

                cfg_path = problem_dir / ".meta" / "config.json"
                if not cfg_path.exists():
                    continue

                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                files = cfg.get("files", {})
                solution_files = files.get("solution", []) or []
                test_files = files.get("test", []) or []
                instructions = _read_instructions(problem_dir)

                # Build seed: copy every file the agent needs to see
                seed = {}
                for rel in solution_files + test_files:
                    src = problem_dir / rel
                    if src.exists():
                        seed[rel] = src.read_text(encoding="utf-8", errors="replace")
                # Preserve ancillary files (Cargo.toml, package.json, etc.) so tests can run
                for extra in _ancillary_files(problem_dir, lang):
                    rel = extra.relative_to(problem_dir).as_posix()
                    seed[rel] = extra.read_text(encoding="utf-8", errors="replace")

                goal = (
                    f"Implement the {lang} solution for this Exercism problem. Only edit the "
                    f"solution file(s) listed below; do NOT change the test files. When you are "
                    f"done, the tests must pass.\n\n"
                    f"SOLUTION FILES (edit these):\n  " + "\n  ".join(solution_files) + "\n\n"
                    f"TEST FILES (do not modify):\n  " + "\n  ".join(test_files) + "\n\n"
                    f"INSTRUCTIONS:\n{instructions[:4000]}"
                )

                yield Task(
                    id=task_id,
                    payload={"config": cfg, "lang": lang, "problem": problem_dir.name},
                    workspace_seed=seed,
                    goal=goal,
                    metadata={
                        "lang": lang,
                        "solution_files": solution_files,
                        "test_files": test_files,
                        "problem_dir": str(problem_dir),
                    },
                )

    # ------- extract -------
    def extract_prediction(self, task: Task, workspace: Path, run_stats: dict[str, Any]) -> Prediction:
        sol_files = task.metadata["solution_files"]
        solutions = {}
        for rel in sol_files:
            p = workspace / rel
            if p.exists():
                solutions[rel] = p.read_text(encoding="utf-8", errors="replace")

        return Prediction(
            task_id=task.id,
            prediction={
                "task_id": task.id,
                "lang": task.metadata["lang"],
                "solutions": solutions,
            },
            error=None if solutions else "no solution files found in workspace",
        )

    # ------- evaluate -------
    def evaluate(self, predictions_path: Path, cache_dir: Path, out_dir: Path) -> dict[str, Any]:
        root = cache_dir / "polyglot-benchmark"
        results = []
        passed = 0
        total = 0

        for line in Path(predictions_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            pred = row.get("prediction") or {}
            task_id = pred.get("task_id") or row["task_id"]
            lang = pred.get("lang") or task_id.split("/")[0]
            problem = task_id.split("/", 1)[-1]
            solutions = pred.get("solutions", {}) or {}

            if not solutions:
                results.append({"task_id": task_id, "passed": False, "reason": "no solution"})
                total += 1
                continue

            # Reproduce a fresh workspace from the upstream exercise, overlay solutions, run tests
            problem_src = root / lang / "exercises" / "practice" / problem
            if not problem_src.exists():
                results.append({"task_id": task_id, "passed": False, "reason": "problem dir missing"})
                total += 1
                continue

            ws = out_dir / "eval_workspaces" / task_id
            if ws.exists():
                shutil.rmtree(ws)
            shutil.copytree(problem_src, ws)
            for rel, content in solutions.items():
                target = ws / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")

            cmd = self.LANG_TEST_CMD.get(lang)
            if not cmd:
                results.append({"task_id": task_id, "passed": False, "reason": f"no runner for {lang}"})
                total += 1
                continue

            try:
                proc = subprocess.run(cmd, cwd=str(ws), capture_output=True, text=True, timeout=180)
                ok = proc.returncode == 0
            except Exception as e:
                ok = False
                proc = None

            results.append({
                "task_id": task_id,
                "passed": ok,
                "exit_code": proc.returncode if proc else None,
                "stderr_tail": (proc.stderr[-500:] if proc and proc.stderr else ""),
            })
            total += 1
            if ok:
                passed += 1

        # Per-lang summary
        per_lang: dict[str, dict[str, int]] = {}
        for r in results:
            lang = r["task_id"].split("/")[0]
            per_lang.setdefault(lang, {"total": 0, "passed": 0})
            per_lang[lang]["total"] += 1
            if r["passed"]:
                per_lang[lang]["passed"] += 1

        (out_dir / "per_task_results.json").write_text(json.dumps(results, indent=2))
        return {
            "name": "aider_polyglot",
            "total": total,
            "passed": passed,
            "pass_rate": f"{100 * passed / total:.1f}%" if total else "0/0",
            "per_lang": per_lang,
        }


# ---- helpers ----

def _read_instructions(problem_dir: Path) -> str:
    docs = problem_dir / ".docs"
    parts = []
    for name in ("introduction.md", "instructions.md", "instructions.append.md", "hints.md"):
        p = docs / name
        if p.exists():
            parts.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n\n".join(parts).strip() or "(no instructions)"


def _ancillary_files(problem_dir: Path, lang: str) -> list[Path]:
    """Files outside solution/test that are needed for the test runner."""
    ancillary = []
    if lang == "rust":
        for p in ["Cargo.toml", "Cargo.lock"]:
            f = problem_dir / p
            if f.exists():
                ancillary.append(f)
    elif lang == "javascript":
        for p in ["package.json", "package-lock.json", ".eslintrc.cjs", "babel.config.cjs"]:
            f = problem_dir / p
            if f.exists():
                ancillary.append(f)
    elif lang == "java":
        for p in ["build.gradle", "settings.gradle", "gradlew", "gradlew.bat"]:
            f = problem_dir / p
            if f.exists():
                ancillary.append(f)
    elif lang == "go":
        for p in ["go.mod", "go.sum"]:
            f = problem_dir / p
            if f.exists():
                ancillary.append(f)
    elif lang == "cpp":
        for p in ["CMakeLists.txt"]:
            f = problem_dir / p
            if f.exists():
                ancillary.append(f)
    return ancillary
