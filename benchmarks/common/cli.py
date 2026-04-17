"""Shared CLI helpers for all benchmark adapters."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Any

from .adapter import BenchmarkAdapter
from .runner import AnvilRunner
from .results import ResultsWriter, ResultsReader


def build_parser(name: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=f"benchmarks.{name}.run")

    phases = p.add_mutually_exclusive_group(required=True)
    phases.add_argument("--fetch", action="store_true", help="Download/clone the dataset into the cache dir")
    phases.add_argument("--predict", action="store_true", help="Run Anvil over tasks, emit predictions.jsonl")
    phases.add_argument("--evaluate", action="store_true", help="Run the benchmark's grader on predictions.jsonl")
    phases.add_argument("--end-to-end", action="store_true", help="Fetch + predict + evaluate in one go")
    phases.add_argument("--list", action="store_true", help="List available task IDs and exit")

    p.add_argument("--tasks", default="all", help="'all' or comma-separated task IDs")
    p.add_argument("--workers", type=int, default=4, help="parallel workers (default 4)")
    p.add_argument("--max-cycles", type=int, default=None, help="cap Anvil cycles per task (adapter default)")
    p.add_argument("--timeout-sec", type=int, default=None, help="per-task wall-clock cap (adapter default)")
    p.add_argument("--anvil-root", default="/workspace/swarm",
                   help="Anvil source root; must contain autonomous/ package")
    p.add_argument("--cache-dir", default=None, help="dataset cache (default ~/.cache/anvil_benchmarks/<name>)")
    p.add_argument("--out", default=None, help="output dir (default benchmark_output/<name>/)")
    p.add_argument("--limit", type=int, default=None, help="cap on number of tasks (after selector)")
    p.add_argument("--resume", action="store_true",
                   help="skip tasks already in predictions.jsonl")

    return p


def _paths(name: str, args: argparse.Namespace) -> tuple[Path, Path]:
    cache = Path(args.cache_dir) if args.cache_dir else Path.home() / ".cache" / "anvil_benchmarks" / name
    out = Path(args.out) if args.out else Path("benchmark_output") / name
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    return cache, out


def dispatch(name: str, adapter: BenchmarkAdapter, argv: list[str] | None = None) -> int:
    parser = build_parser(name)
    args = parser.parse_args(argv)
    cache_dir, out_dir = _paths(name, args)
    predictions_path = out_dir / "predictions.jsonl"

    if args.list:
        for task in adapter.load_tasks(cache_dir, "all"):
            print(task.id)
        return 0

    if args.fetch or args.end_to_end:
        print(f"[{name}] fetching dataset into {cache_dir}")
        adapter.fetch(cache_dir)
        print(f"[{name}] fetch complete")

    if args.predict or args.end_to_end:
        tasks = list(adapter.load_tasks(cache_dir, args.tasks))
        if args.limit:
            tasks = tasks[: args.limit]

        done_ids: set[str] = set()
        if args.resume and predictions_path.exists():
            done_ids = ResultsReader(predictions_path).task_ids()
            print(f"[{name}] resume: skipping {len(done_ids)} already-done tasks")
        tasks = [t for t in tasks if t.id not in done_ids]

        if not tasks:
            print(f"[{name}] no tasks to run")
        else:
            print(f"[{name}] running {len(tasks)} tasks, workers={args.workers}")
            runner = AnvilRunner(
                anvil_root=Path(args.anvil_root),
                sandbox_root=Path.home() / ".cache" / "anvil_benchmarks" / name / "sandbox",
                max_cycles=args.max_cycles or adapter.default_max_cycles,
                timeout_sec=args.timeout_sec or adapter.default_timeout_sec,
            )

            done = 0
            total = len(tasks)
            with ResultsWriter(predictions_path) as writer:
                def _on_done(run, pred):
                    nonlocal done
                    done += 1
                    status = "OK " if run.success else "ERR"
                    print(f"  [{done:>4}/{total}] {status} {run.task_id}  ({run.wall_sec:.1f}s)")
                    writer.write(run, pred)

                runner.run_many(adapter, tasks, workers=args.workers, on_complete=_on_done)
            print(f"[{name}] predictions written to {predictions_path}")

    if args.evaluate or args.end_to_end:
        if not predictions_path.exists():
            print(f"[{name}] no predictions file at {predictions_path}", file=sys.stderr)
            return 1
        print(f"[{name}] evaluating")
        summary = adapter.evaluate(predictions_path, cache_dir, out_dir)
        import json
        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        print(f"[{name}] summary: {summary.get('pass_rate', '?')}  ({summary_path})")

    return 0
