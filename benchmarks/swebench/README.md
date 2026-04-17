# SWE-bench

Real GitHub issues from 12 popular Python projects (Django, sympy, scikit-learn, matplotlib, astropy, etc.). Each task gives Anvil a repo at a specific commit + the issue text. Anvil must produce a patch that makes the hidden test suite go from failing to passing without breaking any currently-passing tests.

This is the industry flagship for coding agents.

## Splits

- **SWE-bench Lite** (300) — hand-filtered to remove tasks requiring deep repo knowledge. ~3 hours at 8 parallel.
- **SWE-bench Verified** (500) — OpenAI-curated subset with cleaner specs. ~6 hours at 8 parallel.
- **SWE-bench** (2,294) — the full set. ~24 hours at 8 parallel.

## Install grader

```bash
pip install swebench datasets
docker pull ghcr.io/swebench/sweb.eval.x86_64.latest  # grader's base image
```

You need Docker running; the grader spins up a clean container per task.

## Run

```bash
# Smoke test (one task)
python -m benchmarks.swebench.run --split=lite --fetch
python -m benchmarks.swebench.run --split=lite --predict --tasks "astropy__astropy-7746" --workers 1 --timeout-sec 1800

# Full Lite
python -m benchmarks.swebench.run --split=lite --end-to-end --workers 8

# Verified
python -m benchmarks.swebench.run --split=verified --end-to-end --workers 8
```

Control the grader's Docker concurrency via `SWEBENCH_WORKERS=N` env var.

## How each task runs

1. Adapter clones the target repo into `/tmp/anvil_bench/swebench_lite/<task_id>/`
2. Checks out the base commit; tags it `anvil-base`
3. Invokes Anvil with the issue as goal
4. After Anvil exits, `git diff anvil-base HEAD` captures the patch
5. Untracked files are synthesised into the diff as well
6. Predictions get rewritten into `swebench_predictions.json`
7. `swebench.harness.run_evaluation` spins up per-task Docker containers, applies the patch, runs the hidden test suite

## Scoring

Primary metric: **% resolved** = tasks where `FAIL_TO_PASS` tests now pass AND `PASS_TO_PASS` tests still pass.

## Cost + time

- **Lite**: 300 tasks × ~30-60 s agent time + ~30-60 s grader time = 5-10 min wall-clock per task. 3 hrs at 8 parallel.
- **Verified**: same per-task cost; 5-10 hrs total.
- **Storage**: ~20 GB of Docker images + per-task repo clones (tasks reuse clones if you set `GIT_CACHE_DIR`).

## Notes

- The default `max_cycles=8` — SWE-bench issues often need multiple read/edit iterations.
- The default `timeout_sec=1800` — some tasks have slow test suites (sympy especially).
- Parallel workers here is **agent parallelism** (Anvil subprocesses); grader has its own `--max_workers` for Docker parallelism.
- Leaderboard: https://www.swebench.com/
- SWE-bench Live (continuously-updated, contamination-resistant) dataset is also available: `princeton-nlp/SWE-bench_Live` — same adapter works with `split="live"` once `DATASETS` is extended.
