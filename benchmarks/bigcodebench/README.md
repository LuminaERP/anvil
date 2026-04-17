# BigCodeBench

1,140 Python tasks that exercise real libraries (pandas, numpy, requests, cryptography, regex, etc.). Two prompt styles:

- `complete` — function signature + docstring → body (like HumanEval, but harder)
- `instruct` — natural-language instruction → full function

## Install grader

```bash
pip install bigcodebench datasets
# Docker required for sandboxed execution
```

## Run

```bash
python -m benchmarks.bigcodebench.run --fetch
python -m benchmarks.bigcodebench.run --predict --workers 8 --timeout-sec 300
python -m benchmarks.bigcodebench.run --evaluate
```

## Switching subset

Edit the adapter instantiation in `run.py` to use `subset="instruct"` for the instruct variant.

## Scoring

Two metrics reported by the grader:

- **pass@1** — percentage of tasks where the generated code passes the hidden test suite
- **pass@1 (calibrated)** — same but with stricter scoring: we also check the calibration tests

## Notes

- Runs in Docker — each task's test runs in an isolated container so `rm -rf /` in a task solution can't hurt anything.
- 1,140 tasks × ~15s each = ~2 hrs at 8 parallel workers.
- Leaderboard: https://bigcode-bench.github.io/
- **We default to the `complete` subset** for faster iteration; `instruct` is the "agent" variant and generally scores lower.
