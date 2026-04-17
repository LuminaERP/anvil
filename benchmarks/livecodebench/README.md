# LiveCodeBench

Competitive-programming problems scraped monthly from LeetCode, AtCoder, and Codeforces. The "lite" split is ~600 problems; the full benchmark grows over time. Contamination-resistant — new problems drop every release, so models trained before a cutoff cannot have memorised them.

## Install grader

```bash
pip install datasets
# grader cloned automatically during --fetch into ~/.cache/anvil_benchmarks/livecodebench/LiveCodeBench
cd ~/.cache/anvil_benchmarks/livecodebench/LiveCodeBench
pip install -r requirements.txt
```

## Run

```bash
python -m benchmarks.livecodebench.run --fetch
python -m benchmarks.livecodebench.run --predict --workers 8 --timeout-sec 300
python -m benchmarks.livecodebench.run --evaluate
```

## Release selection

LiveCodeBench publishes a new `release_v{N}` every month. To pin to a specific release, construct the adapter with `LiveCodeBenchAdapter(release="release_v2")` instead of the default latest.

## Scoring

`lcb_runner` in the LiveCodeBench repo runs each solution against the public + private test cases. Metric: pass@1.

## Notes

- Most problems are under 50 lines of code; `max_cycles=4` is ample.
- The adapter includes up to 3 public examples in the prompt so Anvil can self-check.
- Starter code (when the problem provides a class/method stub) is passed through.
- Leaderboard: https://livecodebench.github.io/leaderboard.html
