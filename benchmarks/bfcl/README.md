# BFCL (Berkeley Function Calling Leaderboard)

Measures the specific skill of *picking the right function with the right arguments* from a list of available tools. Each task gives Anvil a user message + JSON schemas for 1-N functions; Anvil must produce the correct call(s).

## Install grader

```bash
git clone https://github.com/ShishirPatil/gorilla.git ~/.cache/anvil_benchmarks/bfcl/gorilla
pip install -r ~/.cache/anvil_benchmarks/bfcl/gorilla/berkeley-function-call-leaderboard/requirements.txt
```

## Run

```bash
python -m benchmarks.bfcl.run --fetch
python -m benchmarks.bfcl.run --predict --workers 8
python -m benchmarks.bfcl.run --evaluate
```

## Categories

Default categories (most common): `simple`, `multiple`, `parallel`, `parallel_multiple`.
Add `--categories relevance,irrelevance,java,javascript,REST` to the adapter construction
to cover more.

## Scoring

AST-comparison against gold answers. The adapter invokes `openfunctions_evaluation.py` from
gorilla's repo. If that's missing, a fallback name-match grader runs (coarser).

## Notes

- Anvil runs each task with `max_cycles=1` since BFCL is single-shot — no need for long ReAct loops.
- BFCL-v3 expanded with multi-turn categories. Those work with this adapter but benefit from higher `max_cycles`.
- Leaderboard: https://gorilla.cs.berkeley.edu/leaderboard.html
