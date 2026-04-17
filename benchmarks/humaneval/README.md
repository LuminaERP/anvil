# HumanEval

164 single-function Python problems. Each problem provides a function signature + docstring; Anvil must produce a correct body. Scored by running OpenAI's reference test suite.

## Install grader

```bash
pip install human-eval
```

## Run

```bash
# Single smoke test
python -m benchmarks.humaneval.run --fetch
python -m benchmarks.humaneval.run --predict --tasks "HumanEval/0,HumanEval/1" --workers 1

# Full suite
python -m benchmarks.humaneval.run --end-to-end --workers 8

# Evaluate only
python -m benchmarks.humaneval.run --evaluate
```

## Expected output

- Predictions: `benchmark_output/humaneval/predictions.jsonl`
- Grader input: `benchmark_output/humaneval/samples.jsonl`
- Grader output: `benchmark_output/humaneval/samples.jsonl_results.jsonl`
- Summary: `benchmark_output/humaneval/summary.json`

## Scoring

`human_eval.evaluate_functional_correctness` is the official grader. It runs the stored
test assertions for each problem against `prompt + completion` and reports `pass@1`.

## Notes

- `pass@k` with k > 1 requires generating k samples per problem. This adapter is single-sample (pass@1). To run pass@10, submit 10 predictions per task with the same task_id — the grader aggregates.
- The adapter's `_extract_body` isolates just the function body from whatever the agent wrote, so adding `def ...:` in the solution file doesn't cause a duplicate-signature error.
- HumanEval is **largely saturated** — most frontier models score >90%. It's still useful as a baseline smoke test that Anvil's basic code-writing loop works.
