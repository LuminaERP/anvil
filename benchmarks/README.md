# Anvil benchmarks

Adapters that let Anvil run against industry-standard agent evaluation suites. Each adapter translates the benchmark's task format into an Anvil goal, invokes the agent, and extracts the prediction in the shape the benchmark's grader expects.

## Supported benchmarks

| Suite | Tasks | Scope | Runtime (8 parallel) | Status |
|-------|-------|-------|----------------------|--------|
| [HumanEval](humaneval/) | 164 | Single-function Python completion | ~10 min | ready |
| [BFCL v3](bfcl/) | ~2,000 | Tool / function calling | ~20 min | ready |
| [Aider Polyglot](aider_polyglot/) | 225 | Multi-language Exercism edits | ~30 min | ready |
| [BigCodeBench](bigcodebench/) | 1,140 | Complex Python with diverse libraries | ~2 hr | ready |
| [LiveCodeBench](livecodebench/) | ~600 | Contamination-resistant competitive Python | ~1 hr | ready |
| [SWE-bench Lite](swebench/) | 300 | Real GitHub issue resolution | ~3 hr | ready |

## Layout

```
benchmarks/
├── common/                 # Shared infrastructure
│   ├── adapter.py          # BenchmarkAdapter base class
│   ├── runner.py           # Anvil invocation + isolation
│   ├── results.py          # JSONL result writer
│   └── cli.py              # Shared CLI flags
├── humaneval/              # Each benchmark:
│   ├── adapter.py          #   adapter logic
│   ├── run.py              #   CLI entrypoint
│   └── README.md           #   usage + scoring
└── ...
```

## Running any benchmark

```bash
# Fetch dataset (one-time)
python -m benchmarks.humaneval.run --fetch

# Full suite
python -m benchmarks.humaneval.run --predict --tasks all --workers 8

# Evaluate
python -m benchmarks.humaneval.run --evaluate --predictions out/humaneval/predictions.jsonl

# All-in-one
python -m benchmarks.humaneval.run --end-to-end --workers 8
```

All adapters obey the same flags:
- `--fetch` — download dataset
- `--predict` — run Anvil on tasks, emit predictions
- `--evaluate` — score predictions against expected outputs
- `--end-to-end` — fetch + predict + evaluate
- `--tasks ID[,ID,...]` or `--tasks all` — task selection
- `--workers N` — parallelism (per-benchmark limits apply)
- `--max-cycles N` — cap on Anvil cycles per task (default 5)
- `--timeout-sec N` — per-task wall-clock cap (default 600)
- `--out DIR` — output directory (default `benchmark_output/<name>/`)

## Isolation model

Each task runs in its own sandboxed workspace at `/tmp/anvil_bench/<suite>/<task_id>/`:

- `AGENT_WORKSPACE=<task_dir>` — Anvil cannot read/write outside this dir
- `AGENT_DATA=<task_dir>/.anvil` — memory / skills / checkpoints are per-task (no cross-contamination)
- Post-run: diff extracted, workspace optionally torn down

This means a task's lessons don't leak into its siblings — purity matters for academic rigor.

## Scoring

Each benchmark emits `predictions.jsonl` in the format its official grader expects. Scoring is then:

- **HumanEval**: `human-eval` pip package runs reference tests
- **BFCL**: gorilla-cli AST-compare against gold function calls
- **Aider Polyglot**: per-language test runners inside language toolchain Docker
- **BigCodeBench**: `bigcodebench` pip package runs hidden tests in sandbox
- **LiveCodeBench**: repo-provided evaluator runs public + private test cases
- **SWE-bench**: `swebench` pip package clones repo, applies patch, runs hidden test suite in Docker

## Fleet requirements

Anvil runs on the local vLLM fleet (`:8000`-`:8003`). For benchmark runs we recommend:

- Supervisor / planner: Qwen 2.5 14B AWQ or better
- Coder / executor: Qwen3-Coder 30B AWQ (largest context helps on SWE-bench)
- Max cycles: 5 per task default; bump to 10 for SWE-bench
- Timeout: 600 s per task default; bump to 1800 s for SWE-bench

## Cost expectations

At current tokens-per-task (rough, measured on our 10-cycle runs):

| Suite | ~tokens/task | Local fleet | Claude Sonnet 4.6 API |
|-------|--------------|-------------|-----------------------|
| HumanEval | 3 K | ~free | ~$2 |
| BFCL | 2 K | ~free | ~$10 |
| Aider Polyglot | 8 K | ~free | ~$4 |
| BigCodeBench | 15 K | ~free | ~$40 |
| LiveCodeBench | 12 K | ~free | ~$20 |
| SWE-bench Lite | 30 K | ~free | ~$30 |

The local fleet makes this basically free once the GPU is up. Without one, SWE-bench Verified + BigCodeBench via API will run >$100.
