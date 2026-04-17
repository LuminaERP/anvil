# Aider Polyglot

225 Exercism problems across 6 languages (C++, Go, Java, JavaScript, Python, Rust). Anvil must edit the solution file so all tests pass.

## Toolchain requirements

To evaluate predictions, each language's toolchain must be installed:

- **C++**: cmake, ninja, a C++ compiler
- **Go**: Go 1.21+
- **Java**: JDK 17+ and `./gradlew` wrapper
- **JS**: Node 20+
- **Python**: pytest
- **Rust**: cargo + rustc

Easiest on Linux:

```bash
apt-get install -y cmake ninja-build golang-go openjdk-17-jdk nodejs npm
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
pip install pytest
```

Or run evaluation inside a Docker image that bundles all six — see the Anvil repo's `benchmarks/aider_polyglot/Dockerfile`.

## Run

```bash
python -m benchmarks.aider_polyglot.run --fetch
python -m benchmarks.aider_polyglot.run --predict --workers 4 --timeout-sec 300
python -m benchmarks.aider_polyglot.run --evaluate
```

Restrict to one language for quick smoke:

```bash
python -m benchmarks.aider_polyglot.run --predict --tasks python/leap,python/hamming --workers 1
```

## Scoring

For each task the adapter:

1. Copies the upstream problem directory to `eval_workspaces/<task_id>/`
2. Overlays Anvil's solution file
3. Runs the language's test command (`pytest`, `go test`, `cargo test`, etc.)
4. Records pass / fail based on exit code

Summary JSON includes per-language pass rates.

## Notes

- **Dataset is canonical**: same 225 problems Aider uses to publish their public score, so results are directly comparable.
- `max_cycles=3` is a good default; complex problems like Rust lifetime puzzles may need 5.
- Network access during evaluation is needed for `npm install` and `cargo` crate fetches — offline runners must pre-warm caches.
- Aider's leaderboard: https://aider.chat/docs/leaderboards/
