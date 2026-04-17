<p align="center">
  <img src="assets/anvil-logo.png" alt="Anvil — Autonomous Coding System" width="480">
</p>

<p align="center">
  <em>Autonomous multi-agent coding system with neural safety, semantic memory, and local fleet inference.</em>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="#benchmarks"><img src="https://img.shields.io/badge/HumanEval-87.2%25-brightgreen.svg" alt="HumanEval 87.2%"></a>
</p>

---


Anvil is a long-running autonomous agent that plans, implements, reviews, and learns. It reads before it writes, grounds every claim in tool output, and refuses to ship code that fails static analysis or regresses existing tests. It runs against a local vLLM fleet, so a single overnight session can chew through fifty-plus backlog items for the cost of the electricity to keep the GPU warm.

---

## What it does

Hand it a backlog, tell it how many cycles, go to sleep. Each cycle:

1. **Plan** — a 14 B supervisor decomposes the current goal into subgoals, grounded in a deterministic AST inventory of the workspace so it cannot invent function names or files.
2. **Execute** — a 30 B coder runs a ReAct loop with 54 tools (filesystem, shell, git, research, and optionally PSKit for Windows-style ops) until the subgoal is done or the turn cap is hit.
3. **Review** — a 7 B reviewer inspects the diff for correctness, style, and missing edge cases.
4. **Reflect** — the supervisor decides: done, continue, or stuck-replan. A completeness gate overrides "done" when the goal mentions a target count and the count has not been met.
5. **Learn** — lessons ("don't use single-port args with port_status") and skills ("fetch_models_data: fan out to N vLLM endpoints in parallel") persist across cycles and sessions.

Every `.py` write goes through two hard gates: ruff `F821/F822/F823` and an existing-test regression check. Anything that introduces an undefined name or breaks a green test gets auto-rolled back on disk.

---

## Architecture

```
┌─────────────── Supervisor (Qwen 2.5 14B AWQ, :8000) ───────────────┐
│   Plans, reflects, proposes next goal when backlog is empty.       │
└─────────────────────────────────────────────────────────────────────┘
          │                                         ▲
          ▼                                         │
┌─── Coder (Qwen3-Coder 30B AWQ, :8001) ───┐        │
│   Executes ReAct with 54 tools.           ├────────┘
│   18-turn cap per subgoal.                │
└────────────────────────────────────────────┘
          │
          ▼
┌─── Reviewer (Qwen 2.5 7B AWQ, :8002) ───┐
│   Diff review, style, edge-case audit.   │
└──────────────────────────────────────────┘
          │
          ▼
┌─── Worker (Qwen 2.5 3B, :8003) ──────┐
│   Parallelisable scut-work subgoals.  │
└───────────────────────────────────────┘
```

All four endpoints are OpenAI-compatible (FastAPI / vLLM). Any model can be swapped for any other model served at that URL; nothing in the code is hard-wired to Qwen.

---

## Tool suite (54 tools when PSKit is enabled)

**Native (21):**
`read_file`, `write_file`, `edit_file`, `apply_patch`, `list_dir`, `glob_files`, `grep`, `run_bash`, `run_pytest`, `python_eval`, `git_status`, `git_diff`, `git_log`, `git_show`, `git_blame`, `web_search`, `web_fetch`, `context7_resolve`, `context7_docs`, `nia_package_search`, `nia_search`.

**PSKit (33 additional)** via MCP stdio bridge — opt-in with `PSKIT_ENABLED=1`:
`pskit_read_file`, `pskit_edit_file`, `pskit_run_command`, `pskit_memory_usage`, `pskit_disk_usage`, `pskit_port_status`, `pskit_process_info`, `pskit_gpu_status`, `pskit_http_request`, `pskit_which`, `pskit_install_package`, full `pskit_git_*` suite, and more. Every PSKit call runs through PSKit's five-tier neural safety pipeline (KAN scorer + blocklist + path check + Gemma review + cache) with a ~1 ms overhead and an append-only audit log.

---

## Quick start

```bash
git clone https://github.com/Nickalus12/anvil.git
cd anvil
pip install -e .
```

Running requires OpenAI-compatible endpoints at `:8000–8003`. Point them at any model fleet — the Qwen defaults are the recommended starting stack for a single 80 GB GPU.

### One-shot mode
```bash
AGENT_YOLO=1 python -m autonomous.main --yolo "Add pytest fixtures to tests/conftest.py for the new DB mock"
```

### Daemon mode (reads `autonomous/BACKLOG.txt`, one goal per line)
```bash
AGENT_YOLO=1 python -m autonomous.daemon --workspace . --yolo --max-cycles 10
```

### Daemon with PSKit tools enabled
```bash
AGENT_YOLO=1 PSKIT_ENABLED=1 \
  CONTEXT7_API_KEY=... NIA_API_KEY=... \
  python -m autonomous.daemon --workspace . --yolo --max-cycles 10
```

To stop mid-run without data loss: `touch autonomous/STOP` — the daemon finishes its current cycle and exits cleanly.

---

## Safety model

| Layer | What it catches | Cost |
|-------|-----------------|------|
| Shell allowlist | Destructive / privileged commands reach an approval prompt instead of executing | pre-execution |
| Python write syntax check | Every `write_file`/`edit_file` parses before hitting disk | ~1 ms |
| Ruff `F821`/`F822`/`F823` diff gate | New undefined-name / used-before-assignment references → rollback on disk | ~50 ms |
| Test regression gate | Any passing test that starts failing after the edit → rollback on disk | seconds (depends on suite) |
| Reflector completeness override | Executor claims "done" on a "do all N" goal when fewer than N are done → cycle continues | ~0 ms |
| PSKit neural safety pipeline (optional) | Command fingerprint scored by KAN + blocklist + path safety + Gemma review | ~1 ms |
| Planner inventory grounding | LLM must propose goals that reference real function / file / test names from AST scan | 1 regex check |

---

## Memory

Three stores, all SQLite:

- **`autonomous/data/memory.sqlite`** — lessons + episodes, retrieved by cosine similarity on an embedded description. "Don't delete the `hex_color = region["color"]` line when narrowing the exception below."
- **`autonomous/data/skills.sqlite`** — named tool sequences that have succeeded, with success/failure counts. "`fetch_models_data`: call `pskit_http_request` against 8000, 8001, 8002, 8003 in sequence; return a JSON dict."
- **`autonomous/data/checkpoints.sqlite`** — LangGraph's session checkpoints so a session can resume after a crash.

All stores are local to each deployment (gitignored). Seed a fresh machine with `python -m autonomous.seed_docs --workspace .` which pre-loads Context7 docs for the libraries in your code into the lesson store.

---

## Status

This is a working prototype. It has produced measurable wins on real tasks (e.g. 7-of-7 pytest smoke tests generated for a 580-line pipeline, 100 % passing after the read-before-write discipline landed; automated porting of a Windows-only MCP server to cross-platform with all 9 CI jobs green). It still makes mistakes that a human reviewer would catch — don't merge its diffs unattended.

See `CHANGELOG.md` for release-by-release improvements.

---

## Licence

MIT. See [LICENSE](LICENSE).
