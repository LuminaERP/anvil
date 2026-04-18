"""
Executor node — ReAct loop with native OpenAI-style tool calling.

Processes ONE subgoal from state.subgoals (the next pending one whose deps are met).
Runs a bounded inner loop: model decides tool calls → we execute them → feed
observations back → repeat until the model emits a final answer OR we hit the cap.

Terminal events:
  subgoal.status = "done"   -> subgoal.result is the final answer
  subgoal.status = "failed" -> error captured in subgoal.result
"""
from __future__ import annotations
import datetime
import json
import os
import time
import traceback
from typing import Any

from openai import OpenAI, BadRequestError

from ..config import CONFIG
from ..state import AgentState, Subgoal, HistoryEvent, ToolCall
from ..tools import REGISTRY, ToolError


_SYSTEM_PROMPT = """You are the Executor, a ReAct coding agent with tools. Call tools via the tool_calls API (NOT as text with <tool_call> tags).

You will be given:
  - ORIGINAL GOAL
  - SUBGOAL (what you must complete NOW)
  - DEPENDENCY OUTPUTS from prior subgoals
  - MEMORY: relevant lessons from prior sessions (may include pre-seeded library docs)

MANDATORY DISCIPLINE — violations cause the reviewer to REJECT your work:

1. **READ BEFORE WRITING.** If you will write tests for module X or edit module X:
   a. FIRST call read_file on module X (or grep for `^def ` in X) to confirm what functions ACTUALLY exist and their signatures.
   b. Use the REAL function names from the file, not what you think they should be called.
   c. If the module name is `stitchforge_pipeline.py`, import as `stitchforge_pipeline`, not `pipeline`.

2. **GROUND UNFAMILIAR APIs IN REAL SOURCES.** Before writing code that uses a library you're unsure of:
   a. For **API reference / syntax**: call context7_resolve → context7_docs with a narrow topic.
   b. For **concrete code examples**: call nia_package_search with the registry (pypi/npm/etc.), package name, and 1-3 semantic queries. This returns REAL SOURCE CODE from the actual package — trust it over your training intuition.
   c. If both fail, note the uncertainty in your final answer rather than guessing.

3. **PREFER edit_file OVER apply_patch.** Unified diffs are hard to get right; the patch format gets rejected if a single whitespace is wrong. Use edit_file(path, start_line, end_line, new_content) for targeted changes — it's far more reliable.

4. **VERIFY AFTER WRITING.** After write_file or edit_file:
   a. If you wrote a .py file, call run_bash with `python -c "import ast; ast.parse(open('PATH').read())"` to prove it parses.
   b. If you wrote a test file, call run_pytest on it and report pass/fail in your final answer.
   c. If you wrote a report, read_file the first 30 lines back to confirm it contains what you intended.

5. **CITE FROM OBSERVATIONS, NEVER INVENT.**
   - URLs in reports MUST be URLs that appeared in an actual web_fetch / web_search / context7 observation.
   - Line numbers in bug reports MUST come from a read_file that shows that line.
   - Function names MUST come from a read_file or grep observation.
   - Code patterns from nia_package_search are GROUND TRUTH; prefer them to your own intuition.

6. **NO TOOL-CALL TEXT.** Never write `<tool_call>` or `<function=>` as content. Use the proper tool_calls API.

7. **NO STUBS.** Tests with `pass`, `...`, or placeholder asserts (`assert True`, `assert result == True` where result is hardcoded True) are automatic failures.

8. **EDIT EXISTING, DON'T CREATE NEW.** When the goal is to fix a bug or resolve an issue in an existing codebase:
   a. Your job is to MODIFY existing files, not to create new ones. Creating a reproduction script or a new test file is NOT a fix.
   b. Use list_symbols(path) to navigate unfamiliar files before read_file — it's 50x cheaper on context.
   c. Use read_symbol(path, name) to see just the relevant function, not the whole file.
   d. Only create a new file if the goal explicitly says to create it (e.g., "write a new test suite named X").

9. **CONTEXT ECONOMY.** Every tool observation goes into your context window. Don't read the whole file if you only need 20 lines — use read_file_range, list_symbols, or read_symbol. Don't list a huge directory — use glob_files with a specific pattern.

10. **NO REPETITIONS.** If you just called a tool with specific arguments and got an observation, calling the exact same tool+arguments again wastes context. Either read the prior observation more carefully or try a different approach.

11. Emit a final text answer (no tool_calls) ONLY when steps 1-10 are satisfied."""


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _pick_next_subgoal(subgoals: list[Subgoal]) -> int | None:
    """Return index of next runnable subgoal, or None if all done/blocked."""
    done_ids = {s.id for s in subgoals if s.status == "done"}
    for i, s in enumerate(subgoals):
        if s.status != "pending":
            continue
        if all(d in done_ids for d in s.depends_on):
            return i
    return None


def _dep_context(subgoal: Subgoal, subgoals: list[Subgoal]) -> str:
    if not subgoal.depends_on:
        return ""
    lines = ["DEPENDENCY OUTPUTS:"]
    for dep_id in subgoal.depends_on:
        dep = next((s for s in subgoals if s.id == dep_id), None)
        if dep and dep.result:
            lines.append(f"\n[{dep_id}: {dep.description}]\n{dep.result[:4000]}")
    return "\n".join(lines)


def _format_user_turn(goal: str, subgoal: Subgoal, subgoals: list[Subgoal], memory_ctx: str) -> str:
    parts = [
        f"ORIGINAL GOAL:\n{goal}",
        f"\nSUBGOAL ({subgoal.id}):\n{subgoal.description}",
    ]
    dep = _dep_context(subgoal, subgoals)
    if dep:
        parts.append("\n" + dep)
    if memory_ctx:
        parts.append(f"\nMEMORY:\n{memory_ctx}")
    return "\n".join(parts)


_REPETITION_CACHE: dict[str, dict[str, str]] = {}  # session_id -> {args_hash: result}


def _args_hash(name: str, args: dict) -> str:
    import hashlib
    import json
    try:
        key = f"{name}::{json.dumps(args, sort_keys=True, default=str)}"
    except (TypeError, ValueError):
        key = f"{name}::{args!r}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _run_tool(name: str, args: dict, session_id: str = "__default__") -> ToolCall:
    start = time.time()
    tc = ToolCall(name=name, args=args)

    # Wrap in execute_tool {name} span so Grafana / Tempo / Jaeger can show
    # tool-level timing breakdowns. The span stays open for the full tool run.
    from .. import telemetry as _tel
    with _tel.tool_span(tool_name=name, args=args, session_id=session_id, subgoal_id=session_id) as tspan:

        # Repetition detector: if the same (tool, args) was just called, short-circuit
        # with a warning so the agent sees "you already did this".
        session_cache = _REPETITION_CACHE.setdefault(session_id, {})
        h = _args_hash(name, args)
        if h in session_cache:
            tc.result = (
                f"[REPEAT DETECTED] You just called {name} with identical arguments. "
                f"The prior observation was:\n"
                f"---\n{session_cache[h][:1500]}\n---\n"
                f"Do not call the same tool+args again; use the result above or try a different approach."
            )
            tc.duration_ms = int((time.time() - start) * 1000)
            _tel.record_tool_result(tspan, success=True, duration_ms=tc.duration_ms,
                                    result_chars=len(tc.result or ""), error="repeat")
            return tc

        try:
            tool = REGISTRY.get(name)
            tc.result = tool.fn(**args)
        except ToolError as e:
            tc.error = str(e)
        except TypeError as e:
            tc.error = f"bad arguments: {e}"
        except Exception as e:
            tc.error = f"unexpected {type(e).__name__}: {e}"
        tc.duration_ms = int((time.time() - start) * 1000)

        # Apply context-discipline truncation before this observation flows back
        # into the executor's conversation. Short outputs pass through unchanged.
        if tc.result:
            try:
                from ..safety.output_truncation import truncate_observation
                tr = truncate_observation(name, args, tc.result)
                if tr.truncated:
                    tc.result = tr.text
            except Exception:
                pass

        # Record successful calls in the repetition cache so repeats can be detected
        if tc.result and not tc.error:
            session_cache[h] = tc.result

        # Finalize the tool span with outcome attributes
        _tel.record_tool_result(
            tspan,
            success=(tc.error is None),
            duration_ms=tc.duration_ms,
            result_chars=len(tc.result or ""),
            error=tc.error or "",
        )
        return tc


def _reset_repetition_cache(session_id: str = "__default__") -> None:
    """Called at subgoal boundaries so cache doesn't leak between subgoals."""
    _REPETITION_CACHE.pop(session_id, None)


def executor_node(state: AgentState) -> dict:
    subgoals: list[Subgoal] = state.get("subgoals") or []
    idx = _pick_next_subgoal(subgoals)
    if idx is None:
        # Nothing runnable; let the router see all done (or blocked)
        return {"status": "reflecting"}

    sg = subgoals[idx]
    session_id_for_span = state.get("session_id", "") or ""
    # Open the invoke_agent span here so everything in this executor pass
    # nests under it in Grafana / Tempo / Jaeger.
    from .. import telemetry as _tel
    agent_span_cm = _tel.agent_span(
        agent_name="anvil.executor",
        session_id=session_id_for_span,
        cycle=state.get("iterations", 0) if isinstance(state, dict) else 0,
        node="executor",
    )
    _ROOT_AGENT_SPAN = agent_span_cm.__enter__()
    try:
        _ROOT_AGENT_SPAN.set_attribute("anvil.subgoal_id", sg.id)
        _ROOT_AGENT_SPAN.set_attribute("anvil.subgoal_description", (sg.description or "")[:300])
    except Exception:
        pass

    sg.status = "running"
    sg.attempts += 1
    fleet = CONFIG["fleet"]
    budget = CONFIG["budget"]

    # Fresh repetition cache per subgoal; prior subgoal's observations don't count
    _reset_repetition_cache(sg.id)

    # Small/cheap subgoals go to worker; reasoning-heavy to executor.
    ep = fleet.worker if sg.role == "worker" else fleet.executor
    client = OpenAI(base_url=ep.base_url, api_key="EMPTY")

    history_events: list[HistoryEvent] = [HistoryEvent(
        kind="subgoal_start",
        subgoal_id=sg.id,
        content=f"[{sg.role}] {sg.description}",
        timestamp=_now(),
    )]

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": _format_user_turn(
            state["goal"], sg, subgoals, state.get("memory_context", ""))},
    ]
    tool_schemas = REGISTRY.schemas() if sg.role == "executor" else None
    all_tool_calls: list[ToolCall] = []

    # Difficulty-aware turn cap: adapter/planner/env can attach `difficulty`.
    # Easy tasks finish fast; hard ones get room to iterate.
    from ..config import resolve_turn_cap
    difficulty = (
        getattr(sg, "difficulty", None)
        or (state.get("task_difficulty") if isinstance(state, dict) else None)
        or os.environ.get("AGENT_TASK_DIFFICULTY")
    )
    max_turns = resolve_turn_cap(difficulty, default=budget.executor_max_turns)

    final_text = ""

    # Lazy imports for budget + compaction so execution still works if the
    # optional telemetry packages aren't installed.
    try:
        from ..budget import get_ledger, BudgetExceeded
        from ..context_compaction import maybe_compact
        from .. import telemetry as tel
        session_id = state.get("session_id", "")
        ledger = get_ledger(session_id) if session_id else None
    except Exception:
        ledger = None
        maybe_compact = None  # type: ignore[assignment]
        tel = None  # type: ignore[assignment]

    # Determine the model's context limit so compaction has a number to aim at.
    # vLLM publishes `max_model_len`; we conservatively use ep.max_context or 16k.
    model_ctx_limit = int(getattr(ep, "max_context", 0) or 16_000)
    # Cheap summarizer model: prefer the fleet's worker (smallest/cheapest).
    try:
        summarizer_model = fleet.worker.name
        summarizer_client = OpenAI(base_url=fleet.worker.base_url, api_key="EMPTY")
    except Exception:
        summarizer_model = ep.name  # fallback to the executor itself
        summarizer_client = client

    for turn in range(max_turns):
        # Proactive compaction: check before each LLM call so we never blow
        # the model's context window.
        if maybe_compact is not None:
            try:
                messages, compaction_result = maybe_compact(
                    messages=messages,
                    model_ctx_limit=model_ctx_limit,
                    openai_client=summarizer_client,
                    summarizer_model=summarizer_model,
                    session_id=state.get("session_id", ""),
                    soft_ratio=0.70,
                    keep_last_n_turns=3,
                )
                if compaction_result is not None:
                    history_events.append(HistoryEvent(
                        kind="compaction",
                        subgoal_id=sg.id,
                        content=(f"compacted {compaction_result.messages_before}→"
                                 f"{compaction_result.messages_after} msgs, "
                                 f"~{compaction_result.tokens_before}→"
                                 f"{compaction_result.tokens_after} tokens"),
                        timestamp=_now(),
                    ))
                    if ledger:
                        ledger.reset_degrade_trigger()
            except Exception:
                pass  # compaction is best-effort; never break a turn over it

        # Pre-call budget check
        if ledger is not None:
            try:
                rendered_prompt = "\n".join((m.get("content") or "") for m in messages)
                decision = ledger.check_pre_call(
                    model=ep.name,
                    prompt_text=rendered_prompt,
                    max_output_tokens=ep.max_tokens,
                )
                if decision.action == "stop":
                    sg.status = "failed"
                    sg.result = f"[BUDGET EXCEEDED] {decision.reason}"
                    history_events.append(HistoryEvent(
                        kind="budget_exceeded",
                        subgoal_id=sg.id,
                        content=decision.reason,
                        timestamp=_now(),
                    ))
                    break
                # 'degrade' triggers already-done compaction on the next turn;
                # we just log here.
                if decision.action == "degrade":
                    history_events.append(HistoryEvent(
                        kind="budget_degrade",
                        subgoal_id=sg.id,
                        content=decision.reason,
                        timestamp=_now(),
                    ))
            except Exception:
                pass

        kwargs: dict[str, Any] = dict(
            model=ep.name,
            messages=messages,
            max_tokens=ep.max_tokens,
            temperature=ep.temperature,
        )
        if tool_schemas:
            kwargs["tools"] = tool_schemas
            kwargs["tool_choice"] = "auto"

        try:
            r = client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            sg.status = "failed"
            sg.result = f"LLM bad request: {e}"
            # Record for error taxonomy
            try:
                from ..metrics import METRICS
                METRICS.record_error(state.get("session_id", ""), str(e))
            except Exception:
                pass
            break
        except Exception as e:
            sg.status = "failed"
            sg.result = f"LLM call failed: {e}"
            try:
                from ..metrics import METRICS
                METRICS.record_error(state.get("session_id", ""), str(e))
            except Exception:
                pass
            break

        # Token + cost accounting
        try:
            from ..metrics import METRICS, extract_usage
            p_tok, c_tok = extract_usage(r)
            if p_tok or c_tok:
                METRICS.record_llm_call(
                    session_id=state.get("session_id", ""),
                    model=ep.name,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    node="executor" if sg.role == "executor" else "worker",
                )
        except Exception:
            pass

        msg = r.choices[0].message
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in (msg.tool_calls or [])
            ] if msg.tool_calls else None,
        })

        # No tool calls => final answer
        if not msg.tool_calls:
            final_text = msg.content or "(no content)"
            sg.status = "done"
            sg.result = final_text
            break

        # Execute each tool call, append tool responses
        if len(msg.tool_calls) > budget.max_tool_calls_per_step:
            # Truncate: model is fanning out too hard
            msg.tool_calls = msg.tool_calls[: budget.max_tool_calls_per_step]

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as e:
                obs = f"ERROR: invalid JSON arguments: {e}"
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": obs})
                continue
            run = _run_tool(tc.function.name, args, session_id=sg.id)
            all_tool_calls.append(run)
            obs = run.result if run.error is None else f"ERROR: {run.error}"
            # Truncate massive observations to protect context
            if len(obs or "") > 20_000:
                obs = obs[:20_000] + "\n... (observation truncated)"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": obs or ""})

            history_events.append(HistoryEvent(
                kind="tool_call",
                subgoal_id=sg.id,
                content=f"{run.name}({', '.join(f'{k}={str(v)[:60]}' for k, v in args.items())}) -> {'err' if run.error else 'ok'}",
                data={"tool": run.name, "args": args, "error": run.error, "duration_ms": run.duration_ms},
                timestamp=_now(),
            ))
    else:
        # Hit turn cap without terminating
        sg.status = "failed"
        sg.result = f"executor exhausted {budget.executor_max_turns} turns without a final answer"

    history_events.append(HistoryEvent(
        kind="subgoal_end",
        subgoal_id=sg.id,
        content=f"[{sg.status}] {sg.result[:200] if sg.result else ''}",
        data={"status": sg.status, "attempts": sg.attempts, "tool_calls": len(all_tool_calls)},
        timestamp=_now(),
    ))

    # Close the invoke_agent span with final status
    try:
        _ROOT_AGENT_SPAN.set_attribute("anvil.subgoal_status", sg.status)
        _ROOT_AGENT_SPAN.set_attribute("anvil.tool_calls_count", len(all_tool_calls))
        _ROOT_AGENT_SPAN.set_attribute("anvil.final_turn", int(turn) if 'turn' in dir() else 0)
    except Exception:
        pass
    try:
        agent_span_cm.__exit__(None, None, None)
    except Exception:
        pass

    # Replace subgoals list (mutated in place; also return to state explicitly for safety)
    return {
        "subgoals": subgoals,
        "last_tool_calls": all_tool_calls,
        "last_observation": sg.result or "",
        "status": "reflecting" if _pick_next_subgoal(subgoals) is None else "executing",
        "history": history_events,
    }
