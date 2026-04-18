"""OpenTelemetry integration for Anvil — 2026 GenAI semantic conventions.

Follows the current canonical GenAI semconv:
  - span names: `chat {model}`, `invoke_agent {name}`, `execute_tool {name}`
  - attributes: gen_ai.operation.name, gen_ai.provider.name, gen_ai.request.model,
    gen_ai.response.model, gen_ai.usage.input_tokens, gen_ai.usage.output_tokens,
    gen_ai.agent.name, gen_ai.agent.id, gen_ai.conversation.id, gen_ai.tool.name,
    gen_ai.tool.call.id

Exporter precedence:
  1. OTEL_EXPORTER_OTLP_ENDPOINT → OTLP/gRPC (default for real deployments)
  2. ANVIL_TELEMETRY_CONSOLE=1   → console exporter (dev)
  3. default                     → no-op (safe; calls cost near-zero)

Auto-instrumentation:
  If opentelemetry-instrumentation-openai-v2 is installed, we call
  OpenAIInstrumentor().instrument() so all OpenAI-SDK chat.completions.create
  calls get spans automatically — including vLLM endpoints (which speak the
  OpenAI API). Manual spans (invoke_agent, execute_tool, anvil.compaction)
  wrap around those.

Subprocess propagation: inject_into_env(env) writes TRACEPARENT to the
subprocess environment; child processes can extract via extract_from_env().

atexit force-flush: the BatchSpanProcessor would otherwise drop in-flight
spans on crash. We register a flush on process exit.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)


# ---- semantic convention constants (canonical 2026 GenAI) -------------------

# gen_ai.operation.name enum values
OP_CHAT = "chat"
OP_EXECUTE_TOOL = "execute_tool"
OP_INVOKE_AGENT = "invoke_agent"
OP_CREATE_AGENT = "create_agent"
OP_TEXT_COMPLETION = "text_completion"

# gen_ai.provider.name values
PROVIDER_VLLM = "vllm"
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"

# Anvil-specific (namespaced under anvil.*)
ATTR_SESSION_ID = "anvil.session_id"
ATTR_SUBGOAL_ID = "anvil.subgoal_id"
ATTR_NODE = "anvil.node"
ATTR_CYCLE = "anvil.cycle"
ATTR_COMPACTION_BEFORE = "anvil.compaction.tokens_before"
ATTR_COMPACTION_AFTER = "anvil.compaction.tokens_after"
ATTR_BUDGET_STATUS = "anvil.budget.status"


# ---- lazy singleton state ---------------------------------------------------

_INIT_LOCK = threading.Lock()
_TRACER: Any = None
_PROVIDER: Any = None
_AUTO_INSTRUMENTED = False


def init(
    service_name: str = "anvil",
    service_instance_id: str | None = None,
    auto_instrument_openai: bool = True,
    force: bool = False,
) -> Any:
    """Initialise the OpenTelemetry tracer. Idempotent unless force=True."""
    global _TRACER, _PROVIDER, _AUTO_INSTRUMENTED

    with _INIT_LOCK:
        if _TRACER is not None and not force:
            return _TRACER

        # Opt into latest GenAI semconv even when the stable release trails
        os.environ.setdefault("OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental")

        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,
                ConsoleSpanExporter,
            )
        except ImportError:
            logger.info("opentelemetry not installed; telemetry disabled")
            _TRACER = _noop_tracer()
            return _TRACER

        import socket
        instance_id = service_instance_id or os.environ.get(
            "OTEL_SERVICE_INSTANCE_ID", socket.gethostname()
        )

        resource = Resource.create({
            "service.name": service_name,
            "service.namespace": "lumina",
            "service.version": os.environ.get("ANVIL_VERSION", "0.1.0"),
            "service.instance.id": instance_id,
            "deployment.environment": os.environ.get("ANVIL_ENV", "dev"),
        })
        _PROVIDER = TracerProvider(resource=resource)

        otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
        console_enabled = os.environ.get("ANVIL_TELEMETRY_CONSOLE", "").lower() in ("1", "true", "yes")

        if otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
                _PROVIDER.add_span_processor(BatchSpanProcessor(exporter))
                logger.info("telemetry → OTLP %s", otlp_endpoint)
            except ImportError:
                logger.warning("OTLP exporter missing; "
                               "pip install opentelemetry-exporter-otlp-proto-grpc")

        if console_enabled:
            _PROVIDER.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        if not otlp_endpoint and not console_enabled:
            logger.debug("telemetry: no exporter configured (no-op mode)")

        trace.set_tracer_provider(_PROVIDER)
        _TRACER = trace.get_tracer("anvil")

        # Register atexit force-flush — BatchSpanProcessor otherwise drops in-flight
        # spans when the process exits (including budget-exceeded crashes)
        atexit.register(_flush_on_exit)

        # Auto-instrument the OpenAI SDK if the -v2 package is available.
        # vLLM speaks the OpenAI API, so every planner/executor/reflector LLM
        # call to our local fleet gets a `chat {model}` span for free.
        if auto_instrument_openai and not _AUTO_INSTRUMENTED:
            try:
                from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
                OpenAIInstrumentor().instrument()
                _AUTO_INSTRUMENTED = True
                logger.debug("telemetry: OpenAI SDK auto-instrumented")
            except ImportError:
                logger.debug(
                    "opentelemetry-instrumentation-openai-v2 not installed; "
                    "LLM spans will be manual-only"
                )
            except Exception as e:
                logger.debug("auto-instrument failed: %s", e)

        return _TRACER


def _flush_on_exit() -> None:
    """Force-flush all spans before process exit to avoid dropping them."""
    try:
        if _PROVIDER is not None:
            _PROVIDER.force_flush(timeout_millis=5000)
            _PROVIDER.shutdown()
    except Exception:
        pass


def _noop_tracer() -> Any:
    class _NoopSpan:
        def set_attribute(self, *a, **k): pass
        def set_attributes(self, *a, **k): pass
        def add_event(self, *a, **k): pass
        def record_exception(self, *a, **k): pass
        def set_status(self, *a, **k): pass
        def end(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass

    class _NoopTracer:
        def start_as_current_span(self, *a, **k):
            return _NoopSpan()
        def start_span(self, *a, **k):
            return _NoopSpan()

    return _NoopTracer()


def tracer() -> Any:
    if _TRACER is None:
        init()
    return _TRACER


# ---- span factories (use canonical semconv names) ---------------------------

@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    """Generic span context manager. Prefer the typed factories below."""
    t = tracer()
    attrs = _normalise_attrs(attributes or {})
    try:
        with t.start_as_current_span(name, attributes=attrs) as s:
            yield s
    except Exception:
        yield _noop_tracer().start_as_current_span(name)


@contextmanager
def agent_span(
    agent_name: str,
    session_id: str,
    cycle: int | None = None,
    node: str | None = None,
) -> Iterator[Any]:
    """`invoke_agent {agent_name}` span. Wrap each planner/executor/reflector pass."""
    attrs: dict[str, Any] = {
        "gen_ai.operation.name": OP_INVOKE_AGENT,
        "gen_ai.agent.name": agent_name,
        "gen_ai.agent.id": agent_name,
        "gen_ai.conversation.id": session_id,
        ATTR_SESSION_ID: session_id,
    }
    if cycle is not None:
        attrs[ATTR_CYCLE] = cycle
    if node:
        attrs[ATTR_NODE] = node
    with span(f"invoke_agent {agent_name}", attrs) as s:
        yield s


@contextmanager
def tool_span(
    tool_name: str,
    args: dict,
    session_id: str = "",
    subgoal_id: str = "",
    tool_call_id: str = "",
) -> Iterator[Any]:
    """`execute_tool {tool_name}` span."""
    attrs = {
        "gen_ai.operation.name": OP_EXECUTE_TOOL,
        "gen_ai.tool.name": tool_name,
        "gen_ai.tool.call.id": tool_call_id or _args_hash(tool_name, args),
        ATTR_SESSION_ID: session_id,
        ATTR_SUBGOAL_ID: subgoal_id,
    }
    with span(f"execute_tool {tool_name}", attrs) as s:
        yield s


@contextmanager
def compaction_span(session_id: str, tokens_before: int) -> Iterator[Any]:
    """`anvil.compaction` span wrapping a compaction pass."""
    attrs = {
        ATTR_SESSION_ID: session_id,
        ATTR_COMPACTION_BEFORE: tokens_before,
    }
    with span("anvil.compaction", attrs) as s:
        yield s


# ---- attribute recorders (attach on exit of a span) -------------------------

def record_llm_usage(
    span_obj: Any,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float = 0.0,
    provider: str = PROVIDER_VLLM,
) -> None:
    """Attach gen_ai.usage.* + gen_ai.*.model + cost."""
    try:
        span_obj.set_attribute("gen_ai.provider.name", provider)
        span_obj.set_attribute("gen_ai.response.model", model)
        span_obj.set_attribute("gen_ai.usage.input_tokens", int(prompt_tokens))
        span_obj.set_attribute("gen_ai.usage.output_tokens", int(completion_tokens))
        if cost_usd:
            # Custom attribute; no canonical semconv for cost yet
            span_obj.set_attribute("anvil.cost.usd", float(cost_usd))
    except Exception:
        pass


def record_tool_result(
    span_obj: Any,
    success: bool,
    duration_ms: int,
    result_chars: int = 0,
    error: str = "",
) -> None:
    try:
        span_obj.set_attribute("anvil.tool.success", bool(success))
        span_obj.set_attribute("anvil.tool.duration_ms", int(duration_ms))
        if result_chars:
            span_obj.set_attribute("anvil.tool.result_chars", int(result_chars))
        if error:
            span_obj.set_attribute("anvil.tool.error", str(error)[:500])
    except Exception:
        pass


def record_compaction(
    span_obj: Any,
    tokens_after: int,
    messages_kept: int,
    messages_summarised: int,
) -> None:
    try:
        span_obj.set_attribute(ATTR_COMPACTION_AFTER, tokens_after)
        span_obj.set_attribute("anvil.compaction.messages_kept", messages_kept)
        span_obj.set_attribute("anvil.compaction.messages_summarised", messages_summarised)
    except Exception:
        pass


def record_budget(span_obj: Any, status: str, spent_usd: float, spent_tokens: int) -> None:
    try:
        span_obj.set_attribute(ATTR_BUDGET_STATUS, status)
        span_obj.set_attribute("anvil.budget.spent_usd", float(spent_usd))
        span_obj.set_attribute("anvil.budget.spent_tokens", int(spent_tokens))
    except Exception:
        pass


# ---- subprocess trace propagation -------------------------------------------

def inject_into_env(env: dict[str, str]) -> dict[str, str]:
    """Write TRACEPARENT (and TRACESTATE) into an env dict for subprocess.run.

    The child process can call `extract_from_env(os.environ)` to get a Context
    and pass it to start_as_current_span(context=ctx).
    """
    try:
        from opentelemetry.propagate import inject
        # TextMapPropagator wants a mutable mapping with setdefault semantics.
        # A simple dict works — inject() writes string keys.
        inject(env)
    except Exception:
        pass
    return env


def extract_from_env(env: dict[str, str] | None = None) -> Any:
    """Extract a trace context from environment variables (for child procs)."""
    if env is None:
        env = dict(os.environ)
    try:
        from opentelemetry.propagate import extract
        return extract(env)
    except Exception:
        return None


# ---- helpers ----------------------------------------------------------------

def _args_hash(name: str, args: dict) -> str:
    try:
        blob = f"{name}::{json.dumps(args, sort_keys=True, default=str)}"
    except (TypeError, ValueError):
        blob = f"{name}::{args!r}"
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _normalise_attrs(d: dict) -> dict:
    """OTel only accepts str/int/float/bool/seq-of-same on attributes."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (str, int, float, bool)) for x in v):
            out[k] = v
        else:
            out[k] = str(v)[:500]
    return out
