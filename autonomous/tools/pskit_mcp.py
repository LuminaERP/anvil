"""
pskit MCP bridge — spawns pskit in stdio mode and registers each of its tools
in our agent's REGISTRY with a `pskit_` prefix.

Discovery happens lazily on first call — spinning up pwsh sessions is slow, so
we avoid it at import time. Subsequent calls reuse the same ClientSession.
"""
from __future__ import annotations
import asyncio
import atexit
import json
import logging
import os
import re
import shutil
import threading
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .base import REGISTRY, Tool, ToolError, register

logger = logging.getLogger(__name__)

# pskit's server + manager emit an INFO log for every tool call (safety
# pipeline trace, session state, etc.). At INFO this floods daemon output
# with ~10 lines per tool invocation. Our bridge already logs the observable
# interface; the internals are only useful at debug level.
for _noisy in ("pskit.manager", "pskit.server", "pskit.safety", "mcp.server"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# -- category mapping: pskit tool metadata -> our "read" / "write" / "shell"
_READ_PREFIXES = (
    "read_", "list_", "find_", "search_", "diff_", "git_status", "git_diff",
    "git_log", "git_branch", "git_blame", "which", "get_env", "memory_usage",
    "disk_usage", "port_status", "process_info", "gpu_status",
)
_WRITE_PREFIXES = (
    "write_", "edit_", "create_", "move_", "delete_", "git_commit",
    "git_checkout", "git_push", "git_stash",
)


def _category_for(name: str) -> str:
    n = name.lower()
    if any(n.startswith(p) for p in _READ_PREFIXES):
        return "read"
    if any(n.startswith(p) for p in _WRITE_PREFIXES):
        return "write"
    # run_command, http_request, build_project, test_project, install_package
    return "shell"


# -- background event loop dedicated to MCP IO
# FastMCP's stdio transport is strictly async. Our tool fns are sync (they
# return a str). Simplest reliable bridge: one background thread running an
# asyncio loop, and each tool call posts a coroutine to it.

class _PSKitBridge:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._start_lock = threading.Lock()
        self._tools_registered = False
        self._boot_error: Exception | None = None

    # ---- public API ----
    def ensure_started(self, timeout: float = 180.0) -> None:
        with self._start_lock:
            if self._ready.is_set() and self._session is not None:
                return
            if shutil.which("pskit") is None:
                raise ToolError(
                    "pskit CLI not found on PATH. Install with "
                    "`pip install pskit-mcp` (>= 0.3.2 recommended on Linux/macOS)."
                )
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run_loop, name="pskit-mcp", daemon=True
                )
                self._thread.start()
            if not self._ready.wait(timeout):
                raise ToolError(
                    f"pskit MCP bridge failed to start within {timeout}s. "
                    "pskit pre-warms 3 pwsh sessions (~45s each on first run)."
                )
            if self._boot_error is not None:
                raise ToolError(f"pskit MCP bridge boot failed: {self._boot_error}")

    def call(self, tool_name: str, arguments: dict[str, Any], timeout: float = 120.0) -> str:
        self.ensure_started()
        assert self._loop is not None and self._session is not None
        coro = self._session.call_tool(tool_name, arguments=arguments)
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            result = fut.result(timeout=timeout)
        except asyncio.TimeoutError:
            raise ToolError(f"pskit tool {tool_name!r} timed out after {timeout}s")
        except Exception as e:
            raise ToolError(f"pskit tool {tool_name!r} failed: {e}")
        return _format_result(result)

    def list_tools(self) -> list[Any]:
        self.ensure_started()
        assert self._loop is not None and self._session is not None
        fut = asyncio.run_coroutine_threadsafe(self._session.list_tools(), self._loop)
        return fut.result(timeout=30.0).tools

    def shutdown(self) -> None:
        # signal the session-holder coroutine to exit; it will close cleanly
        # from within its own task (avoids anyio cross-task cancel issues).
        # IMPORTANT: set _stop first, then wait for _hold_session to observe
        # it and return naturally. Only loop.stop() as a last resort — calling
        # it out from under a running Future produces the noisy RuntimeError.
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread:
            self._thread.join(timeout=5)
            if self._thread.is_alive() and self._loop and self._loop.is_running():
                # Holder didn't return — force-stop the loop.
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._thread.join(timeout=3)

    # ---- internals ----
    async def _hold_session(self) -> None:
        """Enter the MCP session context from a single task; keep it open
        until _stop is set. This is the *only* task that touches _session's
        lifecycle, so there are no cross-task cancel-scope issues."""
        env = os.environ.copy()
        env.setdefault("PSKIT_ALLOWED_ROOT", env.get("HOME", "/workspace"))
        params = StdioServerParameters(command="pskit", args=["serve"], env=env)
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    logger.info("pskit MCP session initialized")
                    self._ready.set()
                    # Park until shutdown is signalled.
                    while not self._stop.is_set():
                        await asyncio.sleep(0.25)
        except Exception as e:
            self._boot_error = e
            logger.exception("pskit session holder crashed")
            self._ready.set()  # unblock waiters so they see the error
        finally:
            self._session = None

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._hold_session())
        except RuntimeError as e:
            # `loop.stop()` during `run_until_complete` produces this — it is
            # a normal shutdown path, not a crash. Log at debug.
            msg = str(e)
            if "stopped before Future completed" in msg:
                logger.debug("pskit bridge loop stopped during shutdown")
            else:
                logger.exception("pskit bridge loop crashed: %s", e)
                self._boot_error = e
                self._ready.set()
        except Exception as e:
            logger.exception("pskit bridge loop crashed: %s", e)
            self._boot_error = e
            self._ready.set()
        finally:
            try:
                pending = asyncio.all_tasks(self._loop)
                for t in pending:
                    t.cancel()
            except Exception:
                pass
            self._loop.close()


# CSI sequences (ESC [ …) + bare SS3 / simple mode controls.
# pwsh on Linux emits these into stdout when the line-discipline isn't a TTY.
# Match both raw ESC and the \u001b JSON-escaped form (pskit JSON-encodes the
# raw PowerShell stdout before returning it, so the escape codes survive as
# literal 6-char sequences inside the output string).
_ANSI_RAW_RE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[PX^_][^\x1b]*\x1b\\|\][^\x07]*(?:\x07|\x1b\\)|[@-Z\\-_])"
)
_ANSI_ESCAPED_RE = re.compile(
    r"\\u001[bB](?:\[[0-?]*[ -/]*[@-~]|[PX^_][^\\]*\\u001[bB]\\\\|\][^\\]*(?:\\u0007|\\u001[bB]\\\\)|[@-Z\\\\-_])"
)


def _strip_ansi(s: str) -> str:
    s = _ANSI_RAW_RE.sub("", s)
    s = _ANSI_ESCAPED_RE.sub("", s)
    return s


def _format_result(result: Any) -> str:
    """Turn a CallToolResult into a str observation."""
    parts: list[str] = []
    for item in getattr(result, "content", []) or []:
        t = getattr(item, "type", None)
        if t == "text":
            parts.append(_strip_ansi(getattr(item, "text", "")))
        else:
            parts.append(json.dumps({"type": t, "repr": str(item)[:500]}))
    out = "\n".join(parts).strip()
    if getattr(result, "isError", False):
        return f"[pskit error] {out or 'unknown error'}"
    return out or "[pskit: no output]"


_BRIDGE = _PSKitBridge()


def _format_arg_hint(schema: dict[str, Any]) -> str:
    """Return a compact hint like 'url (str, required), method=GET, body=""'.

    Models that skim descriptions pick up parameter names far more reliably
    when they appear in prose than when hidden in JSON-Schema.
    """
    props = (schema or {}).get("properties") or {}
    required = set((schema or {}).get("required") or [])
    parts: list[str] = []
    for name, spec in list(props.items())[:8]:
        t = spec.get("type") or "any"
        if name in required or "default" not in spec:
            parts.append(f"{name}:{t}")
        else:
            default = spec.get("default")
            if isinstance(default, str) and default == "":
                parts.append(f"{name}?")
            else:
                parts.append(f"{name}={default!r}")
    return ", ".join(parts)


def _make_tool_fn(tool_name: str):
    def _call(**kwargs: Any) -> str:
        return _BRIDGE.call(tool_name, kwargs)
    _call.__name__ = f"pskit_{tool_name}"
    return _call


def register_pskit_tools(prefix: str = "pskit_") -> int:
    """Discover pskit tools and register each in our REGISTRY. Returns count."""
    if _BRIDGE._tools_registered:
        return 0
    try:
        tools = _BRIDGE.list_tools()
    except Exception as e:
        logger.warning("pskit discovery failed: %s — no pskit tools registered", e)
        return 0

    count = 0
    existing = set(REGISTRY.names())
    for t in tools:
        name = f"{prefix}{t.name}"
        if name in existing:
            logger.debug("skip duplicate pskit tool %s", name)
            continue
        params = t.inputSchema or {"type": "object", "properties": {}}
        # Worker LLMs often miss parameter names when they live only in the
        # JSON-Schema. Surface them at the head of the description so the
        # first tokens the model reads are the callable arguments.
        raw_desc = (t.description or f"pskit tool {t.name}")[:1024]
        raw_desc = raw_desc.strip().split("\n")[0][:300]  # first line only, trimmed
        arg_hint = _format_arg_hint(params)
        desc = f"[pskit] {raw_desc}  args: {arg_hint}" if arg_hint else f"[pskit] {raw_desc}"
        try:
            register(Tool(
                name=name,
                description=desc,
                parameters=params,
                category=_category_for(t.name),
                fn=_make_tool_fn(t.name),
            ))
            count += 1
        except Exception as e:
            logger.warning("failed to register %s: %s", name, e)
    _BRIDGE._tools_registered = True
    logger.info("registered %d pskit tools (prefix=%r)", count, prefix)
    return count


@atexit.register
def _shutdown():
    try:
        _BRIDGE.shutdown()
    except Exception:
        pass
