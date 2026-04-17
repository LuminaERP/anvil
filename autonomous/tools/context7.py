"""
Context7 tools: ground-truth library documentation lookup.

Two tools:
  context7_resolve(name)             — find a library's canonical ID
  context7_docs(library_id, topic, tokens)  — fetch real docs

Why this matters: the executor kept hallucinating API signatures because it was
guessing from training data. Context7 returns CURRENT, VERSION-SPECIFIC docs,
eliminating an entire class of mistakes.
"""
from __future__ import annotations
import os
from typing import Optional

import httpx

from .base import Tool, ToolError, register


_BASE = "https://context7.com/api/v1"


def _api_key() -> str:
    k = os.environ.get("CONTEXT7_API_KEY", "")
    if not k:
        raise ToolError("CONTEXT7_API_KEY not set")
    return k


def _context7_resolve(name: str, limit: int = 5) -> str:
    """Search Context7 for a library. Returns id + trust score for top matches."""
    if not name.strip():
        raise ToolError("name required")
    try:
        r = httpx.get(f"{_BASE}/search", params={"query": name},
                      headers={"Authorization": f"Bearer {_api_key()}"},
                      timeout=15.0)
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError as e:
        raise ToolError(f"context7 search failed: {e}")

    results = data.get("results") or []
    if not results:
        return f"(no matches for {name!r})"
    lines = []
    for i, res in enumerate(results[:limit], 1):
        tid = res.get("id", "?")
        title = res.get("title", "")
        desc = (res.get("description", "") or "")[:200]
        trust = res.get("trustScore", "?")
        versions = res.get("versions") or []
        vtag = f"  v{versions[0]}" if versions else ""
        lines.append(f"[{i}] id={tid}  trust={trust}{vtag}\n    {title}\n    {desc}")
    return "\n\n".join(lines)


def _context7_docs(library_id: str, topic: str = "", tokens: int = 4000) -> str:
    """
    Fetch docs for a resolved library id (e.g. "/pytest-dev/pytest").
    Optional topic narrows the result (e.g. "fixtures", "parametrize").
    """
    if not library_id.strip():
        raise ToolError("library_id required")
    lid = library_id.lstrip("/")
    params: dict = {"tokens": max(500, min(tokens, 15000))}
    if topic:
        params["topic"] = topic
    try:
        r = httpx.get(f"{_BASE}/{lid}", params=params,
                      headers={"Authorization": f"Bearer {_api_key()}"},
                      timeout=20.0)
        r.raise_for_status()
        text = r.text
    except httpx.HTTPError as e:
        raise ToolError(f"context7 docs failed: {e}")

    # Context7 returns markdown already.
    if len(text) > 25_000:
        text = text[:25_000] + "\n... (truncated; request with lower tokens or a topic)"
    return f"# docs: {library_id}" + (f"  topic={topic}" if topic else "") + f"\n\n{text}"


register(Tool(
    name="context7_resolve",
    description="Find a library's canonical Context7 ID. Call this FIRST before context7_docs. Returns ID, trust score, and version for top matches.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Library name, e.g. 'pytest', 'langgraph', 'pyembroidery'."},
            "limit": {"type": "integer", "default": 5},
        },
        "required": ["name"],
    },
    category="read",
    fn=_context7_resolve,
))


register(Tool(
    name="context7_docs",
    description="Fetch AUTHORITATIVE, version-current documentation for a library (by Context7 id). Use this BEFORE writing code that calls unfamiliar APIs. Optional topic narrows results (e.g. 'fixtures', 'mock', 'async').",
    parameters={
        "type": "object",
        "properties": {
            "library_id": {"type": "string", "description": "Context7 id, e.g. '/pytest-dev/pytest'. Get from context7_resolve."},
            "topic":      {"type": "string", "default": "", "description": "Optional topic filter."},
            "tokens":     {"type": "integer", "default": 4000, "description": "Max doc tokens to return (500-15000)."},
        },
        "required": ["library_id"],
    },
    category="read",
    fn=_context7_docs,
))
