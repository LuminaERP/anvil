"""
Research tools: web_search + web_fetch.
Uses duckduckgo-search (no API key required) and trafilatura for clean HTML->text.
"""
from __future__ import annotations
import json
import httpx
import trafilatura

from .base import Tool, ToolError, register


def _web_search(query: str, max_results: int = 8) -> str:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        raise ToolError("duckduckgo-search not installed")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        raise ToolError(f"search failed: {e}")
    if not results:
        return "(no results)"
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        href = r.get("href", "") or r.get("url", "")
        body = r.get("body", "")
        lines.append(f"[{i}] {title}\n    {href}\n    {body[:300]}")
    return "\n\n".join(lines)


def _web_fetch(url: str, max_chars: int = 12000) -> str:
    try:
        with httpx.Client(follow_redirects=True, timeout=20.0, headers={"User-Agent": "autonomous-agent/0.1"}) as c:
            r = c.get(url)
            r.raise_for_status()
            raw = r.text
    except Exception as e:
        raise ToolError(f"fetch failed: {e}")
    extracted = trafilatura.extract(raw, include_comments=False, include_tables=True) or ""
    if not extracted.strip():
        # Fall back to raw with basic cleanup
        extracted = " ".join(raw.split())
    if len(extracted) > max_chars:
        extracted = extracted[:max_chars] + f"\n... (truncated at {max_chars} chars)"
    return f"URL: {url}\n\n{extracted}"


register(Tool(
    name="web_search",
    description="Search the web via DuckDuckGo. Returns title + URL + snippet for each result. Use when you need current information, documentation, or to find authoritative sources.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query. Be specific."},
            "max_results": {"type": "integer", "default": 8, "description": "Max results to return (1-20)."},
        },
        "required": ["query"],
    },
    category="read",
    fn=_web_search,
))


register(Tool(
    name="web_fetch",
    description="Fetch a URL and return the main article text (cleaned). Use this after web_search to read a specific result in detail.",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "max_chars": {"type": "integer", "default": 12000},
        },
        "required": ["url"],
    },
    category="read",
    fn=_web_fetch,
))
