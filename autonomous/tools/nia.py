"""
Nia tools — deeper library knowledge than Context7.

Context7 gives a library's published docs. Nia indexes the SOURCE CODE of
packages and can answer semantic questions against real implementations.

Three tools:
  nia_package_search — semantic code search inside a specific package (pypi/npm/etc.)
  nia_oracle_ask     — multi-step research across Nia's indexed knowledge (async job with polling)
  nia_search         — unified search (modes: query, web, deep, universal)

Oracle is the killer: it runs a real research task over Nia's 150M-doc corpus
and returns a grounded answer with citations. Use it for "how do I design X"
or "what's SOTA for Y" questions that the local models would hallucinate.
"""
from __future__ import annotations
import json
import os
import time

import httpx

from .base import Tool, ToolError, register


_BASE = "https://apigcp.trynia.ai/v2"


def _key() -> str:
    k = os.environ.get("NIA_API_KEY", "")
    if not k:
        raise ToolError("NIA_API_KEY not set")
    return k


def _headers() -> dict:
    return {"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"}


def _nia_package_search(package: str, queries: list[str], registry: str = "pypi", limit: int = 5) -> str:
    if not queries:
        raise ToolError("at least one query required")
    if isinstance(queries, str):
        queries = [queries]
    body = {
        "registry": registry,
        "package_name": package,
        "semantic_queries": queries[:5],
        "limit": max(1, min(limit, 10)),
    }
    try:
        r = httpx.post(f"{_BASE}/packages/search", headers=_headers(), json=body, timeout=45.0)
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError as e:
        raise ToolError(f"nia package search failed: {e}")
    version = data.get("version_used", "?")
    results = data.get("results") or []
    if not results:
        return f"(no matches for {package}:{queries})"
    parts = [f"# nia/packages: {package} @ {version}"]
    for i, r in enumerate(results[:limit], 1):
        doc = r.get("document", "") or ""
        if len(doc) > 3000:
            doc = doc[:3000] + "\n... (snippet truncated)"
        parts.append(f"\n--- match {i} ---\n{doc}")
    out = "\n".join(parts)
    return out[:25000]


def _format_oracle_answer(state: dict, job_id: str = "") -> str:
    """Format a completed Oracle session/job into markdown with citations."""
    report = state.get("final_report") or state.get("answer") or state.get("result") or ""
    if not report:
        return f"# oracle completed (no final_report)\n\n{json.dumps(state, indent=2, default=str)[:8000]}"

    parts = [f"# Oracle Research: {state.get('title','')}"]
    if state.get("query"):
        parts.append(f"*Query:* {state['query']}")
    parts.append(f"\n## Report\n\n{str(report)[:15000]}")

    citations = state.get("citations") or []
    if citations:
        parts.append("\n## Citations\n")
        for c in citations[:10]:
            sid = c.get("source_id", "?")
            tool = c.get("tool", "")
            summary = c.get("summary", "")
            if isinstance(summary, str) and len(summary) > 600:
                summary = summary[:600] + "..."
            parts.append(f"- **[Source {sid}]** via `{tool}` — {summary}")

    dur = state.get("duration_ms")
    iters = state.get("iterations")
    meta = []
    if iters is not None: meta.append(f"iterations={iters}")
    if dur is not None:   meta.append(f"duration_ms={dur}")
    if job_id:            meta.append(f"job={job_id}")
    if meta:
        parts.append(f"\n*{' • '.join(meta)}*")
    return "\n".join(parts)


def _nia_oracle_ask(query: str, max_wait_s: int = 300, poll_every_s: int = 6) -> str:
    """
    Submit an Oracle job and poll until completion.
    Returns the grounded research answer in markdown with citations.
    """
    if not query.strip():
        raise ToolError("query required")

    # 1. Submit job
    try:
        r = httpx.post(f"{_BASE}/oracle/jobs", headers=_headers(),
                       json={"query": query}, timeout=30.0)
        r.raise_for_status()
        submit = r.json()
    except httpx.HTTPError as e:
        raise ToolError(f"oracle submit failed: {e}")
    job_id = submit.get("job_id")
    session_id = submit.get("session_id")
    if not job_id:
        raise ToolError(f"no job_id returned: {submit}")

    # 2. Poll. When the job completes, the FULL report + citations live on the SESSION,
    #    not the job itself — so fetch session when status transitions to completed.
    start = time.time()
    last_status = None
    with httpx.Client(headers=_headers(), timeout=25.0) as client:
        while time.time() - start < max_wait_s:
            try:
                pr = client.get(f"{_BASE}/oracle/jobs/{job_id}")
                pr.raise_for_status()
                job_state = pr.json()
            except httpx.HTTPError as e:
                # keep polling on transient errors
                time.sleep(poll_every_s); continue
            status = job_state.get("status", "unknown")
            if status != last_status:
                last_status = status
            if status in ("completed", "done", "success", "finished"):
                # Fetch session for the final report + citations
                try:
                    sr = client.get(f"{_BASE}/oracle/sessions/{session_id}")
                    sr.raise_for_status()
                    sess = sr.json()
                    return _format_oracle_answer(sess, job_id)
                except Exception:
                    return _format_oracle_answer(job_state, job_id)
            if status in ("failed", "error", "cancelled"):
                err = job_state.get("error") or job_state.get("message") or ""
                return f"# oracle job {job_id} ended with status={status}\n\n{err}"
            time.sleep(poll_every_s)

    return (f"# oracle timed out after {max_wait_s}s\n\nlast status: {last_status}\n"
            f"job: {job_id}  session: {session_id}\n\n"
            f"Retrieve later:  GET /oracle/sessions/{session_id}")


def _nia_search(query: str, mode: str = "query", limit: int = 5) -> str:
    """Unified Nia search. Modes: query (default), web, deep, universal."""
    if not query.strip():
        raise ToolError("query required")
    if mode not in ("query", "web", "deep", "universal"):
        raise ToolError(f"invalid mode {mode!r}, must be query/web/deep/universal")
    body = {"query": query, "mode": mode, "limit": max(1, min(limit, 20))}
    try:
        r = httpx.post(f"{_BASE}/search", headers=_headers(), json=body, timeout=60.0)
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError as e:
        raise ToolError(f"nia search failed: {e}")
    # Shape may include {results: [...]} with each having title/url/content/etc.
    results = data.get("results") or data.get("hits") or []
    if not results:
        return json.dumps(data, indent=2, default=str)[:8000]
    parts = [f"# nia search ({mode}): {query}"]
    for i, r in enumerate(results[:limit], 1):
        title = r.get("title") or r.get("name") or ""
        url = r.get("url") or r.get("source_url") or ""
        content = r.get("content") or r.get("snippet") or r.get("text") or ""
        parts.append(f"\n[{i}] {title}\n    {url}\n    {content[:500]}")
    return "\n".join(parts)[:25000]


register(Tool(
    name="nia_package_search",
    description="Semantic search inside a specific package's SOURCE CODE (not just docs). Best for 'how does X work in package Y' questions. registry is 'pypi', 'npm', 'crates', or 'go'.",
    parameters={
        "type": "object",
        "properties": {
            "package":  {"type": "string", "description": "Package name, e.g. 'pytest'."},
            "queries":  {"type": "array", "items": {"type": "string"}, "description": "1-5 semantic queries."},
            "registry": {"type": "string", "default": "pypi", "description": "pypi, npm, crates, or go."},
            "limit":    {"type": "integer", "default": 5},
        },
        "required": ["package", "queries"],
    },
    category="read",
    fn=_nia_package_search,
))


import os as _os  # noqa
if _os.environ.get("AGENT_ENABLE_ORACLE", "0") in ("1", "true", "yes"):
    register(Tool(
        name="nia_oracle_ask",
        description="Submit a research question to Nia Oracle. Disabled by default due to reliability issues.",
        parameters={
            "type": "object",
            "properties": {
                "query":      {"type": "string"},
                "max_wait_s": {"type": "integer", "default": 300},
            },
            "required": ["query"],
        },
        category="read",
        fn=_nia_oracle_ask,
    ))


register(Tool(
    name="nia_search",
    description="Unified Nia search. Modes: 'query' (default, fast search of indexed sources), 'web' (live web), 'deep' (more thorough, slower), 'universal' (everything).",
    parameters={
        "type": "object",
        "properties": {
            "query":  {"type": "string"},
            "mode":   {"type": "string", "default": "query", "enum": ["query", "web", "deep", "universal"]},
            "limit":  {"type": "integer", "default": 5},
        },
        "required": ["query"],
    },
    category="read",
    fn=_nia_search,
))
