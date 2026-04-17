"""
Plan-time doc pre-fetch.

Before the planner decomposes a goal, scan the goal text for library names
(and any files mentioned) and proactively build a context block with:
  - Known library docs (from prior Context7 seeds in memory)
  - Fresh Context7 fetches for libraries that aren't cached
  - First 40 lines of any file path mentioned (for project-grounding)
"""
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Iterable

from ..memory import Memory


# Common library names the planner might see. Extend as needed.
KNOWN_LIBS = {
    "pytest", "unittest", "mock", "langgraph", "langchain", "openai",
    "pyembroidery", "numpy", "pandas", "opencv", "cv2", "PIL", "pillow",
    "flask", "fastapi", "httpx", "requests", "aiohttp",
    "sqlalchemy", "sqlite", "sqlite_vec", "redis",
    "pydantic", "dataclasses", "asyncio",
    "torch", "transformers", "vllm", "huggingface_hub",
    "pytest-asyncio", "pytest-cov", "hypothesis",
    "flutter", "dart", "hive",
}


FILE_PATH_RE = re.compile(r"(/[\w\-\./]+\.(?:py|md|ts|tsx|js|dart|yaml|yml|toml|json))")
LIB_HINT_RE  = re.compile(r"\b([a-zA-Z][a-zA-Z0-9_\-]{2,})\b")


def extract_libs_from_goal(goal: str) -> list[str]:
    """Pull library-sounding words from goal + filter by KNOWN_LIBS."""
    hits = set()
    for m in LIB_HINT_RE.finditer(goal.lower()):
        word = m.group(1)
        if word in KNOWN_LIBS:
            hits.add(word)
    return sorted(hits)


def extract_files_from_goal(goal: str) -> list[str]:
    return FILE_PATH_RE.findall(goal)


def memory_lookup_libraries(libs: Iterable[str]) -> dict[str, str]:
    """For each lib, try to find a prior Context7-seeded lesson in memory (semantic recall)."""
    mem = Memory()
    found: dict[str, str] = {}
    for lib in libs:
        hits = mem.recall_lessons(f"library {lib} documentation API", k=3)
        for h in hits:
            if f"library:{lib}" in (h.get("tags") or []):
                found[lib] = h["text"]
                break
            # Fallback: the lesson text often starts with "Library [name]"
            if h["text"].startswith(f"Library [{lib}]"):
                found[lib] = h["text"]
                break
    return found


def live_fetch_libraries(libs: Iterable[str], tokens: int = 2500) -> dict[str, str]:
    """Call Context7 directly for any libs not already in memory."""
    if not os.environ.get("CONTEXT7_API_KEY"):
        return {}
    from ..tools.base import REGISTRY
    from ..tools.base import ToolError
    resolve = REGISTRY.get("context7_resolve").fn
    docs = REGISTRY.get("context7_docs").fn
    out: dict[str, str] = {}
    for lib in libs:
        try:
            ro = resolve(name=lib, limit=2)
            m = re.search(r"id=(\S+)\s+trust=", ro)
            if not m:
                continue
            d = docs(library_id=m.group(1), tokens=tokens)
            out[lib] = d[:8000]
        except ToolError:
            continue
        except Exception:
            continue
    return out


def file_previews(paths: Iterable[str], max_lines: int = 40) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in paths:
        fp = Path(p)
        if not fp.exists() or not fp.is_file():
            continue
        try:
            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()[:max_lines]
            numbered = [f"{i+1:5d} | {ln}" for i, ln in enumerate(lines)]
            out[p] = "\n".join(numbered)
        except Exception:
            continue
    return out


def build_doc_context(goal: str) -> str:
    """Produce a 'PRE-FETCHED CONTEXT' block to inject into the planner's user message."""
    libs = extract_libs_from_goal(goal)
    files = extract_files_from_goal(goal)

    memo_libs = memory_lookup_libraries(libs) if libs else {}
    fresh_libs = {k: v for k, v in live_fetch_libraries([l for l in libs if l not in memo_libs]).items()}
    previews = file_previews(files)

    parts: list[str] = []
    if memo_libs or fresh_libs:
        parts.append("=== KNOWN LIBRARY DOCS (from Context7) ===")
        for lib, text in {**memo_libs, **fresh_libs}.items():
            source = "memory" if lib in memo_libs else "live"
            parts.append(f"\n--- {lib} ({source}) ---\n{text[:4000]}")
    if previews:
        parts.append("\n=== RELEVANT FILES (first lines) ===")
        for path, excerpt in previews.items():
            parts.append(f"\n--- {path} ---\n{excerpt}")

    return "\n".join(parts) if parts else ""
