"""
Memory pre-seeder.

Scans a workspace for dependency manifests (requirements.txt, pyproject.toml,
package.json, pubspec.yaml), resolves each package via Context7, pulls a docs
summary, stores it as a high-priority Lesson with tag 'library:<name>'.

Also scans top-level Python imports as a fallback when no manifest exists.

Usage:
  python -m autonomous.seed_docs --workspace /path/to/project
  python -m autonomous.seed_docs --workspace . --extra pytest,langgraph
  python -m autonomous.seed_docs --list
"""
from __future__ import annotations
import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable

# Make sure tool side-effect registers CONTEXT7 key check
from .tools import context7 as _c7  # noqa: F401
from .tools.base import REGISTRY
from .memory import Memory
from .state import Lesson


SKIP_NAMES = {
    # stdlib-ish / non-lib
    "python", "pip", "setuptools", "wheel", "pkg_resources",
    # too generic or platform-specific
    "os", "sys", "json", "re", "time", "datetime", "pathlib", "argparse",
    "typing", "subprocess", "functools", "itertools", "collections",
}


def _parse_requirements(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # "pkg==1.0", "pkg>=1.0", "pkg[extra]==1.0"
        m = re.match(r"^([A-Za-z0-9_.\-]+)", line)
        if m:
            out.append(m.group(1).lower())
    return out


def _parse_pyproject(path: Path) -> list[str]:
    text = path.read_text(errors="replace")
    out = set()
    # [tool.poetry.dependencies]
    for m in re.finditer(r"^\s*([A-Za-z0-9_.\-]+)\s*=\s*[\"'\^\>\<\~\d]", text, re.MULTILINE):
        name = m.group(1).lower()
        if name not in ("python",):
            out.add(name)
    # dependencies = ["pkg>=1.0", ...]
    for m in re.finditer(r"[\"']([A-Za-z0-9_.\-]+)\s*[\[=>~<,;]", text):
        out.add(m.group(1).lower())
    return sorted(out)


def _parse_package_json(path: Path) -> list[str]:
    import json
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    out = set()
    for key in ("dependencies", "devDependencies"):
        out.update((data.get(key) or {}).keys())
    return sorted(n.lower() for n in out)


def _parse_pubspec(path: Path) -> list[str]:
    text = path.read_text(errors="replace")
    out = set()
    in_deps = False
    for line in text.splitlines():
        stripped = line.rstrip()
        if stripped.startswith("dependencies:") or stripped.startswith("dev_dependencies:"):
            in_deps = True; continue
        if in_deps and line and not line.startswith(" "):
            in_deps = False
        if in_deps:
            m = re.match(r"^\s{2}([A-Za-z0-9_\-]+)\s*:", line)
            if m and m.group(1).lower() not in ("flutter", "sdk"):
                out.add(m.group(1).lower())
    return sorted(out)


def _scan_python_imports(workspace: Path, max_files: int = 80) -> list[str]:
    imports: set[str] = set()
    for i, p in enumerate(workspace.rglob("*.py")):
        if i >= max_files:
            break
        try:
            txt = p.read_text(errors="replace")
        except Exception:
            continue
        for m in re.finditer(r"^(?:from\s+([A-Za-z0-9_\.]+)|import\s+([A-Za-z0-9_\.]+))", txt, re.MULTILINE):
            name = (m.group(1) or m.group(2)).split(".")[0].lower()
            imports.add(name)
    return sorted(imports - SKIP_NAMES)


def discover_libs(workspace: Path) -> list[str]:
    libs: set[str] = set()
    for name, parser in [
        ("requirements.txt", _parse_requirements),
        ("pyproject.toml",   _parse_pyproject),
        ("package.json",     _parse_package_json),
        ("pubspec.yaml",     _parse_pubspec),
    ]:
        f = workspace / name
        if f.exists():
            libs.update(parser(f))
    # Also scan nested (up to 1 level deep) — many projects have /backend/requirements.txt etc.
    for sub in workspace.iterdir() if workspace.is_dir() else []:
        if sub.is_dir() and not sub.name.startswith("."):
            for name, parser in [
                ("requirements.txt", _parse_requirements),
                ("pyproject.toml", _parse_pyproject),
                ("package.json", _parse_package_json),
                ("pubspec.yaml", _parse_pubspec),
            ]:
                f = sub / name
                if f.exists():
                    libs.update(parser(f))

    # Fallback: scan Python imports
    if not libs and (workspace / "*.py").parent.exists():
        libs.update(_scan_python_imports(workspace))

    return sorted(libs - SKIP_NAMES)


def seed_library(mem: Memory, name: str, session_id: str, tokens: int = 3500) -> dict:
    """Resolve + fetch docs for one library; write as a Lesson."""
    resolve = REGISTRY.get("context7_resolve").fn
    docs = REGISTRY.get("context7_docs").fn
    result = {"name": name, "status": "pending"}

    try:
        resolve_out = resolve(name=name, limit=3)
    except Exception as e:
        result.update(status="resolve_failed", error=str(e)); return result

    # Parse first hit from resolve output (format "[1] id=<id>  trust=...")
    m = re.search(r"id=(\S+)\s+trust=([\d\.]+)", resolve_out)
    if not m:
        result.update(status="no_match"); return result
    lib_id, trust = m.group(1), m.group(2)

    try:
        doc_text = docs(library_id=lib_id, tokens=tokens)
    except Exception as e:
        result.update(status="docs_failed", error=str(e), library_id=lib_id); return result

    # Trim intro if needed
    head = doc_text[:6000]
    lesson_text = (
        f"Library [{name}] (context7 id={lib_id}, trust={trust}):\n"
        f"{head}\n"
        f"(Docs pulled from Context7 at seed time. Always call context7_docs for current details.)"
    )
    try:
        mem.add_lesson(session_id, Lesson(
            text=lesson_text[:8000],  # SQLite soft-cap
            severity="info",
            tags=["library", f"library:{name}", "context7-seed"],
        ))
        result.update(status="seeded", library_id=lib_id, trust=trust, bytes=len(lesson_text))
    except Exception as e:
        result.update(status="memory_failed", error=str(e))
    return result


def main() -> int:
    p = argparse.ArgumentParser(prog="autonomous.seed_docs")
    p.add_argument("--workspace", default=".")
    p.add_argument("--extra", default="", help="Comma-separated extra library names to seed.")
    p.add_argument("--list", action="store_true", help="Just print discovered libs and exit.")
    p.add_argument("--tokens", type=int, default=3500)
    args = p.parse_args()

    ws = Path(args.workspace).resolve()
    if not os.environ.get("CONTEXT7_API_KEY"):
        print("ERROR: set CONTEXT7_API_KEY env var first", file=sys.stderr); return 2

    libs = discover_libs(ws)
    if args.extra:
        libs.extend(x.strip().lower() for x in args.extra.split(",") if x.strip())
    libs = sorted(set(libs))

    if not libs:
        print("no libraries discovered", file=sys.stderr); return 1

    print(f"discovered {len(libs)} libs in {ws}:", file=sys.stderr)
    for l in libs:
        print(f"  - {l}", file=sys.stderr)
    if args.list:
        return 0

    mem = Memory()
    session_id = mem.start_session(f"seed_docs from {ws}")
    ok = 0
    for name in libs:
        res = seed_library(mem, name, session_id, tokens=args.tokens)
        mark = "OK" if res["status"] == "seeded" else "SKIP"
        extra = f" id={res.get('library_id','?')} trust={res.get('trust','?')}" if res["status"] == "seeded" else f" ({res['status']})"
        print(f"  [{mark}] {name}{extra}", file=sys.stderr)
        if res["status"] == "seeded":
            ok += 1
        # be kind to the API
        time.sleep(0.25)
    mem.finish_session(session_id, f"seeded {ok}/{len(libs)}")
    print(f"\ndone — seeded {ok}/{len(libs)} libraries into memory", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
