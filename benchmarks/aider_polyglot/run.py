"""CLI entrypoint for Aider Polyglot."""
from __future__ import annotations
import sys
from .adapter import AiderPolyglotAdapter
from ..common.cli import dispatch


def main() -> int:
    return dispatch("aider_polyglot", AiderPolyglotAdapter())


if __name__ == "__main__":
    sys.exit(main())
