"""CLI entrypoint for LiveCodeBench."""
from __future__ import annotations
import sys
from .adapter import LiveCodeBenchAdapter
from ..common.cli import dispatch


def main() -> int:
    return dispatch("livecodebench", LiveCodeBenchAdapter())


if __name__ == "__main__":
    sys.exit(main())
