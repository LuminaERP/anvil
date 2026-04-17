"""CLI entrypoint for BigCodeBench."""
from __future__ import annotations
import sys
from .adapter import BigCodeBenchAdapter
from ..common.cli import dispatch


def main() -> int:
    # Default to 'complete' subset. For 'instruct' use: --end-to-end with a different adapter.
    return dispatch("bigcodebench", BigCodeBenchAdapter(subset="complete"))


if __name__ == "__main__":
    sys.exit(main())
