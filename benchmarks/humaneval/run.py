"""CLI entrypoint for the HumanEval benchmark."""
from __future__ import annotations
import sys
from .adapter import HumanEvalAdapter
from ..common.cli import dispatch


def main() -> int:
    return dispatch("humaneval", HumanEvalAdapter())


if __name__ == "__main__":
    sys.exit(main())
