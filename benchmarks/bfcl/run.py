"""CLI entrypoint for BFCL."""
from __future__ import annotations
import sys
from .adapter import BFCLAdapter
from ..common.cli import dispatch


def main() -> int:
    return dispatch("bfcl", BFCLAdapter())


if __name__ == "__main__":
    sys.exit(main())
