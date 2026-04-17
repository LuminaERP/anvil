"""CLI entrypoint for SWE-bench (Lite by default)."""
from __future__ import annotations
import sys
import argparse
from .adapter import SWEBenchAdapter
from ..common.cli import dispatch


def main() -> int:
    # Extract --split before the common CLI chews on argv
    split = "lite"
    remaining = []
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == "--split":
            split = argv[i + 1]
            i += 2
        elif argv[i].startswith("--split="):
            split = argv[i].split("=", 1)[1]
            i += 1
        else:
            remaining.append(argv[i])
            i += 1

    adapter = SWEBenchAdapter(split=split)
    return dispatch(f"swebench_{split}", adapter, argv=remaining)


if __name__ == "__main__":
    sys.exit(main())
