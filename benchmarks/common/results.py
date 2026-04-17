"""JSONL result writer/reader for benchmark predictions + summaries."""
from __future__ import annotations
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator

from .adapter import Prediction
from .runner import RunResult


class ResultsWriter:
    """Append-only JSONL writer. One line per task."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, run: RunResult, pred: Prediction) -> None:
        row = {
            "task_id": pred.task_id,
            "prediction": pred.prediction,
            "anvil_stats": pred.anvil_stats,
            "run": {
                "success": run.success,
                "wall_sec": round(run.wall_sec, 2),
                "cycles_used": run.cycles_used,
                "error": run.error,
                "stderr_tail": run.stderr_tail[-500:],
            },
            "error": pred.error,
        }
        self._fh.write(json.dumps(row, default=_json_default) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "ResultsWriter":
        return self

    def __exit__(self, *args) -> None:
        self.close()


class ResultsReader:
    """Iterator over a results JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)

    def task_ids(self) -> set[str]:
        return {row["task_id"] for row in self}


def _json_default(o: Any) -> Any:
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, bytes):
        return o.decode("utf-8", errors="replace")
    raise TypeError(f"not JSON-serialisable: {type(o).__name__}")
