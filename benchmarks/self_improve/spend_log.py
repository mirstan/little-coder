"""Append-only, flush-per-line audit log of every live invocation and GEPA
iteration during a real live-eval run -- survives a SIGKILL up to the last
completed exercise, and is the only way to answer "did the run actually
accept anything, and why" after the fact. The old frozen-data design could
never answer that question (GEPA could never accept anything, and there was
no record of individual live runs to inspect since there were none).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class SpendLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a")

    def _write(self, record: dict[str, Any]) -> None:
        record.setdefault("ts", time.time())
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()

    def run_start(self, **fields: Any) -> None:
        self._write({"event": "run_start", **fields})

    def estimate(self, **fields: Any) -> None:
        self._write({"event": "estimate", **fields})

    def exercise(self, **fields: Any) -> None:
        self._write({"event": "exercise", **fields})

    def iteration_end(self, **fields: Any) -> None:
        self._write({"event": "iteration_end", **fields})

    def run_end(self, **fields: Any) -> None:
        self._write({"event": "run_end", **fields})

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "SpendLog":
        return self

    def __exit__(self, *_exc) -> bool:
        self.close()
        return False


def _read_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a truncated final line from a SIGKILL is a skip, not a crash
    return records


def summarize(path: Path) -> dict[str, Any]:
    """Totals for a spend_log.jsonl: exercise count, memo hit rate,
    per-status breakdown, cumulative live wall clock."""
    records = _read_records(path)
    exercises = [r for r in records if r.get("event") == "exercise"]
    total = len(exercises)
    memo_hits = sum(1 for r in exercises if r.get("memo_hit"))
    by_status: dict[str, int] = {}
    total_wall_s = 0.0
    for r in exercises:
        status = r.get("status", "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        total_wall_s += r.get("duration_s") or 0
    return {
        "total_exercises": total,
        "memo_hits": memo_hits,
        "memo_hit_rate": (memo_hits / total) if total else 0.0,
        "by_status": by_status,
        "total_wall_s": total_wall_s,
    }
