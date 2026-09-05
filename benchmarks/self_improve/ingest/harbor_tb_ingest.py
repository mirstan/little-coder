"""Ingest harbor_adapter/tb_adapter run logs into NormalizedTrajectory
objects. This is the least-structured source (TDD_SPEC.md §0/§5): plain-text
per-task *.log files with `=== stop_reason: <value> ===` and
`>> tool(args)` lines, no structured JSON. Graceful degradation on
missing/malformed data is the core contract here -- never raise from load().
"""
import logging
import re
from pathlib import Path
from typing import Literal

from benchmarks.self_improve.schema import NormalizedTrajectory

logger = logging.getLogger(__name__)

_STOP_REASON_RE = re.compile(r"===\s*stop_reason:\s*(\S+)\s*===")
_TOOL_CALL_RE = re.compile(r"^>>\s+(.*)$", re.MULTILINE)


def load(log_root: Path, benchmark: Literal["harbor", "tb"]) -> list[NormalizedTrajectory]:
    trajectories = []
    for log_path in sorted(Path(log_root).glob("*.log")):
        traj = _load_one(log_path, benchmark)
        if traj is not None:
            trajectories.append(traj)
    return trajectories


def _load_one(log_path: Path, benchmark: Literal["harbor", "tb"]) -> NormalizedTrajectory | None:
    try:
        text = log_path.read_text(encoding="utf-8", errors="strict")
    except (UnicodeDecodeError, OSError) as e:
        logger.warning("harbor_tb_ingest: skipping unreadable %s: %s", log_path, e)
        return None

    try:
        m = _STOP_REASON_RE.search(text)
        stop_reason = m.group(1) if m else "unknown"
        tool_lines = _TOOL_CALL_RE.findall(text)
        turn_count = len(tool_lines)
        success = stop_reason == "agent_end"
        summarized = "\n".join(f">> {line}" for line in tool_lines)[:8_000]

        return NormalizedTrajectory(
            benchmark=benchmark,
            task_id=log_path.stem,
            success=success,
            stop_reason=stop_reason,
            turn_count=turn_count,
            partial_score=1.0 if success else 0.0,
            components_used=[],
            failure_signals=[] if success else ["approximated_from_stop_reason"],
            summarized_transcript=summarized,
            raw_paths={"log": str(log_path)},
        )
    except Exception as e:
        logger.warning("harbor_tb_ingest: failed to parse %s: %s", log_path, e)
        return None
