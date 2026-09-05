"""Ingest harbor_adapter/tb_adapter run logs into NormalizedTrajectory
objects.

Real structure, confirmed by actually running `tb run` (hello-world task,
pi routed through fake_pi.py via LITTLE_CODER_PI_BIN_OVERRIDE) -- see
tests/fixtures/real_tb_run/:
  <log_root>/results.json                                    run-level aggregate (ignored)
  <log_root>/<task_id>/<task_id>.N-of-M.<ts>/results.json     per-trial: is_resolved (ground truth)
  <log_root>/<task_id>/<task_id>.N-of-M.<ts>/agent-logs/*.log little-coder's own log:
    === stop_reason: X ===
    === assistant text ===
    >> tool(args)
    << result

is_resolved is the ground-truth success signal -- NOT stop_reason. A run
can report stop_reason=agent_end (pi finished the turn) while the task
itself still failed (confirmed: fake_pi's "clean" mode always reports
agent_end, but the real terminal-bench test suite found hello.txt was never
created). Graceful degradation is the core contract: never raise from
load() on a missing/malformed trial.
"""
import json
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
    for task_dir in sorted(Path(log_root).iterdir()):
        if not task_dir.is_dir():
            continue
        for trial_dir in sorted(task_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            traj = _load_trial(trial_dir, benchmark)
            if traj is not None:
                trajectories.append(traj)
    return trajectories


def _load_trial(trial_dir: Path, benchmark: Literal["harbor", "tb"]) -> NormalizedTrajectory | None:
    results_path = trial_dir / "results.json"
    if not results_path.exists():
        return None

    try:
        result = json.loads(results_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("harbor_tb_ingest: malformed results.json at %s: %s", results_path, e)
        return None

    task_id = result.get("task_id", trial_dir.parent.name)
    success = bool(result.get("is_resolved", False))

    log_paths = list((trial_dir / "agent-logs").glob("*.log")) if (trial_dir / "agent-logs").is_dir() else []
    stop_reason = "unknown"
    turn_count = 0
    summarized = ""
    raw_paths = {"results": str(results_path)}

    if log_paths:
        log_path = log_paths[0]
        raw_paths["log"] = str(log_path)
        try:
            text = log_path.read_text(encoding="utf-8", errors="strict")
            m = _STOP_REASON_RE.search(text)
            stop_reason = m.group(1) if m else "unknown"
            tool_lines = _TOOL_CALL_RE.findall(text)
            turn_count = len(tool_lines)
            summarized = "\n".join(f">> {line}" for line in tool_lines)[:8_000]
        except (UnicodeDecodeError, OSError) as e:
            logger.warning("harbor_tb_ingest: unreadable log %s: %s", log_path, e)

    return NormalizedTrajectory(
        benchmark=benchmark,
        task_id=task_id,
        success=success,
        stop_reason=stop_reason,
        turn_count=turn_count,
        partial_score=1.0 if success else 0.0,
        components_used=[],
        failure_signals=[] if success else ["is_resolved_false"],
        summarized_transcript=summarized,
        raw_paths=raw_paths,
    )
