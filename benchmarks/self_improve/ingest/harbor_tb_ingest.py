"""Ingest harbor_adapter/tb_adapter run logs into NormalizedTrajectory
objects.

harbor and tb have GENUINELY DIFFERENT real structures -- confirmed by
actually running both real CLIs (`harbor run` / `tb run`, hello-world task,
pi routed through fake_pi.py via LITTLE_CODER_PI_BIN_OVERRIDE), not assumed
from one shared format. Dispatched by the `benchmark` argument, since the
caller already knows which one it is.

tb (classic terminal-bench), see tests/fixtures/real_tb_run/:
  <log_root>/<task_id>/<task_id>.N-of-M.<ts>/results.json     (plural) per-trial: is_resolved
  <log_root>/<task_id>/<task_id>.N-of-M.<ts>/agent-logs/*.log little-coder's own log:
    === stop_reason: X === / === assistant text === / >> tool(args)

harbor (Terminal-Bench 2.0), see tests/fixtures/real_harbor_run/:
  <log_root>/<trial_name>/result.json     (singular) per-trial, richer:
    agent_result.metadata: {stop_reason, n_tool_calls, n_turns, n_compactions, n_notifications}
    verifier_result.rewards.reward: float -- GROUND TRUTH (0.0/1.0, not a boolean)
    task_name: "org/name"
  <log_root>/<trial_name>/agent/*.log     (singular "agent", not "agent-logs")
  harbor's structured metadata is read directly -- no log regex needed for
  stop_reason/turn_count, unlike tb where it's the only source.

Graceful degradation is the core contract for both: never raise from load()
on missing/malformed data.
"""
import json
import logging
import re
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from benchmarks.self_improve.schema import NormalizedTrajectory

logger = logging.getLogger(__name__)

_STOP_REASON_RE = re.compile(r"===\s*stop_reason:\s*(\S+)\s*===")
_TOOL_CALL_RE = re.compile(r"^>>\s+(.*)$", re.MULTILINE)


def load(log_root: Path, benchmark: Literal["harbor", "tb"]) -> list[NormalizedTrajectory]:
    if benchmark == "harbor":
        return _load_harbor(log_root)
    return _load_tb(log_root, benchmark)


# ── harbor ────────────────────────────────────────────────────────────────

def _load_harbor(log_root: Path) -> list[NormalizedTrajectory]:
    log_root = Path(log_root)
    if not log_root.is_dir():
        logger.warning("harbor_tb_ingest: log_root does not exist or is not a directory: %s", log_root)
        return []

    trajectories = []
    for trial_dir in sorted(log_root.iterdir()):
        if not trial_dir.is_dir():
            continue
        traj = _load_harbor_trial(trial_dir)
        if traj is not None:
            trajectories.append(traj)
    return trajectories


def _load_harbor_trial(trial_dir: Path) -> NormalizedTrajectory | None:
    result_path = trial_dir / "result.json"
    if not result_path.exists():
        return None

    try:
        result = json.loads(result_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("harbor_tb_ingest: malformed result.json at %s: %s", result_path, e)
        return None

    if not isinstance(result, dict):
        # Real gap, confirmed by review: this module's own docstring promises
        # "never raise from load() on missing/malformed data" -- valid JSON
        # with a non-object ROOT (e.g. a bare list) made every .get() below
        # raise AttributeError. Validated explicitly at each level instead of
        # relying on a broad except around the whole function, which risked
        # masking a genuine coding bug as "malformed data".
        logger.warning("harbor_tb_ingest: result.json root is not an object in %s", result_path)
        return None

    # `or trial_dir.name`, not .get(key, default): guards against an
    # explicit JSON null (not just a missing key) -- .get(k, default) only
    # substitutes the default when the key is ABSENT.
    task_id = result.get("task_name") or trial_dir.name

    verifier_result = result.get("verifier_result")
    if not isinstance(verifier_result, dict):
        verifier_result = {}
    rewards = verifier_result.get("rewards")
    if not isinstance(rewards, dict):
        rewards = {}
    reward = rewards.get("reward")
    try:
        success = reward is not None and float(reward) > 0.0
    except (TypeError, ValueError):
        logger.warning("harbor_tb_ingest: non-numeric reward %r in %s", reward, result_path)
        success = False

    agent_result = result.get("agent_result")
    if not isinstance(agent_result, dict):
        agent_result = {}
    metadata = agent_result.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    stop_reason = metadata.get("stop_reason") or "unknown"
    turn_count = metadata.get("n_turns", 0) or 0

    raw_paths = {"result": str(result_path)}
    summarized = ""
    log_dir = trial_dir / "agent"
    if log_dir.is_dir():
        # sorted(): glob() order is filesystem-dependent: pick the same log
        # deterministically across runs when a trial has more than one.
        log_paths = sorted(log_dir.glob("*.log"))
        if log_paths:
            raw_paths["log"] = str(log_paths[0])
            try:
                text = log_paths[0].read_text(encoding="utf-8", errors="strict")
                tool_lines = _TOOL_CALL_RE.findall(text)
                summarized = "\n".join(f">> {line}" for line in tool_lines)[:8_000]
            except (UnicodeDecodeError, OSError) as e:
                logger.warning("harbor_tb_ingest: unreadable log %s: %s", log_paths[0], e)

    try:
        return NormalizedTrajectory(
            benchmark="harbor",
            task_id=task_id,
            success=success,
            stop_reason=stop_reason,
            turn_count=turn_count,
            partial_score=1.0 if success else 0.0,
            components_used=[],
            failure_signals=[] if success else ["reward_zero"],
            summarized_transcript=summarized,
            raw_paths=raw_paths,
        )
    except ValidationError as e:
        # Real gap, confirmed by review: a wrong-TYPED (not missing/null)
        # value that survives the isinstance guards above (e.g. task_name is
        # a non-string truthy value) fails NormalizedTrajectory's own field
        # validation -- narrowly caught here, at the one place it's actually
        # expected, rather than a broad except around the whole function
        # that could also mask a genuine coding bug.
        logger.warning("harbor_tb_ingest: invalid trajectory fields in %s: %s", result_path, e)
        return None


# ── tb (classic terminal-bench) ──────────────────────────────────────────

def _load_tb(log_root: Path, benchmark: Literal["harbor", "tb"]) -> list[NormalizedTrajectory]:
    log_root = Path(log_root)
    if not log_root.is_dir():
        logger.warning("harbor_tb_ingest: log_root does not exist or is not a directory: %s", log_root)
        return []

    trajectories = []
    for task_dir in sorted(log_root.iterdir()):
        if not task_dir.is_dir():
            continue
        for trial_dir in sorted(task_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            traj = _load_tb_trial(trial_dir, benchmark)
            if traj is not None:
                trajectories.append(traj)
    return trajectories


def _load_tb_trial(trial_dir: Path, benchmark: Literal["harbor", "tb"]) -> NormalizedTrajectory | None:
    results_path = trial_dir / "results.json"
    if not results_path.exists():
        return None

    try:
        result = json.loads(results_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("harbor_tb_ingest: malformed results.json at %s: %s", results_path, e)
        return None

    if not isinstance(result, dict):
        # Same gap as _load_harbor_trial -- see its comment.
        logger.warning("harbor_tb_ingest: results.json root is not an object in %s", results_path)
        return None

    task_id = result.get("task_id") or trial_dir.parent.name
    is_resolved = result.get("is_resolved", False)
    if not isinstance(is_resolved, bool):
        # Hardening, confirmed by review: real captured data has this as a
        # genuine JSON boolean (tests/fixtures/real_tb_run), but bool(...) on
        # any OTHER type is a classic Python trap -- bool("false") is True.
        # Treat a wrong-typed value as unresolved/failed rather than trust it.
        logger.warning("harbor_tb_ingest: non-boolean is_resolved %r in %s, treating as unresolved",
                        is_resolved, results_path)
        is_resolved = False
    success = is_resolved

    # sorted(): glob() order is filesystem-dependent: pick the same log
    # deterministically across runs when a trial has more than one.
    log_paths = sorted((trial_dir / "agent-logs").glob("*.log")) if (trial_dir / "agent-logs").is_dir() else []
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

    try:
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
    except ValidationError as e:
        # Same gap as _load_harbor_trial -- see its comment.
        logger.warning("harbor_tb_ingest: invalid trajectory fields in %s: %s", results_path, e)
        return None
