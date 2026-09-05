import json

import pytest

from benchmarks.self_improve.ingest import aider_polyglot_ingest


def _write_trajectory(path, **fields):
    base = {
        "attempt": "1", "agent_ended": True, "turn_count": 1,
        "compaction_events": 0, "assistant_text": "", "tool_calls": [],
    }
    base.update(fields)
    path.write_text(json.dumps(base))


@pytest.fixture
def aider_run(tmp_path):
    log_root = tmp_path / "logs"
    two_fer = log_root / "python" / "two-fer"
    two_fer.mkdir(parents=True)
    _write_trajectory(
        two_fer / "trajectory_1.json",
        turn_count=4,
        tool_calls=[{"name": "write", "args": {}, "result_text": "ok", "is_error": False}],
    )
    (two_fer / "final_output_1.txt").write_text("1 passed")

    leap = log_root / "python" / "leap"
    leap.mkdir(parents=True)
    _write_trajectory(leap / "trajectory_1.json", agent_ended=False, turn_count=1)
    _write_trajectory(leap / "trajectory_2.json", turn_count=3)
    (leap / "final_output_2.txt").write_text("FAILED test_leap")

    results_json = tmp_path / "results_full_polyglot.json"
    results_json.write_text(json.dumps({"exercises": {
        "python/two-fer": {
            "status": "pass_1", "stop_reason_1": "agent_end", "stop_reason_2": None,
            "turn_count": 4,
        },
        "python/leap": {
            "status": "fail", "stop_reason_1": "deadline", "stop_reason_2": "agent_end",
            "turn_count": 4,
        },
    }}))

    return {"log_root": log_root, "results_json": results_json}


def test_load_one_trajectory_per_exercise_using_latest_attempt(aider_run):
    """One NormalizedTrajectory per exercise (not per attempt file) -- the
    exercise's FINAL recorded status is what matters for scoring; attempt_1
    of a two-attempt exercise must not each become a separate example."""
    trajs = aider_polyglot_ingest.load(aider_run["log_root"], aider_run["results_json"])
    assert {t.task_id for t in trajs} == {"python/two-fer", "python/leap"}


def test_load_maps_status_to_success_and_partial_score(aider_run):
    trajs = {t.task_id: t for t in aider_polyglot_ingest.load(aider_run["log_root"], aider_run["results_json"])}
    assert trajs["python/two-fer"].success is True
    assert trajs["python/two-fer"].partial_score == 1.0
    assert trajs["python/leap"].success is False
    assert trajs["python/leap"].partial_score == 0.0


def test_load_pass_2_gets_partial_credit_score(tmp_path):
    """A status of pass_2 (fixed on retry) scores 0.7, per the metric design
    in the plan -- distinguishes a clean first-try pass from a
    needed-a-nudge pass without conflating either with a hard failure."""
    log_root = tmp_path / "logs"
    ex = log_root / "python" / "ex"
    ex.mkdir(parents=True)
    _write_trajectory(ex / "trajectory_1.json")
    _write_trajectory(ex / "trajectory_2.json")
    results_json = tmp_path / "results_full_polyglot.json"
    results_json.write_text(json.dumps({"exercises": {
        "python/ex": {"status": "pass_2", "stop_reason_1": "agent_end",
                       "stop_reason_2": "agent_end", "turn_count": 2},
    }}))
    trajs = aider_polyglot_ingest.load(log_root, results_json)
    assert trajs[0].success is True
    assert trajs[0].partial_score == 0.7


def test_load_uses_stop_reason_2_when_present_else_stop_reason_1(aider_run):
    trajs = {t.task_id: t for t in aider_polyglot_ingest.load(aider_run["log_root"], aider_run["results_json"])}
    assert trajs["python/leap"].stop_reason == "agent_end"      # stop_reason_2
    assert trajs["python/two-fer"].stop_reason == "agent_end"   # only stop_reason_1 exists


def test_load_components_used_is_empty_pending_notification_capture(aider_run):
    """Confirmed gap (see TDD_SPEC.md §0): aider_polyglot trajectories do not
    currently carry notifications, so components_used MUST be empty, not
    fabricated or inferred from tool_calls. This test locks in the gap so a
    future fix to aider_polyglot.py's _dump_trajectory is a deliberate,
    visible change to this test's expectation -- not a silent behavior
    drift."""
    for t in aider_polyglot_ingest.load(aider_run["log_root"], aider_run["results_json"]):
        assert t.components_used == []


def test_load_skips_exercise_missing_from_results_json(tmp_path):
    """An exercise dir exists on disk (e.g. from a stale/aborted run) but has
    no entry in results_full_polyglot.json -- must be skipped, not crash."""
    log_root = tmp_path / "logs"
    ex = log_root / "python" / "orphan"
    ex.mkdir(parents=True)
    _write_trajectory(ex / "trajectory_1.json")
    results_json = tmp_path / "results_full_polyglot.json"
    results_json.write_text(json.dumps({"exercises": {}}))
    trajs = aider_polyglot_ingest.load(log_root, results_json)
    assert trajs == []


def test_load_missing_results_json_raises_clear_error(tmp_path):
    """Distinguish 'no results yet' (caller error, should be loud) from
    'exercise not yet scored' (expected, silent skip)."""
    with pytest.raises(FileNotFoundError):
        aider_polyglot_ingest.load(tmp_path, tmp_path / "does_not_exist.json")
