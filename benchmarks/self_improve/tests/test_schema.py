import pytest
from pydantic import ValidationError

from benchmarks.self_improve.schema import ComponentUsage, NormalizedTrajectory


def test_component_usage_minimal_construction():
    """ComponentUsage requires only pred_name; other fields have sane defaults."""
    u = ComponentUsage(pred_name="skills_tools_bash")
    assert u.invocation_count == 0
    assert u.was_error_context is False


def test_component_usage_rejects_negative_count():
    """invocation_count must be >= 0 -- a negative count is a parsing bug, not data."""
    with pytest.raises(ValidationError):
        ComponentUsage(pred_name="x", invocation_count=-1)


def test_normalized_trajectory_minimal_construction():
    """Only benchmark/task_id/success/stop_reason/turn_count are required;
    everything else defaults to empty/None so a degraded ingest source can
    still produce a valid object."""
    t = NormalizedTrajectory(
        benchmark="harbor", task_id="t1", success=False,
        stop_reason="deadline", turn_count=3,
    )
    assert t.partial_score is None
    assert t.components_used == []
    assert t.failure_signals == []
    assert t.summarized_transcript == ""
    assert t.raw_paths == {}


def test_normalized_trajectory_rejects_unknown_benchmark():
    with pytest.raises(ValidationError):
        NormalizedTrajectory(
            benchmark="not_a_real_benchmark", task_id="t1",
            success=True, stop_reason="agent_end", turn_count=1,
        )


def test_normalized_trajectory_partial_score_bounds():
    """partial_score, when present, must be in [0.0, 1.0] -- it feeds directly
    into the metric's base_score and an out-of-range value would silently
    corrupt GEPA's Pareto comparisons."""
    with pytest.raises(ValidationError):
        NormalizedTrajectory(
            benchmark="aider_polyglot", task_id="t1", success=True,
            stop_reason="agent_end", turn_count=1, partial_score=1.5,
        )


def test_normalized_trajectory_is_json_roundtrippable():
    """Ingestion modules persist a normalized cache to disk between runs
    (see run_gepa.py's --log-roots caching) -- must survive a dump/load cycle
    with identical field values, including nested ComponentUsage list."""
    t = NormalizedTrajectory(
        benchmark="gaia", task_id="t1", success=True, stop_reason="agent_end",
        turn_count=5, components_used=[ComponentUsage(pred_name="x", invocation_count=2)],
    )
    restored = NormalizedTrajectory.model_validate_json(t.model_dump_json())
    assert restored == t
