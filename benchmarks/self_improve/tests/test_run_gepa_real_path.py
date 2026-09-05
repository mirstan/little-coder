"""Tests for run_gepa.py's real (non-dry-run) path, up to but NEVER including
the actual GEPA.compile() call -- that spends real API budget and must only
ever run on explicit human invocation with a real reflection_lm configured.
No test here sets REFLECTION_LM_API_KEY or passes --confirm-real-run with a
real model, so _real_run() always exits via the safety gate in these tests.
"""
import argparse
import os

import pytest

from benchmarks.self_improve.run_gepa import build_examples, split_train_val, _real_run
from benchmarks.self_improve.schema import NormalizedTrajectory


def _traj(task_id, benchmark="aider_polyglot", success=True):
    return NormalizedTrajectory(
        benchmark=benchmark, task_id=task_id, success=success,
        stop_reason="agent_end", turn_count=1,
        partial_score=1.0 if success else 0.0,
    )


def test_build_examples_carries_trajectory_and_task_context():
    traj = _traj("t1")
    examples = build_examples([traj])
    assert len(examples) == 1
    assert examples[0].trajectory is traj
    assert examples[0].inputs().task_context == traj.task_id  # no summarized_transcript set


def test_split_train_val_stratifies_by_benchmark_and_success():
    trajs = [_traj(f"a{i}", "aider_polyglot", True) for i in range(6)] + \
            [_traj(f"g{i}", "gaia", False) for i in range(4)]
    train, val = split_train_val(trajs, train_frac=0.5, seed=1)
    assert len(train) + len(val) == 10
    # both groups represented in train (not all of one benchmark dumped into val)
    train_benchmarks = {t.benchmark for t in train}
    assert train_benchmarks == {"aider_polyglot", "gaia"}


def test_split_train_val_is_deterministic_given_seed():
    trajs = [_traj(f"t{i}") for i in range(10)]
    train1, val1 = split_train_val(trajs, 0.7, seed=42)
    train2, val2 = split_train_val(trajs, 0.7, seed=42)
    assert [t.task_id for t in train1] == [t.task_id for t in train2]
    assert [t.task_id for t in val1] == [t.task_id for t in val2]


def test_split_train_val_handles_singleton_group():
    """A group with exactly one trajectory must not crash or vanish --
    reused for both train and val rather than producing an empty val."""
    trajs = [_traj("only", "tb", True)]
    train, val = split_train_val(trajs, 0.7, seed=1)
    assert len(train) == 1
    assert len(val) == 1


def _args(**overrides):
    defaults = dict(
        reflection_model=None, confirm_real_run=False,
        train_frac=0.7, seed=42, out_dir="/tmp/unused",
        allow_partial_ingest=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_real_run_refuses_without_reflection_model(monkeypatch):
    monkeypatch.delenv("REFLECTION_LM_API_KEY", raising=False)
    code = _real_run([_traj("t1")], {"agents_md": "body"}, _args())
    assert code == 1


def test_real_run_refuses_without_api_key_even_with_model_and_confirm(monkeypatch):
    monkeypatch.delenv("REFLECTION_LM_API_KEY", raising=False)
    code = _real_run(
        [_traj("t1")], {"agents_md": "body"},
        _args(reflection_model="some/model", confirm_real_run=True),
    )
    assert code == 1


def test_real_run_refuses_without_confirm_flag_even_with_model_and_key(monkeypatch):
    monkeypatch.setenv("REFLECTION_LM_API_KEY", "fake-key-not-real")
    code = _real_run(
        [_traj("t1")], {"agents_md": "body"},
        _args(reflection_model="some/model", confirm_real_run=False),
    )
    assert code == 1


def test_real_run_refuses_with_no_trajectories_even_if_fully_authorized(monkeypatch):
    """All three gates satisfied but nothing to optimize against -- must
    still refuse rather than calling GEPA.compile() with an empty trainset."""
    monkeypatch.setenv("REFLECTION_LM_API_KEY", "fake-key-not-real")
    code = _real_run(
        [], {"agents_md": "body"},
        _args(reflection_model="some/model", confirm_real_run=True),
    )
    assert code == 1


def test_real_run_refuses_when_a_requested_source_yielded_zero_trajectories(monkeypatch):
    """Real gap, confirmed by review: _ingest_all can silently swallow a
    per-source failure (bad path, malformed logs) and still return a
    non-empty combined list from the OTHER sources -- which would otherwise
    sail past the 'no trajectories' check and train on a dataset the caller
    never actually asked for."""
    monkeypatch.setenv("REFLECTION_LM_API_KEY", "fake-key-not-real")
    code = _real_run(
        [_traj("t1")], {"agents_md": "body"},
        _args(reflection_model="some/model", confirm_real_run=True),
        empty_sources=["gaia"],
    )
    assert code == 1


def test_real_run_allows_partial_ingest_when_explicitly_overridden(monkeypatch, tmp_path):
    """--allow-partial-ingest is the documented, explicit escape hatch --
    with it set, an empty source must not block an otherwise-authorized run."""
    monkeypatch.setenv("REFLECTION_LM_API_KEY", "fake-key-not-real")

    class _StubGEPA:
        def __init__(self, **kw):
            pass

        def compile(self, student, **kw):
            return student

    monkeypatch.setattr("dspy.GEPA", _StubGEPA)
    code = _real_run(
        [_traj("t1")], {"agents_md": "body"},
        _args(reflection_model="some/model", confirm_real_run=True, allow_partial_ingest=True,
              out_dir=str(tmp_path / "out")),
        empty_sources=["gaia"],
    )
    assert code == 0
