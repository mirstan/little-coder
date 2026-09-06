"""Directly drives conftest.py's autouse _restore_dspy_settings fixture to
confirm it actually snapshots and restores dspy.settings' backing dict,
rather than trusting yield-based teardown blindly. Real gap, confirmed by
review: dspy.settings is a process-wide singleton, and every test that
reaches _real_run()'s dspy.settings.configure(lm=DummyLM(...)) call
permanently left the global LM as DummyLM for the rest of the pytest
session with no fixture undoing it."""
import dspy
import pytest

from benchmarks.self_improve.tests.conftest import _dspy_settings_snapshot_and_restore


def test_restore_dspy_settings_fixture_restores_prior_state():
    original_lm = dspy.settings.lm
    gen = _dspy_settings_snapshot_and_restore()
    next(gen)  # advance to yield -- snapshot taken

    dspy.settings.configure(lm="not a real LM, just a marker")
    assert dspy.settings.lm == "not a real LM, just a marker"

    with pytest.raises(StopIteration):
        next(gen)  # drive past yield -- runs the restore

    assert dspy.settings.lm == original_lm


def test_restore_dspy_settings_fixture_undoes_in_place_mutation_of_nested_values():
    """Real follow-up gap, confirmed by review: a shallow dict() copy shares
    references to nested mutable values -- dspy.settings.trace is a list
    DSPy mutates in-place (append) during a real forward pass, not by
    reassignment, so clear()+update() with a shallow snapshot would restore
    the key to point at the SAME, already-mutated list. copy.deepcopy() must
    actually produce an independent copy."""
    original_trace = list(dspy.settings.trace)
    gen = _dspy_settings_snapshot_and_restore()
    next(gen)

    dspy.settings.trace.append("a fake trace entry")
    assert "a fake trace entry" in dspy.settings.trace

    with pytest.raises(StopIteration):
        next(gen)

    assert dspy.settings.trace == original_trace
