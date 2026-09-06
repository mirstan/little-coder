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
