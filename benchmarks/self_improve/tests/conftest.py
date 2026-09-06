import importlib
import json
from pathlib import Path

import pytest

# Real bug, confirmed by CI failure on this branch's PR: benchmarks/self_improve
# is an opt-in subsystem (its own pyproject.toml, installed via
# `pip install -e benchmarks/self_improve[dev]`) deliberately kept out of the
# existing dependency-free benchmarks/*.py scripts' footprint -- CI's
# `benchmarks pytest` job only `pip install`s `pytest` itself. Every test file here imports
# benchmarks.self_improve modules at collection time, which import yaml/
# pydantic/dspy unconditionally, so without this guard collection fails HARD
# for the whole job (14 errors), not just for this directory -- breaking CI for
# the entire repo, not only for self_improve. importorskip in conftest.py skips
# collection of this whole directory when the optional deps aren't installed,
# matching this subsystem's own documented "CI: manual-only for v1" design
# (README.md / architecture plan) that these tests were always meant to
# self-skip rather than run unconditionally.
pytest.importorskip("dspy")
pytest.importorskip("pydantic")
pytest.importorskip("yaml")


@pytest.fixture(autouse=True)
def _isolated_git_config(tmp_path_factory, monkeypatch):
    """Autouse for every test in this directory: test_apply_results.py's
    scratch repos run real `git init`/`commit` subprocesses, which by
    default still read the DEVELOPER's real ~/.gitconfig (and any
    /etc/gitconfig) -- a global commit.gpgsign=true or core.hooksPath there
    could make `git commit` fail (or hang on a passphrase prompt) in a way
    that has nothing to do with what the test is checking. GIT_CONFIG_GLOBAL
    (git >= 2.32) fully replaces the global config path for these
    subprocesses without touching the developer's real file; GIT_CONFIG_
    SYSTEM/NOSYSTEM similarly neutralize any machine-wide config."""
    fake_global = tmp_path_factory.mktemp("git-config") / "gitconfig"
    fake_global.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(fake_global))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(fake_global.parent / "does-not-exist"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def _dspy_settings_snapshot_and_restore():
    """Plain generator backing the _restore_dspy_settings fixture below --
    kept separate from @pytest.fixture so test_conftest_dspy_settings_restore.py
    can drive it directly (pytest fixtures can't be called directly).

    Real gap, confirmed by review: dspy.settings is a process-wide
    singleton backed by a module-level dict (dspy.dsp.utils.settings's
    main_thread_config) -- _real_run() (and any other code exercised here)
    calls dspy.settings.configure(lm=DummyLM(...)) directly, which is never
    undone. Every test in this directory that reaches that call permanently
    leaves the global LM as DummyLM for the REST of the pytest session, so a
    later test asserting on dspy's default settings could pass or fail based
    on prior test order rather than its own behavior, and a future
    regression that removed the configure() call from _real_run() could go
    uncaught by any test that reads dspy.settings.lm back instead of
    asserting on what was actually passed to configure().

    dspy.dsp.utils.settings.settings (the singleton instance) has no public
    "reset" API and overrides __setattr__ to route through configure()
    itself (see components.py's own note on this), so the only way to
    restore state is to snapshot/restore the real module's dict directly --
    `import dspy.dsp.utils.settings as m` resolves to the singleton
    INSTANCE, not the module (the package's __init__ shadows the name), so
    importlib.import_module() is used to reach the actual module object."""
    settings_module = importlib.import_module("dspy.dsp.utils.settings")
    snapshot = dict(settings_module.main_thread_config)
    yield
    settings_module.main_thread_config.clear()
    settings_module.main_thread_config.update(snapshot)


@pytest.fixture(autouse=True)
def _restore_dspy_settings():
    yield from _dspy_settings_snapshot_and_restore()


@pytest.fixture
def gaia_run(tmp_path) -> Path:
    """A minimal gaia benchmark run directory: two task dirs, matching the
    real layout confirmed in TDD_SPEC.md §0 (result.json, tool_calls.jsonl,
    notifications.txt, transcript.txt, prompt.txt per task)."""
    t1 = tmp_path / "task-001"
    t1.mkdir()
    (t1 / "result.json").write_text(json.dumps({
        "model_answer": "42", "gold": "42", "correct": True, "elapsed_s": 12.3,
    }))
    (t1 / "tool_calls.jsonl").write_text(
        json.dumps({"name": "bash", "args": {}, "result_text": "ok", "is_error": False}) + "\n"
    )
    (t1 / "notifications.txt").write_text("[info] skill-inject: +1 [bash]\n")
    (t1 / "transcript.txt").write_text("final answer: 42")
    (t1 / "prompt.txt").write_text("solve this task")

    t2 = tmp_path / "task-002"
    t2.mkdir()
    (t2 / "result.json").write_text(json.dumps({
        "model_answer": "", "gold": "7", "correct": False, "elapsed_s": 900.0,
    }))
    (t2 / "tool_calls.jsonl").write_text("")
    (t2 / "notifications.txt").write_text("")
    (t2 / "transcript.txt").write_text("")
    (t2 / "stderr.log").write_text("Traceback ...\nRuntimeError: pi exited\n")

    return tmp_path
