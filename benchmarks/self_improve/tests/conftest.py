import copy
import importlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

# Runs before any test module imports run_gepa, whose import loads
# $SELF_IMPROVE_DOTENV. Assigned outright, not setdefault: an exported
# SELF_IMPROVE_DOTENV (possibly the real benchmarks/self_improve/.env, which
# can hold a real key) must not reach the test process or the subprocesses it
# spawns. os.devnull parses as an empty .env. LITELLM_MODE=PRODUCTION stops
# litellm's own import-time load_dotenv().
os.environ["SELF_IMPROVE_DOTENV"] = os.devnull
os.environ.setdefault("LITELLM_MODE", "PRODUCTION")

import pytest  # noqa: E402

# benchmarks/self_improve is an opt-in subsystem with its own pyproject.toml,
# deliberately outside the dependency-free benchmarks/*.py scripts' footprint
# -- CI's `benchmarks pytest` job installs only pytest. Every test file here
# imports benchmarks.self_improve modules at collection time, which import
# yaml/pydantic/dspy unconditionally, so without this guard collection fails
# for the whole job, not just this directory, breaking CI repo-wide.
# importorskip here skips collection of the whole directory when the optional
# deps are absent, matching README.md's "CI: manual-only for v1" design.
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

    dspy.settings is a process-wide singleton backed by a module-level dict
    (dspy.dsp.utils.settings's main_thread_config), and a
    dspy.settings.configure(lm=DummyLM(...)) call is never undone. Without
    this restore, every test that reaches one leaves the global LM as
    DummyLM for the REST of the session, so a later test asserting on dspy's
    defaults passes or fails on test order rather than its own behavior.

    dspy.dsp.utils.settings.settings (the singleton instance) has no public
    "reset" API and overrides __setattr__ to route through configure()
    itself (see components.py's own note on this), so the only way to
    restore state is to snapshot/restore the real module's dict directly --
    `import dspy.dsp.utils.settings as m` resolves to the singleton
    INSTANCE, not the module (the package's __init__ shadows the name), so
    importlib.import_module() is used to reach the actual module object.

    A shallow dict() copy shares references to nested mutable values (e.g. the list-valued "trace"
    setting -- components.py's own HarnessProgram docstring documents GEPA
    inspecting dspy.settings.trace after a forward pass, which DSPy mutates
    in-place via append, not reassignment) -- clear()+update() would restore
    the KEY to point at the SAME, already-mutated list, not undo the
    mutation. copy.deepcopy() is used instead. Settings.__getattr__ also
    checks thread_local_overrides (a contextvars.ContextVar for
    dspy.context()'s temporary per-thread overrides) BEFORE main_thread_config
    -- snapshotted/restored the same way for completeness, though nothing in
    this codebase currently uses dspy.context() (its own context manager
    already resets itself via a contextvars.Token on exit)."""
    settings_module = importlib.import_module("dspy.dsp.utils.settings")
    main_snapshot = copy.deepcopy(dict(settings_module.main_thread_config))
    overrides_snapshot = copy.deepcopy(settings_module.thread_local_overrides.get())
    yield
    settings_module.main_thread_config.clear()
    settings_module.main_thread_config.update(main_snapshot)
    settings_module.thread_local_overrides.set(overrides_snapshot)


@pytest.fixture(autouse=True)
def _restore_dspy_settings():
    yield from _dspy_settings_snapshot_and_restore()


REAL_REPO_ROOT = Path(__file__).resolve().parents[3]


def _git_common_dir(path) -> str | None:
    """Resolved `git rev-parse --git-common-dir` of `path`, or None if it is
    not inside a git repo. Every linked worktree of one repo shares it, so
    comparing it (not the checkout path) also catches a sibling worktree of
    the real repo."""
    result = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=path, capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    return os.path.realpath(result.stdout.strip())


#: The real repo's git common dir, read once (read-only) at import.
_REAL_GIT_COMMON_DIR = _git_common_dir(REAL_REPO_ROOT)


@pytest.fixture(scope="session", autouse=True)
def _no_stray_scratch_artifacts():
    """Session-scoped backstop: fails the session if this pytest process
    leaves scratch artifacts behind. Two snapshots, both limited to names
    THIS process could have made (scratch_worktree() names everything
    gepa-scratch-<pid>-<hex>; other sessions work concurrently):

    1. `git worktree list` on the REAL repo -- scratch_worktree() no longer
       registers worktrees at all, so any new entry is a regression to the
       old mechanism or a test touching the real repo;
    2. the system temp dir's top-level entries (tree, .git, .marker.json,
       .lock) -- what a test leaks if it omits parent_dir and teardown fails.

    It only detects a NET change across the session. _forbid_scratch_of_real_repo
    below is the per-test guard against pointing scratch_worktree() at the
    real repo."""
    ours = f"gepa-scratch-{os.getpid()}-"

    def _worktrees() -> set[str]:
        out = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=REAL_REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout
        return {
            line.removeprefix("worktree ")
            for line in out.splitlines()
            if line.startswith("worktree ") and Path(line.removeprefix("worktree ")).name.startswith(ours)
        }

    def _tempdir_entries() -> set[str]:
        return {name for name in os.listdir(tempfile.gettempdir()) if name.startswith(ours)}

    before = (_worktrees(), _tempdir_entries())
    yield
    after = (_worktrees(), _tempdir_entries())
    assert after[0] == before[0], (
        "a gepa-scratch worktree from this test session was left on the REAL "
        "repo -- a test touched the real repo's worktrees instead of a "
        f"throwaway fixture repo.\nbefore: {sorted(before[0])}\nafter: {sorted(after[0])}"
    )
    assert after[1] == before[1], (
        f"scratch artifacts from this test session were left in {tempfile.gettempdir()}.\n"
        f"before: {sorted(before[1])}\nafter: {sorted(after[1])}"
    )


@pytest.fixture(autouse=True)
def _forbid_scratch_of_real_repo(monkeypatch):
    """Per-test guard: scratch_worktree() pointed at the real repo -- or any
    linked worktree of it -- fails the test before the private repo is
    populated. Wraps scratch_worktree._create_private_repo, the one step that
    reads the source's objects."""
    import benchmarks.self_improve.scratch_worktree as sw

    real_create = sw._create_private_repo

    def _guarded(source_repo_root, *args, **kwargs):
        real_common = _REAL_GIT_COMMON_DIR
        if real_common is not None and _git_common_dir(source_repo_root) == real_common:
            pytest.fail(
                f"a test pointed scratch_worktree() at the real repo ({source_repo_root}, "
                f"git common dir {real_common}); use a throwaway `git init` repo instead"
            )
        return real_create(source_repo_root, *args, **kwargs)

    monkeypatch.setattr(sw, "_create_private_repo", _guarded)


@pytest.fixture(autouse=True)
def _forbid_real_pi(monkeypatch, tmp_path):
    """Autouse default: point LITTLE_CODER_PI_BIN_OVERRIDE at a nonexistent
    path unless a test explicitly overrides it, so any test that forgets to
    route a live-eval subprocess through fake_pi.py fails fast with a clear
    FileNotFoundError (from scratch_worktree.resolve_pi_bin) instead of
    silently attempting to spawn a real pi process against a real model
    server. Read live (not cached at import) by resolve_pi_bin(), so this
    monkeypatch is effective even though rpc_client.py itself resolves its
    own PI_BIN at import time."""
    monkeypatch.setenv("LITTLE_CODER_PI_BIN_OVERRIDE", str(tmp_path / "no-real-pi-in-tests"))


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
