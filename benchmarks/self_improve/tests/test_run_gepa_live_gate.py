"""Tests for run_gepa.py's two-budget safety gate and its estimate/baseline
CLI paths -- up to but NEVER including a real gepa.optimize() call, which
spends real reflection-LM budget and must only ever run on explicit human
invocation with a real reflection model configured.

Replaces test_run_gepa_real_path.py (the old dspy.GEPA/frozen-trajectory
design's gate tests) with the new two-gate model: reflection LM spend
(--reflection-model/$REFLECTION_LM_API_KEY/--confirm-real-run) and live
rollout spend (--model/--confirm-live-rollouts/--max-metric-calls) are
independent resources with independent refusal messages, plus a hard
machine-level deny ($SELF_IMPROVE_NO_LIVE_ROLLOUTS) that overrides everything.
"""
import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import benchmarks.self_improve.run_gepa as run_gepa
from benchmarks.self_improve.run_gepa import (
    NO_LIVE_ROLLOUTS_ENV,
    REFLECTION_LM_API_KEY_ENV,
    _check_components_clean,
    _check_gates,
    _resolve_components_yaml,
)

REAL_REPO_ROOT = Path(__file__).resolve().parents[3]
FAKE_PI = REAL_REPO_ROOT / "benchmarks" / "fake_pi.py"
_WORDY_TEST = 'import wordy\n\ndef test_wordy():\n    assert wordy.solve() == "42"\n'
_WORDY_STUB = "def solve():\n    pass\n"
_WORDY_SOLUTION = 'def solve():\n    return "42"\n'


def _args(**overrides):
    defaults = dict(
        model=None, confirm_live_rollouts=False, max_metric_calls=None,
        baseline_only=False, reflection_model=None, confirm_real_run=False,
        exercises=None, exercise_count=6, val_count=3, reflection_minibatch_size=2,
    )
    defaults.update(overrides)
    return type("Args", (), defaults)()


AUTHORIZED_ROLLOUT = dict(model="gpt-fake", confirm_live_rollouts=True, max_metric_calls=20)
AUTHORIZED_REFLECTION = dict(reflection_model="reflection/fake", confirm_real_run=True)


def test_hard_deny_env_var_refuses_regardless_of_everything_else(monkeypatch):
    monkeypatch.setenv(NO_LIVE_ROLLOUTS_ENV, "1")
    monkeypatch.setenv(REFLECTION_LM_API_KEY_ENV, "fake-key")
    messages = _check_gates(_args(**AUTHORIZED_ROLLOUT, **AUTHORIZED_REFLECTION))
    assert len(messages) == 1
    assert NO_LIVE_ROLLOUTS_ENV in messages[0]


@pytest.mark.parametrize("missing", ["model", "confirm_live_rollouts", "max_metric_calls"])
def test_refuses_when_a_rollout_flag_is_missing(monkeypatch, missing):
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    monkeypatch.setenv(REFLECTION_LM_API_KEY_ENV, "fake-key")
    rollout = dict(AUTHORIZED_ROLLOUT)
    rollout[missing] = None if missing != "confirm_live_rollouts" else False
    messages = _check_gates(_args(**rollout, **AUTHORIZED_REFLECTION))
    assert any("LIVE agent-under-test rollouts" in m for m in messages)


@pytest.mark.parametrize("missing", ["reflection_model", "confirm_real_run"])
def test_refuses_when_a_reflection_flag_is_missing_and_not_baseline_only(monkeypatch, missing):
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    monkeypatch.setenv(REFLECTION_LM_API_KEY_ENV, "fake-key")
    reflection = dict(AUTHORIZED_REFLECTION)
    reflection[missing] = None if missing == "reflection_model" else False
    messages = _check_gates(_args(**AUTHORIZED_ROLLOUT, **reflection))
    assert any("Reflection LM tokens" in m for m in messages)


def test_refuses_without_reflection_api_key_even_with_model_and_confirm(monkeypatch):
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    monkeypatch.delenv(REFLECTION_LM_API_KEY_ENV, raising=False)
    messages = _check_gates(_args(**AUTHORIZED_ROLLOUT, **AUTHORIZED_REFLECTION))
    assert any("Reflection LM tokens" in m for m in messages)


def test_baseline_only_skips_the_reflection_gate_entirely(monkeypatch):
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    monkeypatch.delenv(REFLECTION_LM_API_KEY_ENV, raising=False)
    messages = _check_gates(_args(**AUTHORIZED_ROLLOUT, baseline_only=True))
    assert messages == []


def test_baseline_only_still_requires_the_rollout_gate(monkeypatch):
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    messages = _check_gates(_args(baseline_only=True))
    assert any("LIVE agent-under-test rollouts" in m for m in messages)


def test_fully_authorized_non_baseline_run_passes_the_gate(monkeypatch):
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    monkeypatch.setenv(REFLECTION_LM_API_KEY_ENV, "fake-key")
    messages = _check_gates(_args(**AUTHORIZED_ROLLOUT, **AUTHORIZED_REFLECTION))
    assert messages == []


def test_refuses_non_positive_max_metric_calls(monkeypatch):
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    monkeypatch.setenv(REFLECTION_LM_API_KEY_ENV, "fake-key")
    rollout = dict(AUTHORIZED_ROLLOUT, max_metric_calls=0)
    messages = _check_gates(_args(**rollout, **AUTHORIZED_REFLECTION))
    assert any("must be > 0" in m for m in messages)


def test_refuses_reflection_minibatch_larger_than_train_pool(monkeypatch):
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    monkeypatch.setenv(REFLECTION_LM_API_KEY_ENV, "fake-key")
    messages = _check_gates(_args(
        **AUTHORIZED_ROLLOUT, **AUTHORIZED_REFLECTION,
        exercise_count=3, val_count=2, reflection_minibatch_size=5,
    ))
    assert any("reflection-minibatch-size" in m for m in messages)


def test_reflection_minibatch_check_does_not_apply_to_baseline_only(monkeypatch):
    """A baseline-only run never touches reflection_minibatch_size -- the
    default value must not spuriously block a small exercise set."""
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    messages = _check_gates(_args(
        **AUTHORIZED_ROLLOUT, baseline_only=True,
        exercise_count=2, val_count=1, reflection_minibatch_size=2,
    ))
    assert messages == []


def test_refuses_val_count_at_least_exercise_count(monkeypatch):
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    monkeypatch.setenv(REFLECTION_LM_API_KEY_ENV, "fake-key")
    messages = _check_gates(_args(
        **AUTHORIZED_ROLLOUT, **AUTHORIZED_REFLECTION, exercise_count=3, val_count=3,
    ))
    assert any("--val-count" in m for m in messages)


def test_gates_use_the_real_exercises_count_not_exercise_count_when_explicit(monkeypatch):
    """Real bug, confirmed by review: select_exercises() ignores
    --exercise-count entirely when --exercises is given (the real pool size
    is len(explicit)), but the minibatch/val-count gates used to reason from
    the raw --exercise-count flag regardless -- silently passing a
    reflection-minibatch-size that would make GEPA pad the minibatch by
    repeating an exercise (paying twice for zero extra signal)."""
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    monkeypatch.setenv(REFLECTION_LM_API_KEY_ENV, "fake-key")
    # exercise_count says 6 (a healthy train pool), but only 3 exercises are
    # actually named -- val_count=2 leaves a real train pool of 1.
    messages = _check_gates(_args(
        **AUTHORIZED_ROLLOUT, **AUTHORIZED_REFLECTION,
        exercises="a,b,c", exercise_count=6, val_count=2, reflection_minibatch_size=2,
    ))
    assert any("reflection-minibatch-size" in m for m in messages)


def test_gates_val_count_against_real_exercises_length_not_exercise_count(monkeypatch):
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    monkeypatch.setenv(REFLECTION_LM_API_KEY_ENV, "fake-key")
    # exercise_count=6 would normally allow val_count=2, but only 2 exercises
    # are actually named -- val_count must be < 2, not < 6.
    messages = _check_gates(_args(
        **AUTHORIZED_ROLLOUT, **AUTHORIZED_REFLECTION,
        exercises="a,b", exercise_count=6, val_count=2, reflection_minibatch_size=1,
    ))
    assert any("--val-count" in m for m in messages)


def test_resolve_components_yaml_relative_path():
    repo_root = Path("/repo")
    abs_path, rel_path = _resolve_components_yaml(repo_root, "config/components.yaml")
    assert abs_path == repo_root / "config" / "components.yaml"
    assert rel_path == Path("config/components.yaml")


def test_resolve_components_yaml_absolute_path_under_repo_root():
    repo_root = Path("/repo")
    abs_path, rel_path = _resolve_components_yaml(repo_root, "/repo/config/components.yaml")
    assert abs_path == repo_root / "config" / "components.yaml"
    assert rel_path == Path("config/components.yaml")


@pytest.fixture
def source_repo(tmp_path):
    """A throwaway git repo, committed, with real-shaped component files and
    committed copies of aider_polyglot.py/rpc_client.py."""
    repo = tmp_path / "source"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)

    (repo / "AGENTS.md").write_text("# little-coder\n\nOriginal instructions.\n")
    skills_dir = repo / "skills" / "tools"
    skills_dir.mkdir(parents=True)
    (skills_dir / "bash.md").write_text("---\nname: bash\ntype: tool-guidance\n---\nOriginal bash guidance.\n")

    (repo / "config").mkdir()
    (repo / "config" / "components.yaml").write_text(yaml.dump({
        "agents_md": "AGENTS.md",
        "skills_tools_bash": "skills/tools/bash.md",
    }))

    (repo / "benchmarks").mkdir()
    shutil.copy(REAL_REPO_ROOT / "benchmarks" / "aider_polyglot.py", repo / "benchmarks" / "aider_polyglot.py")
    shutil.copy(REAL_REPO_ROOT / "benchmarks" / "rpc_client.py", repo / "benchmarks" / "rpc_client.py")

    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo, check=True)
    return repo


def test_check_components_clean_is_empty_for_a_freshly_committed_repo(source_repo):
    assert _check_components_clean(source_repo, source_repo / "config" / "components.yaml") == []


def test_check_components_clean_refuses_on_uncommitted_changes(source_repo):
    (source_repo / "AGENTS.md").write_text("dirty, uncommitted change\n")
    messages = _check_components_clean(source_repo, source_repo / "config" / "components.yaml")
    assert len(messages) == 1
    assert "uncommitted changes" in messages[0]


def test_check_components_clean_refuses_when_git_status_itself_fails(source_repo, tmp_path):
    """Real bug, confirmed by review: a nonzero git-status exit (e.g. a
    components.yaml entry escaping repo_root, or repo_root not being a git
    checkout at all) was treated identically to "nothing is dirty" -- this
    check must refuse when it can't verify its own invariant, not fail open."""
    not_a_repo = tmp_path / "not-a-git-repo"
    not_a_repo.mkdir()
    messages = _check_components_clean(not_a_repo, source_repo / "config" / "components.yaml")
    assert len(messages) == 1
    assert "Could not verify" in messages[0]


@pytest.fixture
def fake_practice(tmp_path):
    """Three real exercises (stub + genuine failing pytest test each) so
    --exercise-count 3 --val-count 1 has a real pool to select from."""
    practice_root = tmp_path / "polyglot-benchmark"
    for name in ("wordy", "acronym", "leap"):
        ex_dir = practice_root / "python" / "exercises" / "practice" / name
        ex_dir.mkdir(parents=True)
        (ex_dir / f"{name}.py").write_text(_WORDY_STUB)
        (ex_dir / f"{name}_test.py").write_text(_WORDY_TEST.replace("wordy", name))
    return practice_root


def _run_main(args_list):
    argv_backup = sys.argv
    sys.argv = ["run_gepa.py", *args_list]
    try:
        return run_gepa.main()
    finally:
        sys.argv = argv_backup


def test_estimate_only_exits_zero_and_creates_no_worktree(source_repo, fake_practice, tmp_path, monkeypatch):
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    before = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=source_repo,
                             capture_output=True, text=True, check=True).stdout
    out_dir = tmp_path / "run_out"
    code = _run_main([
        "--repo-root", str(source_repo), "--components-config", "config/components.yaml",
        "--benchmark-root", str(fake_practice), "--exercise-count", "3", "--val-count", "1",
        "--model", "gpt-fake", "--confirm-live-rollouts", "--max-metric-calls", "20",
        "--out-dir", str(out_dir), "--estimate-only",
    ])
    assert code == 0
    after = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=source_repo,
                            capture_output=True, text=True, check=True).stdout
    assert before == after
    assert not (out_dir / "spend_log.jsonl").exists()


def test_missing_gate_flags_refuse_before_touching_anything(source_repo, fake_practice, tmp_path, monkeypatch):
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    out_dir = tmp_path / "run_out"
    code = _run_main([
        "--repo-root", str(source_repo), "--components-config", "config/components.yaml",
        "--benchmark-root", str(fake_practice), "--out-dir", str(out_dir),
    ])
    assert code == 1
    assert not out_dir.exists()


def test_refuses_when_a_leftover_stop_file_already_exists(
    source_repo, fake_practice, tmp_path, monkeypatch,
):
    """Real bug, confirmed by review: --out-dir defaults to a FIXED path, so
    a gepa.stop left over from a previous (correctly) stopped run would
    otherwise silently no-op the very next run at the first stop check --
    for a real gepa.optimize() call that's AFTER paying for the full seed
    valset evaluation, writing back the untouched seed as
    optimized_components.yaml and self-reporting "completed"."""
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    out_dir = tmp_path / "run_out"
    out_dir.mkdir(parents=True)
    (out_dir / "gepa.stop").write_text("")

    code = _run_main([
        "--repo-root", str(source_repo), "--components-config", "config/components.yaml",
        "--benchmark-root", str(fake_practice), "--exercise-count", "3", "--val-count", "1",
        "--model", "gpt-fake", "--confirm-live-rollouts", "--max-metric-calls", "1",
        "--out-dir", str(out_dir), "--baseline-only", "--yes",
    ])
    assert code == 1
    # the stop file itself is left in place -- refusal, not silent cleanup
    assert (out_dir / "gepa.stop").exists()
    assert not (out_dir / "seed_baseline.json").exists()


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def test_baseline_only_end_to_end_real_pipeline(source_repo, fake_practice, tmp_path, monkeypatch):
    """Real pipeline, real subprocess, fake_pi -- proves --baseline-only
    actually runs the seed candidate and writes both audit artifacts."""
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    monkeypatch.setenv("ATTEMPT_TIMEOUT_S", "30")
    monkeypatch.setenv("FAKE_PI_MODE", "solve_from_env")
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({"wordy.py": _b64(_WORDY_SOLUTION)}))
    out_dir = tmp_path / "run_out"
    scratch_dir = tmp_path / "scratch"

    code = _run_main([
        "--repo-root", str(source_repo), "--components-config", "config/components.yaml",
        "--benchmark-root", str(fake_practice),
        "--exercises", "wordy,acronym", "--exercise-count", "2", "--val-count", "1",
        "--model", "gpt-fake", "--confirm-live-rollouts", "--max-metric-calls", "5",
        "--out-dir", str(out_dir), "--scratch-dir", str(scratch_dir),
        "--pi-bin", str(FAKE_PI), "--baseline-only", "--yes",
    ])
    assert code == 0
    assert (out_dir / "spend_log.jsonl").exists()
    seed_baseline = json.loads((out_dir / "seed_baseline.json").read_text())
    assert "python/wordy" in seed_baseline
    assert seed_baseline["python/wordy"]["status"] == "pass_1"


def test_check_components_clean_catches_dirty_components_yaml_itself(source_repo):
    """Real bug, confirmed by review: editing components.yaml to point a
    pred_name at a DIFFERENT (already-committed) file left every mapped
    file's own git status clean, so the old check missed that the mapping
    itself -- what the scratch worktree will actually read at its pinned
    commit -- had uncommitted changes."""
    components_yaml = source_repo / "config" / "components.yaml"
    components_yaml.write_text(components_yaml.read_text() + "\nextra_unmapped_key: AGENTS.md\n")
    messages = run_gepa._check_components_clean(source_repo, components_yaml)
    assert len(messages) == 1
    assert "uncommitted changes" in messages[0]


def test_baseline_only_stops_at_the_exact_max_metric_calls_cap(
    source_repo, fake_practice, tmp_path, monkeypatch,
):
    """Real bug, confirmed by review: baseline-only's LiveBudget was
    constructed with est.max_live_runs (max_metric_calls PLUS GEPA's
    2*minibatch+valset overshoot allowance, which baseline mode never uses
    at all), so a small --max-metric-calls cap silently let MORE exercises
    run than requested. Here 3 exercises are selected but --max-metric-calls
    1 must stop after exactly 1."""
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    monkeypatch.setenv("ATTEMPT_TIMEOUT_S", "30")
    monkeypatch.setenv("FAKE_PI_MODE", "solve_from_env")
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({"wordy.py": _b64(_WORDY_SOLUTION)}))
    out_dir = tmp_path / "run_out"
    scratch_dir = tmp_path / "scratch"

    code = _run_main([
        "--repo-root", str(source_repo), "--components-config", "config/components.yaml",
        "--benchmark-root", str(fake_practice),
        "--exercises", "wordy,acronym,leap", "--exercise-count", "3", "--val-count", "1",
        "--model", "gpt-fake", "--confirm-live-rollouts", "--max-metric-calls", "1",
        "--out-dir", str(out_dir), "--scratch-dir", str(scratch_dir),
        "--pi-bin", str(FAKE_PI), "--baseline-only", "--yes",
    ])
    assert code == 3  # budget backstop, not a full run of all 3 exercises
    seed_baseline = json.loads((out_dir / "seed_baseline.json").read_text())
    assert len(seed_baseline) == 1
