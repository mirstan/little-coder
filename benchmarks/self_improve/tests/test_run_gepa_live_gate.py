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


def test_refuses_non_positive_reflection_minibatch_size_even_for_baseline_only(monkeypatch):
    """Unlike the train-pool-compatibility
    check above (genuinely irrelevant to baseline mode), this basic sanity
    bound must apply regardless of --baseline-only -- estimate_cost() is
    called unconditionally in BOTH modes to print the pre-authorization
    banner, and a negative value there produces a negative, nonsensical
    cost/wall-clock estimate before a human even decides whether to
    authorize a real (--baseline-only) run."""
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    messages = _check_gates(_args(
        **AUTHORIZED_ROLLOUT, baseline_only=True, reflection_minibatch_size=-1,
    ))
    assert any("--reflection-minibatch-size" in m for m in messages)


def test_refuses_val_count_at_least_exercise_count(monkeypatch):
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    monkeypatch.setenv(REFLECTION_LM_API_KEY_ENV, "fake-key")
    messages = _check_gates(_args(
        **AUTHORIZED_ROLLOUT, **AUTHORIZED_REFLECTION, exercise_count=3, val_count=3,
    ))
    assert any("--val-count" in m for m in messages)


def test_gates_use_the_real_exercises_count_not_exercise_count_when_explicit(monkeypatch):
    """Select_exercises() ignores
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


def test_resolve_components_yaml_raises_a_clean_error_when_absolute_path_escapes_repo_root():
    """An absolute --components-config
    outside --repo-root made Path.relative_to() raise an UNCAUGHT
    ValueError -- a raw traceback instead of a clean CLI refusal."""
    with pytest.raises(ValueError, match="outside --repo-root"):
        _resolve_components_yaml(Path("/repo"), "/somewhere/else/components.yaml")


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
    """A nonzero git-status exit (e.g. a
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
    """Five real exercises (stub + genuine failing pytest test each) so
    --exercise-count 3 --val-count 1 has a real pool to select from, and any
    3-exercise search split still leaves two for acceptance and test."""
    practice_root = tmp_path / "polyglot-benchmark"
    for name in ("wordy", "acronym", "leap", "bob", "isogram"):
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


def test_main_reports_a_clean_error_when_components_config_escapes_repo_root(
    source_repo, tmp_path, monkeypatch,
):
    """End-to-end version of test_resolve_components_yaml_raises_a_clean_error_
    when_absolute_path_escapes_repo_root above: that test only exercises the
    private helper directly, so it would keep passing even if _run_live's own
    try/except around the call were ever removed, silently regressing the CLI
    back to a raw traceback -- --estimate-only reaches _resolve_components_yaml
    without needing any of the spend-gate flags, so this is a free, real check
    of the actual refusal message a user would see."""
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    escaping_config = tmp_path.parent / "outside-repo-root" / "components.yaml"
    code = _run_main([
        "--repo-root", str(source_repo), "--components-config", str(escaping_config),
        "--estimate-only",
    ])
    assert code == 1


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
    """--out-dir defaults to a FIXED path, so
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
    """Editing components.yaml to point a
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
    """Baseline-only's LiveBudget was
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


def test_spend_log_zeroes_duration_for_a_cache_hit(source_repo, fake_practice, tmp_path, monkeypatch):
    """Logging a cache hit's ORIGINAL
    elapsed_s let SpendLog.summarize()'s total_wall_s double-count the same
    real wall-clock time every time a candidate's result was reused from the
    on-disk memo."""
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    monkeypatch.setenv("ATTEMPT_TIMEOUT_S", "30")
    monkeypatch.setenv("FAKE_PI_MODE", "solve_from_env")
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({
        "wordy.py": _b64(_WORDY_SOLUTION), "acronym.py": _b64(_WORDY_SOLUTION),
    }))
    live_cache_dir = tmp_path / "shared_cache"

    common_args = [
        "--repo-root", str(source_repo), "--components-config", "config/components.yaml",
        "--benchmark-root", str(fake_practice), "--exercises", "wordy,acronym",
        "--exercise-count", "2", "--val-count", "1",
        "--model", "gpt-fake", "--confirm-live-rollouts", "--max-metric-calls", "5",
        "--pi-bin", str(FAKE_PI), "--baseline-only", "--yes", "--live-cache-dir", str(live_cache_dir),
    ]
    # first run: real subprocess, populates the cache
    out_dir_1 = tmp_path / "run_out_1"
    code1 = _run_main([*common_args, "--out-dir", str(out_dir_1), "--scratch-dir", str(tmp_path / "scratch1")])
    assert code1 == 0

    # second run: identical candidate/model/etc -> cache hit, no subprocess
    out_dir_2 = tmp_path / "run_out_2"
    code2 = _run_main([*common_args, "--out-dir", str(out_dir_2), "--scratch-dir", str(tmp_path / "scratch2")])
    assert code2 == 0

    records = [json.loads(line) for line in (out_dir_2 / "spend_log.jsonl").read_text().splitlines()]
    exercise_records = [r for r in records if r.get("event") == "exercise"]
    assert len(exercise_records) == 2
    assert all(r["memo_hit"] is True for r in exercise_records)
    assert all(r["duration_s"] == 0.0 for r in exercise_records)
    # Tokens are spend too: a hit zeroes them just like duration_s. The
    # first run's records carry the real numbers.
    assert all(r["usage"] == {"input_tokens": 0, "cache_read_tokens": 0, "output_tokens": 0}
               for r in exercise_records)
    expected_usage = {"input_tokens": 110, "cache_read_tokens": 10, "output_tokens": 20}
    first_run = [json.loads(line) for line in (out_dir_1 / "spend_log.jsonl").read_text().splitlines()]
    assert [r["usage"] for r in first_run if r.get("event") == "exercise"] == [expected_usage] * 2


def test_baseline_scores_a_persistent_runtime_error_zero_and_logs_its_reason(
    source_repo, fake_practice, tmp_path, monkeypatch,
):
    """A persistent runtime "error" is retried in place, then scored 0.0:
    the run does not stop (exit 0, not 4). Every try is real spend, so each
    reaches spend_log.jsonl with its status and its reason, capped."""
    from benchmarks.self_improve.live_eval import HARNESS_ERROR_RETRIES, LiveRunResult

    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    long_reason = "x" * 800
    reasons = {"wordy": "boom", "acronym": long_reason}
    class _ErroringRunner(run_gepa.PolyglotLiveRunner):
        def _run_one_uncached(self, spec):
            return LiveRunResult(task_id=spec.task_id, exercise=spec.exercise, language=spec.language,
                                 status="error", score=0.0, success=False, error=reasons[spec.exercise])

    monkeypatch.setattr(run_gepa, "PolyglotLiveRunner", _ErroringRunner)
    out_dir = tmp_path / "run_out"
    runs_per_exercise = 1 + HARNESS_ERROR_RETRIES
    code = _run_main([
        "--repo-root", str(source_repo), "--components-config", "config/components.yaml",
        "--benchmark-root", str(fake_practice), "--exercises", "wordy,acronym",
        "--exercise-count", "2", "--val-count", "1",
        "--model", "gpt-fake", "--confirm-live-rollouts",
        "--max-metric-calls", str(2 * runs_per_exercise),
        "--out-dir", str(out_dir), "--scratch-dir", str(tmp_path / "scratch"),
        "--pi-bin", str(FAKE_PI), "--baseline-only", "--yes",
    ])
    assert code == 0
    seed_baseline = json.loads((out_dir / "seed_baseline.json").read_text())
    assert sorted(seed_baseline) == ["python/acronym", "python/wordy"]
    assert all(r["score"] == 0.0 for r in seed_baseline.values())
    records = [json.loads(line) for line in (out_dir / "spend_log.jsonl").read_text().splitlines()]
    exercise_records = [r for r in records if r.get("event") == "exercise"]
    assert len(exercise_records) == 2 * runs_per_exercise
    assert all(r["status"] == "error" for r in exercise_records)
    by_id = {}
    for r in exercise_records:
        by_id.setdefault(r["exercise_id"], []).append(r["error"])
    assert by_id == {"python/wordy": ["boom"] * runs_per_exercise,
                     "python/acronym": [long_reason[:500]] * runs_per_exercise}
    assert not any("probe" in r for r in exercise_records)


@pytest.mark.parametrize("extra_flags,expected_skip", [([], True), (["--no-skip-perfect-score"], False)])
def test_optimize_skips_perfect_scores_by_default_and_passes_perfect_score(
    source_repo, fake_practice, tmp_path, monkeypatch, extra_flags, expected_skip,
):
    """Without skip_perfect_score, GEPA pays for reflection plus a child
    minibatch on a parent that already solved every sampled exercise --
    nothing to improve. perfect_score=1.0 is the per-exercise max (pass on
    attempt 1); GEPA compares each minibatch score against it."""
    import gepa

    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    monkeypatch.setenv(REFLECTION_LM_API_KEY_ENV, "fake-key")
    captured = {}

    def fake_optimize(**kwargs):
        captured.update(kwargs)
        return type("Result", (), {"best_candidate": dict(kwargs["seed_candidate"])})()

    monkeypatch.setattr(gepa, "optimize", fake_optimize)
    out_dir = tmp_path / "run_out"
    code = _run_main([
        "--repo-root", str(source_repo), "--components-config", "config/components.yaml",
        "--benchmark-root", str(fake_practice),
        "--exercises", "wordy,acronym", "--exercise-count", "2", "--val-count", "1",
        "--reflection-minibatch-size", "1",
        "--model", "gpt-fake", "--confirm-live-rollouts", "--max-metric-calls", "5",
        "--reflection-model", "reflection/fake", "--confirm-real-run",
        "--out-dir", str(out_dir), "--scratch-dir", str(tmp_path / "scratch"),
        "--pi-bin", str(FAKE_PI), "--yes", *extra_flags,
    ])
    assert code == 0
    assert captured["skip_perfect_score"] is expected_skip
    assert captured["perfect_score"] == 1.0


def test_baseline_only_persists_partial_results_on_a_persistent_harness_error(
    source_repo, fake_practice, tmp_path, monkeypatch,
):
    """run_batch() now raises LiveEvalHarnessError instead of returning a
    0.0 harness_error result -- baseline mode must still write what DID run
    rather than crash and lose it."""
    from benchmarks.self_improve.live_eval import LiveEvalHarnessError, LiveRunResult, PolyglotLiveRunner

    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)

    def fake_run_batch(self, candidate, specs, *, sample_index=0):
        (spec,) = specs
        if spec.exercise == "acronym":
            raise LiveEvalHarnessError("results file missing")
        return [LiveRunResult(task_id=spec.task_id, exercise=spec.exercise, language=spec.language,
                              status="pass_1", score=1.0, success=True)]

    monkeypatch.setattr(PolyglotLiveRunner, "run_batch", fake_run_batch)
    out_dir = tmp_path / "run_out"
    code = _run_main([
        "--repo-root", str(source_repo), "--components-config", "config/components.yaml",
        "--benchmark-root", str(fake_practice),
        "--exercises", "wordy,acronym", "--exercise-count", "2", "--val-count", "1",
        "--model", "gpt-fake", "--confirm-live-rollouts", "--max-metric-calls", "5",
        "--out-dir", str(out_dir), "--scratch-dir", str(tmp_path / "scratch"),
        "--pi-bin", str(FAKE_PI), "--baseline-only", "--yes",
    ])
    assert code == 4
    seed_baseline = json.loads((out_dir / "seed_baseline.json").read_text())
    assert list(seed_baseline) == ["python/wordy"]
    records = [json.loads(line) for line in (out_dir / "spend_log.jsonl").read_text().splitlines()]
    assert records[-1].get("reason") == "harness_error"


def _optimize_argv(source_repo, fake_practice, tmp_path, out_dir):
    return [
        "--repo-root", str(source_repo), "--components-config", "config/components.yaml",
        "--benchmark-root", str(fake_practice),
        "--exercises", "wordy,acronym", "--exercise-count", "2", "--val-count", "1",
        "--reflection-minibatch-size", "1",
        "--model", "gpt-fake", "--confirm-live-rollouts", "--max-metric-calls", "5",
        "--reflection-model", "reflection/fake", "--confirm-real-run",
        "--out-dir", str(out_dir), "--scratch-dir", str(tmp_path / "scratch"),
        "--pi-bin", str(FAKE_PI), "--yes",
    ]


def _valset_event(idx, candidate, score, is_best):
    return {"iteration": idx, "candidate_idx": idx, "candidate": candidate,
            "scores_by_val_id": {0: score}, "average_score": score,
            "num_examples_evaluated": 1, "total_valset_size": 1, "parent_ids": [],
            "is_best_program": is_best, "outputs_by_val_id": None}


def _raise_budget():
    from benchmarks.self_improve.live_budget import LiveEvalBudgetExceeded
    raise LiveEvalBudgetExceeded("max_live_runs reached")


def _raise_harness():
    from benchmarks.self_improve.live_eval import LiveEvalHarnessError
    raise LiveEvalHarnessError("results file missing")


@pytest.mark.parametrize("raise_fn,expected_code,expected_reason", [
    (_raise_budget, 3, "budget_backstop"),
    (_raise_harness, 4, "harness_error"),
])
def test_optimize_overrun_writes_the_best_candidate_seen_so_far(
    source_repo, fake_practice, tmp_path, monkeypatch, raise_fn, expected_code, expected_reason,
):
    """A budget or harness overrun escaping gepa.optimize() used to return
    with no optimized_components.yaml at all -- every paid-for valset
    evaluation was thrown away. The best candidate GEPA reported so far
    (its own is_best_program verdict, not a later non-best one) must land
    in the same file, sanitized, with the run marked partial."""
    import gepa

    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    monkeypatch.setenv(REFLECTION_LM_API_KEY_ENV, "fake-key")

    def fake_optimize(**kwargs):
        seed = dict(kwargs["seed_candidate"])
        improved = {**seed, "agents_md": "---\nname: dup\n---\nImproved instructions.\n"}
        worse = {**seed, "agents_md": "Worse instructions.\n"}
        # GEPA's notify_callbacks skips a callback without the method.
        for cb in [cb for cb in kwargs["callbacks"] if hasattr(cb, "on_valset_evaluated")]:
            cb.on_valset_evaluated(_valset_event(0, seed, 0.5, True))
            cb.on_valset_evaluated(_valset_event(1, improved, 0.8, True))
            cb.on_valset_evaluated(_valset_event(2, worse, 0.1, False))
        raise_fn()

    monkeypatch.setattr(gepa, "optimize", fake_optimize)
    out_dir = tmp_path / "run_out"
    code = _run_main(_optimize_argv(source_repo, fake_practice, tmp_path, out_dir))
    assert code == expected_code
    written = yaml.safe_load((out_dir / "optimized_components.yaml").read_text())
    assert written["agents_md"] == "Improved instructions.\n"
    run_end = [json.loads(line) for line in (out_dir / "spend_log.jsonl").read_text().splitlines()][-1]
    assert run_end["event"] == "run_end"
    assert run_end["reason"] == expected_reason
    assert run_end["partial"] is True
    assert run_end["best_candidate_idx"] == 1
    assert run_end["best_val_score"] == 0.8


def test_optimize_overrun_before_any_valset_evaluation_writes_nothing(
    source_repo, fake_practice, tmp_path, monkeypatch, capsys,
):
    import gepa

    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    monkeypatch.setenv(REFLECTION_LM_API_KEY_ENV, "fake-key")

    def fake_optimize(**kwargs):
        _raise_budget()

    monkeypatch.setattr(gepa, "optimize", fake_optimize)
    out_dir = tmp_path / "run_out"
    code = _run_main(_optimize_argv(source_repo, fake_practice, tmp_path, out_dir))
    assert code == 3
    assert not (out_dir / "optimized_components.yaml").exists()
    assert "no candidate" in capsys.readouterr().err.lower()
    run_end = [json.loads(line) for line in (out_dir / "spend_log.jsonl").read_text().splitlines()][-1]
    assert run_end["reason"] == "budget_backstop"
    assert run_end["partial"] is True
    assert run_end["best_candidate_idx"] is None


def _gepa_state(i, total_num_evals):
    return type("State", (), {"i": i, "total_num_evals": total_num_evals})()


def test_no_progress_stopper_trips_after_n_iterations_without_a_metric_call():
    """A parent minibatch of perfect cache hits charges no metric call and
    skip_perfect_score skips reflection, so total_num_evals never grows and
    max_metric_calls never stops GEPA."""
    stopper = run_gepa.NoProgressStopper(max_idle_iterations=3)
    assert stopper(_gepa_state(-1, 4)) is False  # before the first iteration
    assert stopper(_gepa_state(0, 6)) is False  # spent two metric calls
    assert stopper(_gepa_state(1, 6)) is False
    assert stopper(_gepa_state(2, 6)) is False
    assert stopper.tripped is False
    assert stopper(_gepa_state(3, 6)) is True
    assert stopper.tripped is True


def test_no_progress_stopper_resets_on_a_metric_call_and_ignores_repeat_checks():
    stopper = run_gepa.NoProgressStopper(max_idle_iterations=2)
    assert stopper(_gepa_state(-1, 0)) is False
    assert stopper(_gepa_state(0, 0)) is False
    assert stopper(_gepa_state(0, 0)) is False  # same iteration checked twice counts once
    assert stopper(_gepa_state(1, 1)) is False  # a live run resets the streak
    assert stopper(_gepa_state(2, 1)) is False
    assert stopper(_gepa_state(3, 1)) is True


def test_optimize_stopped_for_no_progress_records_a_distinct_reason(
    source_repo, fake_practice, tmp_path, monkeypatch, capsys,
):
    import gepa

    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    monkeypatch.setenv(REFLECTION_LM_API_KEY_ENV, "fake-key")

    def fake_optimize(**kwargs):
        (stopper,) = [s for s in kwargs["stop_callbacks"] if isinstance(s, run_gepa.NoProgressStopper)]
        i = -1  # GEPA checks once before iteration 0
        while not stopper(_gepa_state(i, 3)):
            i += 1
        assert i + 1 == run_gepa.NO_PROGRESS_MAX_IDLE_ITERATIONS  # iterations 0..i, all idle
        return type("Result", (), {"best_candidate": dict(kwargs["seed_candidate"]),
                                   "total_metric_calls": 3})()

    monkeypatch.setattr(gepa, "optimize", fake_optimize)
    out_dir = tmp_path / "run_out"
    code = _run_main(_optimize_argv(source_repo, fake_practice, tmp_path, out_dir))
    assert code == 0
    assert (out_dir / "optimized_components.yaml").exists()
    assert "no metric call" in capsys.readouterr().err.lower()
    run_end = [json.loads(line) for line in (out_dir / "spend_log.jsonl").read_text().splitlines()][-1]
    assert run_end["event"] == "run_end"
    assert run_end["reason"] == "no_progress"
    assert run_end["total_metric_calls"] == 3


def _parent_evaluated(iteration):
    return {"iteration": iteration, "candidate_idx": 0}


def _parent_skipped(iteration):
    return {"iteration": iteration, "candidate_idx": 0, "reason": "all_scores_perfect"}


def _child_evaluation_started(iteration):
    return {"iteration": iteration, "candidate_idx": None}


def test_no_progress_stopper_flags_an_idle_window_where_reflection_never_produced_a_child():
    """GEPA 0.1.4 swallows a reflection exception (reflective_mutation.py
    _propose_texts_batch_safe), so a broken reflection model evaluates no
    child and charges nothing -- the same zero-eval signature as a perfect
    cache hit, but nothing was optimized."""
    stopper = run_gepa.NoProgressStopper(max_idle_iterations=2)
    stopper(_gepa_state(-1, 4))
    for i in range(3):
        stopper.on_evaluation_end(_parent_evaluated(i))
        stopper(_gepa_state(i, 4))
    assert stopper.tripped is True
    assert stopper.no_proposals is True


def test_no_progress_stopper_is_not_no_proposals_when_a_cached_child_was_evaluated():
    stopper = run_gepa.NoProgressStopper(max_idle_iterations=2)
    stopper(_gepa_state(-1, 4))
    for i in range(3):
        stopper.on_evaluation_end(_parent_evaluated(i))
        stopper.on_evaluation_start(_child_evaluation_started(i))
        stopper(_gepa_state(i, 4))
    assert stopper.tripped is True
    assert stopper.no_proposals is False


def test_no_progress_stopper_is_not_no_proposals_when_every_parent_was_skipped_as_perfect():
    stopper = run_gepa.NoProgressStopper(max_idle_iterations=2)
    stopper(_gepa_state(-1, 4))
    for i in range(3):
        stopper.on_evaluation_start({"iteration": i, "candidate_idx": 0})
        stopper.on_evaluation_end(_parent_evaluated(i))
        stopper.on_evaluation_skipped(_parent_skipped(i))
        stopper(_gepa_state(i, 4))
    assert stopper.tripped is True
    assert stopper.no_proposals is False


def test_no_progress_stopper_forgets_a_childless_iteration_that_charged_a_metric_call():
    stopper = run_gepa.NoProgressStopper(max_idle_iterations=2)
    stopper(_gepa_state(-1, 0))
    stopper.on_evaluation_end(_parent_evaluated(0))  # a reflection failure, but it paid a live run
    stopper(_gepa_state(0, 2))
    for i in (1, 2):
        stopper.on_evaluation_end(_parent_evaluated(i))
        stopper.on_evaluation_skipped(_parent_skipped(i))
        stopper(_gepa_state(i, 2))
    assert stopper.tripped is True
    assert stopper.no_proposals is False


@pytest.mark.parametrize("emit_child,expected_code,expected_reason", [
    (False, 5, "no_proposals"),
    (True, 0, "no_progress"),
])
def test_optimize_stopped_without_any_proposal_exits_non_zero_as_no_proposals(
    source_repo, fake_practice, tmp_path, monkeypatch, capsys, emit_child, expected_code, expected_reason,
):
    """Without a child evaluation in the idle window, a run whose reflection
    model was failing used to exit 0 as no_progress with the seed written as
    optimized_components.yaml."""
    import gepa

    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    monkeypatch.setenv(REFLECTION_LM_API_KEY_ENV, "fake-key")

    def fake_optimize(**kwargs):
        (stopper,) = [s for s in kwargs["stop_callbacks"] if isinstance(s, run_gepa.NoProgressStopper)]
        callbacks = kwargs["callbacks"]

        def notify(method, event):
            for cb in callbacks:
                if hasattr(cb, method):
                    getattr(cb, method)(event)

        i = -1
        while not stopper(_gepa_state(i, 3)):
            i += 1
            notify("on_evaluation_start", {"iteration": i, "candidate_idx": 0})
            notify("on_evaluation_end", _parent_evaluated(i))
            if emit_child:
                notify("on_evaluation_start", _child_evaluation_started(i))
        return type("Result", (), {"best_candidate": dict(kwargs["seed_candidate"]),
                                   "total_metric_calls": 3})()

    monkeypatch.setattr(gepa, "optimize", fake_optimize)
    out_dir = tmp_path / "run_out"
    code = _run_main(_optimize_argv(source_repo, fake_practice, tmp_path, out_dir))
    assert code == expected_code
    assert (out_dir / "optimized_components.yaml").exists()
    err = capsys.readouterr().err.lower()
    assert ("reflection" in err and "not optimized" in err) is (not emit_child)
    run_end = [json.loads(line) for line in (out_dir / "spend_log.jsonl").read_text().splitlines()][-1]
    assert run_end["reason"] == expected_reason


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1"])
@pytest.mark.parametrize("mode", ["--estimate-only", "--baseline-only"])
def test_refuses_a_non_finite_or_non_positive_max_wall_clock_before_writing_anything(
    source_repo, fake_practice, tmp_path, monkeypatch, capsys, value, mode,
):
    """A NaN deadline makes every LiveBudget/run_batch deadline comparison
    False, silently disabling the wall-clock backstop."""
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    out_dir = tmp_path / "run_out"
    code = _run_main([
        "--repo-root", str(source_repo), "--components-config", "config/components.yaml",
        "--benchmark-root", str(fake_practice), "--exercises", "wordy,acronym",
        "--exercise-count", "2", "--val-count", "1",
        "--model", "gpt-fake", "--confirm-live-rollouts", "--max-metric-calls", "5",
        "--out-dir", str(out_dir), "--scratch-dir", str(tmp_path / "scratch"),
        "--pi-bin", str(FAKE_PI), "--max-wall-clock-s", value, mode, "--yes",
    ])
    assert code == 1
    assert "--max-wall-clock-s" in capsys.readouterr().err
    assert not out_dir.exists()


def test_baseline_run_writes_the_manifest_before_any_spend(source_repo, fake_practice, tmp_path, monkeypatch):
    """manifest.yaml pre-registers the run: the search split is exactly what
    GEPA/baseline evaluates, acceptance and test hold the rest of the pool,
    and it is already on disk when the first exercise runs."""
    from benchmarks.self_improve.live_eval import LiveRunResult, PolyglotLiveRunner
    from benchmarks.self_improve.manifest import Manifest

    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    out_dir = tmp_path / "run_out"
    seen_at_first_run = []

    def fake_run_batch(self, candidate, specs, *, sample_index=0):
        seen_at_first_run.append((out_dir / "manifest.yaml").exists())
        (spec,) = specs
        return [LiveRunResult(task_id=spec.task_id, exercise=spec.exercise, language=spec.language,
                              status="pass_1", score=1.0, success=True)]

    monkeypatch.setattr(PolyglotLiveRunner, "run_batch", fake_run_batch)
    code = _run_main([
        "--repo-root", str(source_repo), "--components-config", "config/components.yaml",
        "--benchmark-root", str(fake_practice),
        "--exercises", "wordy,acronym", "--exercise-count", "2", "--val-count", "1",
        "--model", "gpt-fake", "--confirm-live-rollouts", "--max-metric-calls", "5",
        "--max-wall-clock-s", "600", "--temperature", "0.7", "--seed", "9",
        "--out-dir", str(out_dir), "--scratch-dir", str(tmp_path / "scratch"),
        "--pi-bin", str(FAKE_PI), "--baseline-only", "--yes",
    ])
    assert code == 0
    assert seen_at_first_run and seen_at_first_run[0] is True
    m = Manifest.load(out_dir / "manifest.yaml")
    assert m.splits["search"] == ["wordy", "acronym"]
    assert sorted(m.splits["acceptance"] + m.splits["test"]) == ["bob", "isogram", "leap"]
    assert m.seed == 9
    assert m.language == "python"
    assert m.sampling["temperature"] == 0.7
    assert m.budget == {"max_metric_calls": 5, "max_wall_clock_s": 600.0}
    seed = yaml.safe_load((source_repo / "config" / "components.yaml").read_text())
    assert m.searchable_components == sorted(seed)
    assert m.env_fingerprint["model"] == "gpt-fake"


def test_estimate_only_writes_no_manifest(source_repo, fake_practice, tmp_path, monkeypatch):
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    out_dir = tmp_path / "run_out"
    code = _run_main([
        "--repo-root", str(source_repo), "--components-config", "config/components.yaml",
        "--benchmark-root", str(fake_practice), "--exercise-count", "3", "--val-count", "1",
        "--out-dir", str(out_dir), "--estimate-only",
    ])
    assert code == 0
    assert not (out_dir / "manifest.yaml").exists()


def test_refuses_before_spend_when_nothing_is_left_for_acceptance_and_test(
    source_repo, fake_practice, tmp_path, monkeypatch, capsys,
):
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    out_dir = tmp_path / "run_out"
    code = _run_main([
        "--repo-root", str(source_repo), "--components-config", "config/components.yaml",
        "--benchmark-root", str(fake_practice), "--exercises", "wordy,acronym,leap,bob",
        "--exercise-count", "4", "--val-count", "1",
        "--model", "gpt-fake", "--confirm-live-rollouts", "--max-metric-calls", "5",
        "--out-dir", str(out_dir), "--baseline-only", "--yes",
    ])
    assert code == 1
    assert "acceptance" in capsys.readouterr().err
    assert not out_dir.exists()


def test_refuses_a_non_positive_temperature_before_spend(source_repo, fake_practice, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    out_dir = tmp_path / "run_out"
    code = _run_main([
        "--repo-root", str(source_repo), "--components-config", "config/components.yaml",
        "--benchmark-root", str(fake_practice), "--exercises", "wordy,acronym",
        "--exercise-count", "2", "--val-count", "1", "--temperature", "0",
        "--model", "gpt-fake", "--confirm-live-rollouts", "--max-metric-calls", "5",
        "--out-dir", str(out_dir), "--scratch-dir", str(tmp_path / "scratch"),
        "--baseline-only", "--yes",
    ])
    assert code == 1
    assert "temperature" in capsys.readouterr().err
    assert not (out_dir / "manifest.yaml").exists()
    assert not (out_dir / "spend_log.jsonl").exists()


def _baseline_argv(source_repo, fake_practice, tmp_path, out_dir, *extra):
    return [
        "--repo-root", str(source_repo), "--components-config", "config/components.yaml",
        "--benchmark-root", str(fake_practice),
        "--exercises", "wordy,acronym", "--exercise-count", "2", "--val-count", "1",
        "--model", "gpt-fake", "--confirm-live-rollouts", "--max-metric-calls", "5",
        "--out-dir", str(out_dir), "--scratch-dir", str(tmp_path / "scratch"),
        "--pi-bin", str(FAKE_PI), "--baseline-only", "--yes", *extra,
    ]


def _fake_pass_run_batch(calls):
    from benchmarks.self_improve.live_eval import LiveRunResult

    def fake_run_batch(self, candidate, specs, *, sample_index=0):
        calls.append([s.task_id for s in specs])
        return [LiveRunResult(task_id=s.task_id, exercise=s.exercise, language=s.language,
                              status="pass_1", score=1.0, success=True) for s in specs]
    return fake_run_batch


def test_refuses_to_reuse_an_out_dir_that_already_holds_a_manifest(
    source_repo, fake_practice, tmp_path, monkeypatch, capsys,
):
    """One out-dir is one pre-registered run. A second run into the same dir
    must refuse before any spend and leave the first manifest byte-for-byte."""
    from benchmarks.self_improve.live_eval import PolyglotLiveRunner

    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    calls = []
    monkeypatch.setattr(PolyglotLiveRunner, "run_batch", _fake_pass_run_batch(calls))
    out_dir = tmp_path / "run_out"
    assert _run_main(_baseline_argv(source_repo, fake_practice, tmp_path, out_dir)) == 0
    manifest_bytes = (out_dir / "manifest.yaml").read_bytes()
    calls.clear()
    capsys.readouterr()

    code = _run_main(_baseline_argv(source_repo, fake_practice, tmp_path, out_dir, "--seed", "7"))
    assert code == 1
    assert calls == []
    assert (out_dir / "manifest.yaml").read_bytes() == manifest_bytes
    err = capsys.readouterr().err
    assert "manifest.yaml" in err
    assert "--live-cache-dir" in err


def test_refuses_an_out_dir_with_leftover_gepa_state_and_never_calls_optimize(
    source_repo, fake_practice, tmp_path, monkeypatch,
):
    """gepa.optimize() auto-resumes from <run_dir>/gepa_state.bin, so an
    out-dir holding one must be refused before optimize is ever reached."""
    import gepa

    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    monkeypatch.setenv(REFLECTION_LM_API_KEY_ENV, "fake-key")
    optimize_calls = []
    monkeypatch.setattr(gepa, "optimize", lambda **kw: optimize_calls.append(kw))
    out_dir = tmp_path / "run_out"
    (out_dir / "gepa").mkdir(parents=True)
    (out_dir / "gepa" / "gepa_state.bin").write_bytes(b"state")

    code = _run_main(_optimize_argv(source_repo, fake_practice, tmp_path, out_dir))
    assert code == 1
    assert optimize_calls == []
    assert not (out_dir / "manifest.yaml").exists()
    assert not (out_dir / "spend_log.jsonl").exists()


@pytest.mark.parametrize("marker", [
    "manifest.yaml", "gepa/gepa_state.bin", "optimized_components.yaml",
    "seed_baseline.json", "spend_log.jsonl",
])
def test_every_prior_run_marker_refuses_a_baseline_run(
    source_repo, fake_practice, tmp_path, monkeypatch, marker,
):
    from benchmarks.self_improve.live_eval import PolyglotLiveRunner

    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    calls = []
    monkeypatch.setattr(PolyglotLiveRunner, "run_batch", _fake_pass_run_batch(calls))
    out_dir = tmp_path / "run_out"
    (out_dir / marker).parent.mkdir(parents=True, exist_ok=True)
    (out_dir / marker).write_text("prior")

    assert _run_main(_baseline_argv(source_repo, fake_practice, tmp_path, out_dir)) == 1
    assert calls == []
    assert (out_dir / marker).read_text() == "prior"


def test_an_out_dir_holding_only_a_live_cache_is_accepted(source_repo, fake_practice, tmp_path, monkeypatch):
    """A warm live_cache/ is not prior run state -- only the cache the next
    run may reuse -- so it must not trip the reuse refusal."""
    from benchmarks.self_improve.live_eval import PolyglotLiveRunner

    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    calls = []
    monkeypatch.setattr(PolyglotLiveRunner, "run_batch", _fake_pass_run_batch(calls))
    out_dir = tmp_path / "run_out"
    (out_dir / "live_cache").mkdir(parents=True)
    (out_dir / "live_cache" / "entry.json").write_text("{}")

    assert _run_main(_baseline_argv(source_repo, fake_practice, tmp_path, out_dir)) == 0
    assert calls
    assert (out_dir / "manifest.yaml").exists()


def test_estimate_only_on_an_out_dir_with_prior_run_state_writes_nothing(
    source_repo, fake_practice, tmp_path, monkeypatch,
):
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    out_dir = tmp_path / "run_out"
    (out_dir / "gepa").mkdir(parents=True)
    (out_dir / "manifest.yaml").write_text("prior")
    (out_dir / "gepa" / "gepa_state.bin").write_bytes(b"state")

    def snapshot():
        return {p.relative_to(out_dir): p.read_bytes() for p in out_dir.rglob("*") if p.is_file()}

    before = snapshot()
    code = _run_main([
        "--repo-root", str(source_repo), "--components-config", "config/components.yaml",
        "--benchmark-root", str(fake_practice), "--exercise-count", "3", "--val-count", "1",
        "--out-dir", str(out_dir), "--estimate-only",
    ])
    assert code == 0
    assert snapshot() == before


def test_refuses_a_leftover_nested_gepa_stop_file(source_repo, fake_practice, tmp_path, monkeypatch):
    """GEPA adds its own FileStopper at <run_dir>/gepa.stop (run_dir is
    <out-dir>/gepa), so a leftover there would stop the run at the first
    check exactly like the top-level one."""
    from benchmarks.self_improve.live_eval import PolyglotLiveRunner

    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    calls = []
    monkeypatch.setattr(PolyglotLiveRunner, "run_batch", _fake_pass_run_batch(calls))
    out_dir = tmp_path / "run_out"
    (out_dir / "gepa").mkdir(parents=True)
    (out_dir / "gepa" / "gepa.stop").write_text("")

    assert _run_main(_baseline_argv(source_repo, fake_practice, tmp_path, out_dir)) == 1
    assert calls == []
    assert (out_dir / "gepa" / "gepa.stop").exists()
    assert not (out_dir / "manifest.yaml").exists()


def _env_dumping_pi(tmp_path: Path) -> Path:
    """A pi stand-in that records its own environment, then becomes fake_pi.
    A /bin/sh wrapper avoids a long `#!<venv python>` shebang."""
    import shlex
    dumper = tmp_path / "dump_env.py"
    dumper.write_text(
        "import json, os\n"
        "d = os.environ['PI_ENV_DUMP_DIR']\n"
        "with open(os.path.join(d, f'{os.getppid()}.json'), 'w') as fh:\n"
        "    json.dump(dict(os.environ), fh)\n"
    )
    wrapper = tmp_path / "pi-env-dump"
    py = shlex.quote(sys.executable)
    wrapper.write_text(
        "#!/bin/sh\n"
        f"{py} {shlex.quote(str(dumper))} || exit 1\n"
        f"exec {py} {shlex.quote(str(FAKE_PI))} \"$@\"\n"
    )
    wrapper.chmod(0o755)
    return wrapper


def test_baseline_run_withholds_orchestrator_only_env_from_the_agent(
    source_repo, fake_practice, tmp_path, monkeypatch,
):
    """The reflection LM key and every .env name reach neither pi nor pi's
    bash, while the model-under-test's own key does; the orchestrator's own
    environment keeps the reflection key."""
    import os
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    monkeypatch.setenv("ATTEMPT_TIMEOUT_S", "30")
    monkeypatch.setenv("FAKE_PI_MODE", "solve_from_env")
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({"wordy.py": _b64(_WORDY_SOLUTION)}))
    monkeypatch.setenv(REFLECTION_LM_API_KEY_ENV, "sk-sentinel-reflection")
    monkeypatch.setenv("SOME_DOTENV_SECRET", "dotenv-sentinel")
    monkeypatch.setenv("OMLX_API_KEY", "model-sentinel")
    dump_dir = tmp_path / "env_dumps"
    dump_dir.mkdir()
    monkeypatch.setenv("PI_ENV_DUMP_DIR", str(dump_dir))
    monkeypatch.setattr(run_gepa, "_DOTENV_KEYS", frozenset({REFLECTION_LM_API_KEY_ENV, "SOME_DOTENV_SECRET"}))

    code = _run_main([
        "--repo-root", str(source_repo), "--components-config", "config/components.yaml",
        "--benchmark-root", str(fake_practice),
        "--exercises", "wordy,acronym", "--exercise-count", "2", "--val-count", "1",
        "--model", "gpt-fake", "--confirm-live-rollouts", "--max-metric-calls", "5",
        "--out-dir", str(tmp_path / "run_out"), "--scratch-dir", str(tmp_path / "scratch"),
        "--pi-bin", str(_env_dumping_pi(tmp_path)), "--baseline-only", "--yes", "--no-live-cache",
    ])
    assert code == 0
    dumps = [json.loads(p.read_text()) for p in dump_dir.glob("*.json")]
    assert dumps, "pi was never spawned, so nothing was checked"
    for p in dump_dir.glob("*.json"):
        p.unlink()  # each dump holds the developer's whole environment
    # Assertions compare names only, so a failure never prints env values.
    for env in dumps:
        leaked = [n for n in (REFLECTION_LM_API_KEY_ENV, "SOME_DOTENV_SECRET", "SELF_IMPROVE_DOTENV") if n in env]
        assert leaked == []
        sentinel_hits = sorted(k for k, v in env.items() if v in ("sk-sentinel-reflection", "dotenv-sentinel"))
        assert sentinel_hits == []
        assert env.get("OMLX_API_KEY") == "model-sentinel"
        assert "LLAMACPP_API_KEY" in sorted(env)
    assert os.environ[REFLECTION_LM_API_KEY_ENV] == "sk-sentinel-reflection"
    assert os.environ["SOME_DOTENV_SECRET"] == "dotenv-sentinel"


def test_a_shared_knob_in_dotenv_refuses_before_any_worktree_or_prompt(
    source_repo, fake_practice, tmp_path, monkeypatch, capsys,
):
    monkeypatch.delenv(NO_LIVE_ROLLOUTS_ENV, raising=False)
    monkeypatch.setattr(run_gepa, "_DOTENV_KEYS", frozenset({"ATTEMPT_TIMEOUT_S"}))
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("prompted despite the refusal"))
    scratch_dir = tmp_path / "scratch"

    code = _run_main([
        "--repo-root", str(source_repo), "--components-config", "config/components.yaml",
        "--benchmark-root", str(fake_practice),
        "--exercises", "wordy,acronym", "--exercise-count", "2", "--val-count", "1",
        "--model", "gpt-fake", "--confirm-live-rollouts", "--max-metric-calls", "5",
        "--out-dir", str(tmp_path / "run_out"), "--scratch-dir", str(scratch_dir),
        "--pi-bin", str(FAKE_PI), "--baseline-only",
    ])
    assert code == 1
    assert ("ATTEMPT_TIMEOUT_S is set in benchmarks/self_improve/.env, which is orchestrator-only; "
            "export it in your shell instead.") in capsys.readouterr().err
    assert not scratch_dir.exists()
    assert "gepa-scratch" not in subprocess.run(
        ["git", "worktree", "list"], cwd=source_repo, capture_output=True, text=True, check=True,
    ).stdout
