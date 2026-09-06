"""End-to-end live-eval pipeline test: REAL `git worktree add`, REAL
candidate file writes, a REAL subprocess invocation of aider_polyglot.py, a
REAL `pytest -x -q` scoring run, REAL result-JSON parsing -- with `pi`
routed through fake_pi.py so zero model calls happen. This is the centerpiece
test of the whole live-execution rewrite: it exercises the exact mechanism
(rpc_client.REPO_ROOT resolving to the SCRATCH worktree because
aider_polyglot.py is invoked as a subprocess whose own __file__ lives there)
that makes a candidate's text actually reach a live agent.
"""
import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from benchmarks.self_improve.exercises import ExerciseSpec
from benchmarks.self_improve.live_cache import LiveResultCache
from benchmarks.self_improve.live_eval import PolyglotLiveRunner
from benchmarks.self_improve.scratch_worktree import scratch_worktree

# This module unconditionally drives real `git init`/`git worktree add` and
# real subprocesses -- unlike the sibling fake_pi tests, it never skips on a
# minimal runner without git, which would otherwise hard-error every test
# here (masking the rest of the self_improve suite's own results) instead of
# skipping gracefully, the way the polyglot tests skip when
# node_modules/.bin/pi is absent.
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is required for this e2e suite")

REAL_REPO_ROOT = Path(__file__).resolve().parents[3]  # little-coder-self-improve/
FAKE_PI = REAL_REPO_ROOT / "benchmarks" / "fake_pi.py"

_WORDY_TEST = '''
import wordy

def test_wordy():
    assert wordy.solve() == "42"
'''
_WORDY_STUB = "def solve():\n    pass\n"
_WORDY_SOLUTION = 'def solve():\n    return "42"\n'


@pytest.fixture
def source_repo(tmp_path):
    """A throwaway git repo with committed COPIES (not stubs) of the real
    aider_polyglot.py/rpc_client.py, real-shaped AGENTS.md/skill file, and a
    components.yaml mapping -- copies, not symlinks, so the worktree
    contains real files at real paths and rpc_client.REPO_ROOT resolves to
    it, exactly like a real scratch checkout."""
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


@pytest.fixture
def fake_practice(tmp_path):
    """One real exercise: a stub that fails a genuine pytest test until
    solved, matching aider_polyglot.py's own _prepare_python/_run_python
    layout exactly (glob *.py minus *_test.py for stubs, *_test.py for tests)."""
    practice_root = tmp_path / "polyglot-benchmark"
    ex_dir = practice_root / "python" / "exercises" / "practice" / "wordy"
    ex_dir.mkdir(parents=True)
    (ex_dir / "wordy.py").write_text(_WORDY_STUB)
    (ex_dir / "wordy_test.py").write_text(_WORDY_TEST)
    return practice_root


@pytest.fixture
def runner_factory(source_repo, fake_practice, tmp_path, monkeypatch):
    """Real env for the child subprocess: fast attempt timeout (this suite
    must not be able to hang for 900s), pi routed through fake_pi.py."""
    monkeypatch.setenv("ATTEMPT_TIMEOUT_S", "30")
    monkeypatch.setenv("LITTLE_CODER_PI_BIN_OVERRIDE", str(FAKE_PI))

    def _make(cache=None):
        with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=FAKE_PI) as wt:
            yield PolyglotLiveRunner(
                worktree=wt,
                components_yaml=source_repo / "config" / "components.yaml",
                model="fake/model",
                max_attempts=2,
                benchmark_root=fake_practice,
                cache=cache,
                per_exercise_timeout_s=60,
            )

    return _make


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def test_pipeline_scores_a_pass_as_one(runner_factory, monkeypatch):
    monkeypatch.setenv("FAKE_PI_MODE", "solve_from_env")
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({"wordy.py": _b64(_WORDY_SOLUTION)}))
    for runner in runner_factory():
        results = runner.run_batch({"skills_tools_bash": "Revised guidance.\n"}, [ExerciseSpec("wordy")])
    assert results[0].status == "pass_1"
    assert results[0].score == 1.0
    assert results[0].success is True


def test_pipeline_scores_a_genuine_failure_as_zero(runner_factory, monkeypatch):
    monkeypatch.setenv("FAKE_PI_MODE", "clean")  # writes nothing relevant
    for runner in runner_factory():
        results = runner.run_batch({"skills_tools_bash": "Revised guidance.\n"}, [ExerciseSpec("wordy")])
    assert results[0].status == "fail"
    assert results[0].score == 0.0
    assert results[0].success is False
    assert "assert" in results[0].test_output_tail.lower() or "fail" in results[0].test_output_tail.lower()


def test_pipeline_scores_a_second_attempt_pass_as_partial_credit(runner_factory, tmp_path, monkeypatch):
    """noop_then_solve writes nothing on attempt 1, the real solution on
    attempt 2 -- attempts are separate PROCESSES (a fresh PiRpc/fake_pi per
    attempt), so this also proves the graded formula flows end to end."""
    monkeypatch.setenv("FAKE_PI_MODE", "noop_then_solve")
    monkeypatch.setenv("FAKE_PI_STATE_FILE", str(tmp_path / "attempt_state.txt"))
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({"wordy.py": _b64(_WORDY_SOLUTION)}))
    for runner in runner_factory():
        results = runner.run_batch({"skills_tools_bash": "Revised guidance.\n"}, [ExerciseSpec("wordy")])
    assert results[0].status == "pass_2"
    assert results[0].score == 0.7


def test_materialize_writes_candidate_text_preserving_frontmatter(runner_factory):
    for runner in runner_factory():
        runner.materialize({"skills_tools_bash": "Brand new guidance body.\n"})
        written = (runner.worktree.path / "skills" / "tools" / "bash.md").read_text()
    assert "name: bash" in written  # frontmatter preserved byte-for-byte
    assert "Brand new guidance body." in written
    assert "Original bash guidance." not in written


def test_candidate_text_actually_reaches_the_agent(runner_factory, tmp_path, monkeypatch):
    """The regression test for the mechanism itself: fake_pi echoes the
    REAL --system-prompt file it received back to a marker file, proving
    aider_polyglot.py (run as a subprocess FROM the scratch worktree) picked
    up rpc_client.REPO_ROOT as the worktree, not the source repo."""
    echo_file = tmp_path / "echo.txt"
    monkeypatch.setenv("FAKE_PI_MODE", "read_system_prompt_echo")
    monkeypatch.setenv("FAKE_PI_ECHO_FILE", str(echo_file))
    for runner in runner_factory():
        runner.run_batch({"agents_md": "DISTINCTIVE CANDIDATE TEXT 99887766\n"}, [ExerciseSpec("wordy")])
    assert "DISTINCTIVE CANDIDATE TEXT 99887766" in echo_file.read_text()


def test_two_candidates_that_differ_only_in_text_get_different_scores(runner_factory, monkeypatch):
    """THE regression test for the bug this entire live-execution rewrite
    exists to fix: under the old frozen-historical-data design, metric()
    never read the candidate's actual text, so every candidate scored
    identically and GEPA's acceptance criterion could never fire. Here, two
    candidates differing ONLY in instruction text produce genuinely
    different real outcomes, because fake_pi decides whether to solve based
    on what it actually reads from the live system prompt."""
    monkeypatch.setenv("FAKE_PI_MODE", "solve_if_prompt_contains")
    monkeypatch.setenv("FAKE_PI_MAGIC_TOKEN", "OPEN-SESAME-42")
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({"wordy.py": _b64(_WORDY_SOLUTION)}))

    for runner in runner_factory():
        good_candidate = {"agents_md": "Instructions containing OPEN-SESAME-42 the magic phrase.\n"}
        results_good = runner.run_batch(good_candidate, [ExerciseSpec("wordy")])

    for runner in runner_factory():
        bad_candidate = {"agents_md": "Unrelated instructions with no special phrase.\n"}
        results_bad = runner.run_batch(bad_candidate, [ExerciseSpec("wordy")])

    assert results_good[0].score != results_bad[0].score
    assert results_good[0].success is True
    assert results_bad[0].success is False


def test_cache_hit_skips_the_subprocess_entirely(runner_factory, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_PI_MODE", "solve_from_env")
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({"wordy.py": _b64(_WORDY_SOLUTION)}))
    cache = LiveResultCache(tmp_path / "cache")
    candidate = {"skills_tools_bash": "Some guidance.\n"}

    for runner in runner_factory(cache=cache):
        first = runner.run_batch(candidate, [ExerciseSpec("wordy")])

    for runner in runner_factory(cache=cache):
        # Prove the cache hit means "no subprocess at all", not just "a
        # subprocess ran and happened to agree" -- the strongest possible
        # assertion, stronger than comparing results alone.
        def _must_not_run(*a, **kw):
            raise AssertionError("cache hit should have skipped this exercise entirely")
        monkeypatch.setattr(runner, "_run_one_uncached", _must_not_run)
        second = runner.run_batch(candidate, [ExerciseSpec("wordy")])

    assert first[0].from_cache is False
    assert second[0].from_cache is True
    assert second[0].status == first[0].status
    assert second[0].score == first[0].score
