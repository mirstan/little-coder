"""End-to-end test of the REAL gepa.optimize() call, not a stubbed adapter.

Regression cover for a real bug found running this system for real: GEPA's
own reflective_mutation.py:176 does `self.adapter.propose_new_texts is not
None` -- a DIRECT attribute access, not `getattr(..., None)` -- even though
the real GEPAAdapter Protocol declares `propose_new_texts: ProposalFn | None
= None` as a class-level default (core/adapter.py:204). Since
PolyglotGEPAAdapter deliberately does NOT inherit from the Protocol
(structural typing only, per its own module docstring), it never gets that
default for free, and never declared it itself either. The result: EVERY
reflection attempt raised AttributeError, which reflective_mutation.py
caught internally and treated as "no candidate proposed this iteration" --
not a crash, so raise_on_exception=True never fired -- and GEPA just kept
selecting the same seed program and retrying reflection forever. A real run
against this bug logged ~9,000 failed iterations in a few minutes, 100% CPU,
zero real reflection LM calls, bounded only by --max-wall-clock-s.

None of test_polyglot_adapter.py's tests (a stubbed FakeRunner, never
touching GEPA's own reflective_mutation.py at all) could have caught this --
it's specifically a gap in how PolyglotGEPAAdapter satisfies GEPAAdapter's
structural contract, only reachable by calling the real gepa.optimize().

Real git worktree, real subprocess invocation of aider_polyglot.py, `pi`
routed through fake_pi.py -- zero live-model cost. reflection_lm is a plain
Python callable (a valid LanguageModel per gepa's own Protocol), not a model
string -- zero reflection-LM cost either.
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from benchmarks.self_improve.components import load_components
from benchmarks.self_improve.exercises import ExerciseSpec, split_train_val
from benchmarks.self_improve.live_eval import PolyglotLiveRunner
from benchmarks.self_improve.polyglot_adapter import PolyglotGEPAAdapter
from benchmarks.self_improve.scratch_worktree import scratch_worktree

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is required for this e2e suite")

REAL_REPO_ROOT = Path(__file__).resolve().parents[3]
FAKE_PI = REAL_REPO_ROOT / "benchmarks" / "fake_pi.py"

_TEST_TEMPLATE = 'import {name}\n\ndef test_{name}():\n    assert {name}.solve() == "42"\n'
_STUB = "def solve():\n    pass\n"
_SOLUTION = 'def solve():\n    return "42"\n'


@pytest.fixture
def source_repo(tmp_path):
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
    (repo / "config" / "components.yaml").write_text(yaml.dump({"skills_tools_bash": "skills/tools/bash.md"}))

    (repo / "benchmarks").mkdir()
    shutil.copy(REAL_REPO_ROOT / "benchmarks" / "aider_polyglot.py", repo / "benchmarks" / "aider_polyglot.py")
    shutil.copy(REAL_REPO_ROOT / "benchmarks" / "rpc_client.py", repo / "benchmarks" / "rpc_client.py")

    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo, check=True)
    return repo


@pytest.fixture
def fake_practice(tmp_path):
    """Two real exercises, both trivially solvable -- a genuine train/val
    split of size 1/1, enough for one real reflective-mutation iteration."""
    practice_root = tmp_path / "polyglot-benchmark"
    for name in ("alpha", "beta"):
        ex_dir = practice_root / "python" / "exercises" / "practice" / name
        ex_dir.mkdir(parents=True)
        (ex_dir / f"{name}.py").write_text(_STUB)
        (ex_dir / f"{name}_test.py").write_text(_TEST_TEMPLATE.format(name=name))
    return practice_root


def test_real_gepa_optimize_calls_the_reflection_lm_without_crashing(
    source_repo, fake_practice, tmp_path, monkeypatch,
):
    monkeypatch.setenv("ATTEMPT_TIMEOUT_S", "30")
    # "clean" (writes nothing) rather than a solved candidate: GEPA's own
    # skip_perfect_score=True default correctly skips reflection entirely
    # when the sampled minibatch already scores perfectly ("All subsample
    # scores perfect for parent 0. Skipping.") -- a genuine, correct GEPA
    # behavior, confirmed while building this test, not a bug. Every
    # exercise failing guarantees an imperfect minibatch regardless of which
    # one split_train_val() happens to place in the training set, so
    # reflection is reliably triggered.
    monkeypatch.setenv("FAKE_PI_MODE", "clean")

    reflection_calls: list[str] = []

    def fake_reflection_lm(prompt) -> str:
        reflection_calls.append(prompt if isinstance(prompt, str) else str(prompt))
        return "```\nRevised bash tool guidance, proposed by the fake reflection LM.\n```"

    specs = [ExerciseSpec("alpha"), ExerciseSpec("beta")]
    trainset, valset = split_train_val(specs, val_count=1, seed=1)

    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=FAKE_PI) as wt:
        runner = PolyglotLiveRunner(
            worktree=wt, components_yaml=source_repo / "config" / "components.yaml",
            model="fake/model", max_attempts=1, benchmark_root=fake_practice,
            per_exercise_timeout_s=60,
        )
        adapter = PolyglotGEPAAdapter(
            runner, component_paths={"skills_tools_bash": "skills/tools/bash.md"},
            practice_dir_path=fake_practice / "python" / "exercises" / "practice",
        )
        seed_candidate = load_components(source_repo / "config" / "components.yaml", repo_root=source_repo)

        import gepa
        from gepa.utils.stop_condition import TimeoutStopCondition

        result = gepa.optimize(
            seed_candidate=seed_candidate,
            trainset=trainset, valset=valset,
            adapter=adapter,
            reflection_lm=fake_reflection_lm,
            max_metric_calls=6,
            reflection_minibatch_size=1,
            module_selector="round_robin",
            use_merge=False,
            cache_evaluation=True,
            display_progress_bar=False,
            run_dir=str(tmp_path / "gepa_run"),
            seed=1,
            raise_on_exception=True,
            # Belt-and-braces: if this regresses again, the reflection LM
            # never gets called and GEPA just spins retrying reflection
            # forever (bounded only by wall clock, not max_metric_calls,
            # since no real evaluate() calls happen in that failure mode) --
            # this keeps a REGRESSED test fast-failing instead of hanging.
            stop_callbacks=[TimeoutStopCondition(90)],
        )

    # The actual regression assertion: the reflection LM must have been
    # reached and called for real. Before the fix, this list stays empty
    # forever -- reflective_mutation.py's AttributeError is caught
    # internally and logged as "no candidate proposed", never propagating
    # up as a raised exception, so the only observable symptom was silent
    # non-progress (and, in the real run, ~9,000 burned iterations).
    assert reflection_calls, (
        "reflection_lm was never called -- PolyglotGEPAAdapter is likely missing "
        "propose_new_texts = None again"
    )
    assert "skills_tools_bash" in result.best_candidate
