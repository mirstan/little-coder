"""gepa_scratch_gc.py: all tests run against a THROWAWAY `git init` repo in
tmp_path -- never the real checkout."""
import json
import os
import shutil
import subprocess
import time

import pytest

from benchmarks.self_improve.gepa_scratch_gc import find_scratch_worktrees, main
from benchmarks.self_improve.scratch_worktree import SCRATCH_MARKER_NAME, scratch_worktree


@pytest.fixture
def source_repo(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "AGENTS.md").write_text("body\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo, check=True)
    return repo


def _make_branch_worktree(source_repo, tmp_path, name):
    """A normal, real (non-scratch) worktree with a branch checked out --
    must never be touched by the GC, regardless of what's inside it."""
    path = tmp_path / f"branch-worktree-{name}"
    subprocess.run(["git", "worktree", "add", "-b", name, str(path)], cwd=source_repo,
                    check=True, capture_output=True)
    return path


def test_find_marks_a_worktree_with_no_marker_as_not_removable(source_repo, tmp_path):
    branch_wt = _make_branch_worktree(source_repo, tmp_path, "feature-x")
    entries = find_scratch_worktrees(source_repo, scratch_root=tmp_path)
    matching = [e for e in entries if e["path"] == branch_wt]
    assert len(matching) == 1
    assert matching[0]["removable"] is False
    assert "branch" in matching[0]["reason"]


def test_find_marks_dead_pid_scratch_worktree_as_removable(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi", keep=True) as wt:
        scratch_path = wt.path
    marker = json.loads((scratch_path / SCRATCH_MARKER_NAME).read_text())
    marker["pid"] = 999999999  # extremely unlikely to be a live pid
    (scratch_path / SCRATCH_MARKER_NAME).write_text(json.dumps(marker))

    entries = find_scratch_worktrees(source_repo, scratch_root=tmp_path)
    matching = [e for e in entries if e["path"] == scratch_path]
    assert len(matching) == 1
    assert matching[0]["removable"] is True

    # manual cleanup
    subprocess.run(["git", "worktree", "remove", "--force", str(scratch_path)], cwd=source_repo, check=True)


def test_find_marks_alive_pid_scratch_worktree_as_not_removable(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi", keep=True) as wt:
        scratch_path = wt.path
        marker = json.loads((scratch_path / SCRATCH_MARKER_NAME).read_text())
        assert marker["pid"] == os.getpid()  # this test process is very much alive

        entries = find_scratch_worktrees(source_repo, scratch_root=tmp_path)
        matching = [e for e in entries if e["path"] == scratch_path]
        assert len(matching) == 1
        assert matching[0]["removable"] is False
        assert "still running" in matching[0]["reason"]


def test_find_never_touches_a_branch_worktree_even_with_marker_content_inside(source_repo, tmp_path):
    """Belt-and-braces: a marker file dropped into a real, branch-having
    worktree must never make it removable -- detachment is checked first."""
    branch_wt = _make_branch_worktree(source_repo, tmp_path, "feature-y")
    (branch_wt / SCRATCH_MARKER_NAME).write_text(json.dumps({
        "little_coder_self_improve_scratch": True, "pid": 999999999,
    }))
    entries = find_scratch_worktrees(source_repo, scratch_root=tmp_path)
    matching = [e for e in entries if e["path"] == branch_wt]
    assert matching[0]["removable"] is False
    assert "branch" in matching[0]["reason"]


def test_find_marks_prunable_gone_worktree_as_removable(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi", keep=True) as wt:
        scratch_path = wt.path
    shutil.rmtree(scratch_path)  # simulate a SIGKILL: dir gone, registration remains

    entries = find_scratch_worktrees(source_repo, scratch_root=tmp_path)
    matching = [e for e in entries if e["path"] == scratch_path]
    assert len(matching) == 1
    assert matching[0]["removable"] is True
    assert "prunable" in matching[0]["reason"]


def test_scratch_root_filter_excludes_worktrees_outside_it(source_repo, tmp_path):
    outside = tmp_path.parent / f"outside-{tmp_path.name}"
    outside.mkdir(exist_ok=True)
    branch_wt = _make_branch_worktree(source_repo, outside, "feature-z")
    try:
        entries = find_scratch_worktrees(source_repo, scratch_root=tmp_path)
        matching = [e for e in entries if e["path"] == branch_wt]
        assert matching[0]["removable"] is False
        assert "outside" in matching[0]["reason"]
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(branch_wt)], cwd=source_repo, check=True)
        shutil.rmtree(outside, ignore_errors=True)


def test_cli_clean_without_yes_prompts_and_respects_no(source_repo, tmp_path, monkeypatch, capsys):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi", keep=True) as wt:
        scratch_path = wt.path
    shutil.rmtree(scratch_path)

    monkeypatch.setattr("builtins.input", lambda _: "n")
    code = main(["--repo-root", str(source_repo), "--scratch-root", str(tmp_path), "--clean"])
    assert code == 2
    assert "Aborted" in capsys.readouterr().out
    # still registered -- nothing was removed
    result = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=source_repo,
                             capture_output=True, text=True, check=True)
    assert str(scratch_path) in result.stdout


def test_cli_clean_with_yes_removes_orphans(source_repo, tmp_path, capsys):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi", keep=True) as wt:
        scratch_path = wt.path
    shutil.rmtree(scratch_path)

    code = main(["--repo-root", str(source_repo), "--scratch-root", str(tmp_path), "--clean", "--yes"])
    assert code == 0
    assert "Removed 1 scratch worktree" in capsys.readouterr().out
    result = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=source_repo,
                             capture_output=True, text=True, check=True)
    assert str(scratch_path) not in result.stdout


def test_cli_clean_never_removes_a_real_branch_worktree(source_repo, tmp_path, capsys):
    branch_wt = _make_branch_worktree(source_repo, tmp_path, "feature-real")
    try:
        code = main(["--repo-root", str(source_repo), "--scratch-root", str(tmp_path), "--clean", "--yes"])
        assert code == 0
        assert "Nothing to clean" in capsys.readouterr().out
        result = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=source_repo,
                                 capture_output=True, text=True, check=True)
        assert str(branch_wt) in result.stdout
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(branch_wt)], cwd=source_repo, check=True)


def test_cli_clean_older_than_hours_filters_out_recent_orphans(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi", keep=True) as wt:
        scratch_path = wt.path
    shutil.rmtree(scratch_path)

    code = main(["--repo-root", str(source_repo), "--scratch-root", str(tmp_path),
                 "--clean", "--yes", "--older-than-hours", "999"])
    assert code == 0
    result = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=source_repo,
                             capture_output=True, text=True, check=True)
    assert str(scratch_path) in result.stdout  # too recent, not removed

    # manual cleanup
    subprocess.run(["git", "worktree", "prune"], cwd=source_repo, check=True)


def test_cli_list_prints_something_for_every_worktree(source_repo, tmp_path, capsys):
    branch_wt = _make_branch_worktree(source_repo, tmp_path, "feature-list")
    try:
        main(["--repo-root", str(source_repo), "--list"])
        out = capsys.readouterr().out
        assert str(source_repo) in out
        assert str(branch_wt) in out
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(branch_wt)], cwd=source_repo, check=True)
