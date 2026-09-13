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


def test_find_marks_prunable_gone_worktree_as_removable_when_scoped_and_name_matches(
    source_repo, tmp_path,
):
    """A directory-gone-but-prunable entry, under a correctly-configured
    --scratch-root AND named per our own scratch-worktree convention
    (gepa-scratch-<pid>-<hex8>) -- the tool's actual, common purpose:
    cleaning up its own crashed scratch checkouts."""
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi", keep=True) as wt:
        scratch_path = wt.path
    shutil.rmtree(scratch_path)  # simulate a SIGKILL: dir gone, registration remains

    entries = find_scratch_worktrees(source_repo, scratch_root=tmp_path)
    matching = [e for e in entries if e["path"] == scratch_path]
    assert len(matching) == 1
    assert matching[0]["removable"] is True
    assert "prunable" in matching[0]["reason"]


def test_find_does_not_auto_remove_prunable_gone_worktree_without_scratch_root(source_repo, tmp_path):
    """Without --scratch-root there is no marker left to check (it lived
    inside the now-gone directory) and no other verification that this was
    ever one of ours -- must not silently trust "detached + prunable" alone."""
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi", keep=True) as wt:
        scratch_path = wt.path
    shutil.rmtree(scratch_path)

    entries = find_scratch_worktrees(source_repo, scratch_root=None)
    matching = [e for e in entries if e["path"] == scratch_path]
    assert len(matching) == 1
    assert matching[0]["removable"] is False

    subprocess.run(["git", "worktree", "prune"], cwd=source_repo, check=True)


def test_find_does_not_auto_remove_prunable_gone_worktree_with_a_mismatched_name(
    source_repo, tmp_path,
):
    """Path containment under a caller-given
    --scratch-root is NOT ownership evidence on its own -- an unrelated
    tool's own detached worktree could happen to live under the same
    directory (a shared tmp root, say). Simulate that by creating a
    worktree directly (not via scratch_worktree(), so it has no marker and
    a name that doesn't match our naming convention) under the configured
    scratch root, then removing its directory -- must stay non-removable
    even though it's "detached + prunable + under --scratch-root", since
    the one remaining signal (name) doesn't match either."""
    foreign_path = tmp_path / "not-ours-at-all"
    subprocess.run(["git", "worktree", "add", "--detach", str(foreign_path)],
                    cwd=source_repo, check=True, capture_output=True)
    shutil.rmtree(foreign_path)

    entries = find_scratch_worktrees(source_repo, scratch_root=tmp_path)
    matching = [e for e in entries if e["path"] == foreign_path]
    assert len(matching) == 1
    assert matching[0]["removable"] is False
    assert "no marker" in matching[0]["reason"]

    subprocess.run(["git", "worktree", "prune"], cwd=source_repo, check=True)


def test_find_marks_worktree_with_live_active_pid_as_not_removable(source_repo, tmp_path):
    """The orchestrator's own pid can die (SIGKILL) while a detached exercise
    subprocess -- its own process group, recorded separately -- is still
    alive and actively using the worktree; only checking the orchestrator's
    pid would call this safe to remove out from under that live process."""
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi", keep=True) as wt:
        scratch_path = wt.path
    marker = json.loads((scratch_path / SCRATCH_MARKER_NAME).read_text())
    marker["pid"] = 999999999  # orchestrator "dead"
    marker["active_pid"] = os.getpid()  # exercise subprocess "alive" (this test process)
    (scratch_path / SCRATCH_MARKER_NAME).write_text(json.dumps(marker))

    entries = find_scratch_worktrees(source_repo, scratch_root=tmp_path)
    matching = [e for e in entries if e["path"] == scratch_path]
    assert len(matching) == 1
    assert matching[0]["removable"] is False
    assert "exercise subprocess is still running" in matching[0]["reason"]

    subprocess.run(["git", "worktree", "remove", "--force", str(scratch_path)], cwd=source_repo, check=True)


def test_find_marks_worktree_removable_when_both_pid_and_active_pid_dead(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi", keep=True) as wt:
        scratch_path = wt.path
    marker = json.loads((scratch_path / SCRATCH_MARKER_NAME).read_text())
    marker["pid"] = 999999999
    marker["active_pid"] = 999999998
    (scratch_path / SCRATCH_MARKER_NAME).write_text(json.dumps(marker))

    entries = find_scratch_worktrees(source_repo, scratch_root=tmp_path)
    matching = [e for e in entries if e["path"] == scratch_path]
    assert matching[0]["removable"] is True

    subprocess.run(["git", "worktree", "remove", "--force", str(scratch_path)], cwd=source_repo, check=True)


def test_find_withholds_removal_while_spawn_pending_at_is_present(source_repo, tmp_path):
    """PolyglotLiveRunner writes
    spawn_pending_at (via mark_spawn_pending()) BEFORE subprocess.Popen(),
    then set_active_pid(proc.pid) only AFTER Popen() returns. A SIGKILL in
    that exact window leaves pid/active_pid both dead but spawn_pending_at
    present -- must NOT be treated as orphaned, since a subprocess may be
    mid-spawn (or have just started) and still alive."""
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi", keep=True) as wt:
        scratch_path = wt.path
    marker = json.loads((scratch_path / SCRATCH_MARKER_NAME).read_text())
    marker["pid"] = 999999999  # dead
    marker.pop("active_pid", None)  # never got written -- the exact gap
    marker["spawn_pending_at"] = time.time()  # just now
    (scratch_path / SCRATCH_MARKER_NAME).write_text(json.dumps(marker))

    entries = find_scratch_worktrees(source_repo, scratch_root=tmp_path)
    matching = [e for e in entries if e["path"] == scratch_path]
    assert matching[0]["removable"] is False
    assert "crashed mid-spawn" in matching[0]["reason"]

    subprocess.run(["git", "worktree", "remove", "--force", str(scratch_path)], cwd=source_repo, check=True)


def test_find_withholds_removal_even_when_spawn_pending_at_is_very_old(source_repo, tmp_path):
    """set_active_pid() (called from both the pid-recorded and the
    finally-clause None-cleared path in PolyglotLiveRunner) now always
    clears spawn_pending_at -- so a spawn_pending_at that is still present,
    no matter how old, can only mean the orchestrator crashed before ever
    reaching that call. There is deliberately no time-based expiry back to
    "orphaned": elapsed time is not evidence that nothing went wrong."""
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi", keep=True) as wt:
        scratch_path = wt.path
    marker = json.loads((scratch_path / SCRATCH_MARKER_NAME).read_text())
    marker["pid"] = 999999999
    marker.pop("active_pid", None)
    marker["spawn_pending_at"] = time.time() - 999999  # long ago
    (scratch_path / SCRATCH_MARKER_NAME).write_text(json.dumps(marker))

    entries = find_scratch_worktrees(source_repo, scratch_root=tmp_path)
    matching = [e for e in entries if e["path"] == scratch_path]
    assert matching[0]["removable"] is False
    assert "crashed mid-spawn" in matching[0]["reason"]

    subprocess.run(["git", "worktree", "remove", "--force", str(scratch_path)], cwd=source_repo, check=True)


def test_find_treats_non_dict_marker_as_unreadable_not_a_crash(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi", keep=True) as wt:
        scratch_path = wt.path
    (scratch_path / SCRATCH_MARKER_NAME).write_text(json.dumps([1, 2, 3]))

    entries = find_scratch_worktrees(source_repo, scratch_root=tmp_path)  # must not raise
    matching = [e for e in entries if e["path"] == scratch_path]
    assert matching[0]["removable"] is False

    subprocess.run(["git", "worktree", "remove", "--force", str(scratch_path)], cwd=source_repo, check=True)


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


def test_cli_clean_does_not_prune_prunable_entries_outside_its_own_scratch_root(source_repo, tmp_path):
    """An unconditional repo-wide `git
    worktree prune` after a scoped --clean would also deregister OTHER
    prunable worktrees never selected by this invocation's own filters,
    silently expanding a scoped cleanup into a repo-wide one."""
    # Both worktrees are created BEFORE either directory is removed: creating
    # a scratch worktree calls prune_stale() as its own setup step, which
    # would otherwise retroactively deregister an already-gone worktree
    # created earlier, confounding this test with unrelated behavior.
    other_root = tmp_path.parent / f"other-scratch-{tmp_path.name}"
    other_root.mkdir()
    with scratch_worktree(source_repo, parent_dir=other_root, pi_bin=tmp_path / "pi", keep=True) as wt:
        untouched_path = wt.path
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi", keep=True) as wt:
        target_path = wt.path

    shutil.rmtree(untouched_path)  # prunable, but under a DIFFERENT scratch root
    shutil.rmtree(target_path)

    try:
        code = main(["--repo-root", str(source_repo), "--scratch-root", str(tmp_path), "--clean", "--yes"])
        assert code == 0

        result = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=source_repo,
                                 capture_output=True, text=True, check=True)
        assert str(target_path) not in result.stdout  # the one we asked to clean
        assert str(untouched_path) in result.stdout  # outside scope -- must survive
    finally:
        # an assertion failure above used to
        # skip this cleanup entirely, leaking other_root (a sibling of the
        # pytest-managed tmp_path, so not auto-removed by pytest itself).
        subprocess.run(["git", "worktree", "prune"], cwd=source_repo, check=True)
        shutil.rmtree(other_root, ignore_errors=True)


def test_cli_list_prints_something_for_every_worktree(source_repo, tmp_path, capsys):
    branch_wt = _make_branch_worktree(source_repo, tmp_path, "feature-list")
    try:
        main(["--repo-root", str(source_repo), "--list"])
        out = capsys.readouterr().out
        assert str(source_repo) in out
        assert str(branch_wt) in out
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(branch_wt)], cwd=source_repo, check=True)
