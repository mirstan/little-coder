"""gepa_scratch_gc.py: all tests run against THROWAWAY `git init` repos and
scratch roots in tmp_path -- never the real checkout."""
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid

import pytest

import benchmarks.self_improve.gepa_scratch_gc as gc
import benchmarks.self_improve.scratch_worktree as sw
from benchmarks.self_improve.gepa_scratch_gc import (
    find_legacy_worktrees,
    find_scratch_dirs,
    find_scratch_worktrees,
    main,
)
from benchmarks.self_improve.scratch_worktree import (
    LEGACY_SCRATCH_MARKER_NAME,
    scratch_git_dir,
    scratch_lock_path,
    scratch_marker_path,
    scratch_worktree,
)


def _git(cwd, *args, check=True):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=check)


@pytest.fixture
def source_repo(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "AGENTS.md").write_text("body\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "scratch-root"
    r.mkdir()
    return r.resolve()


def _siblings(path):
    return [path, scratch_git_dir(path), scratch_marker_path(path), scratch_lock_path(path)]


def _make_kept(source_repo, root):
    with scratch_worktree(source_repo, parent_dir=root, pi_bin=root / "pi", keep=True) as wt:
        pass
    return wt


def _edit_marker(path, **changes):
    marker_path = scratch_marker_path(path)
    marker = json.loads(marker_path.read_text())
    for k, v in changes.items():
        if v is None:
            marker.pop(k, None)
        else:
            marker[k] = v
    marker_path.write_text(json.dumps(marker))


def _forge_dead_marker(path):
    """What anyone running as this user can do: make every marker-based
    liveness signal say "dead"."""
    _edit_marker(path, pid=999999999, active_pid=999999998, spawn_pending_at=None)


def _entry(entries, path):
    matching = [e for e in entries if os.path.realpath(e["path"]) == os.path.realpath(path)]
    assert len(matching) == 1, entries
    return matching[0]


# --------------------------------------------------------------------------
# Private-repo layout: find_scratch_dirs
# --------------------------------------------------------------------------


def test_dead_pid_scratch_dir_is_removable(source_repo, root):
    wt = _make_kept(source_repo, root)
    _edit_marker(wt.path, pid=999999999)
    e = _entry(find_scratch_dirs(root), wt.path)
    assert e["removable"] is True
    assert e["kind"] == "private-repo"


def test_alive_pid_scratch_dir_is_not_removable(source_repo, root):
    wt = _make_kept(source_repo, root)
    assert json.loads(wt.marker_path.read_text())["pid"] == os.getpid()
    e = _entry(find_scratch_dirs(root), wt.path)
    assert e["removable"] is False
    assert "still running" in e["reason"]


def test_live_active_pid_is_not_removable(source_repo, root):
    wt = _make_kept(source_repo, root)
    _edit_marker(wt.path, pid=999999999, active_pid=os.getpid())
    e = _entry(find_scratch_dirs(root), wt.path)
    assert e["removable"] is False
    assert "exercise subprocess is still running" in e["reason"]


def test_both_pids_dead_is_removable(source_repo, root):
    wt = _make_kept(source_repo, root)
    _forge_dead_marker(wt.path)
    assert _entry(find_scratch_dirs(root), wt.path)["removable"] is True


@pytest.mark.parametrize("age_s", [0, 999999])
def test_spawn_pending_at_withholds_removal_however_old(source_repo, root, age_s):
    wt = _make_kept(source_repo, root)
    _edit_marker(wt.path, pid=999999999, active_pid=None, spawn_pending_at=time.time() - age_s)
    e = _entry(find_scratch_dirs(root), wt.path)
    assert e["removable"] is False
    assert "crashed mid-spawn" in e["reason"]


def test_non_dict_marker_is_not_ours_and_not_a_crash(source_repo, root):
    wt = _make_kept(source_repo, root)
    wt.marker_path.write_text(json.dumps([1, 2, 3]))
    e = _entry(find_scratch_dirs(root), wt.path)
    assert e["removable"] is False
    assert "not ours" in e["reason"]


def test_marker_that_is_a_symlink_is_not_followed(source_repo, root, tmp_path):
    wt = _make_kept(source_repo, root)
    real = tmp_path / "elsewhere.json"
    real.write_text(json.dumps({"little_coder_self_improve_scratch": True, "pid": 999999999}))
    wt.marker_path.unlink()
    wt.marker_path.symlink_to(real)
    e = _entry(find_scratch_dirs(root), wt.path)
    assert e["removable"] is False


def test_gc_ignores_dirs_with_matching_name_but_no_external_marker(root):
    d = root / "gepa-scratch-12345-deadbeef"
    d.mkdir()
    (d / "file.txt").write_text("someone else's\n")
    e = _entry(find_scratch_dirs(root), d)
    assert e["removable"] is False
    assert "no external marker" in e["reason"]
    assert main(["--scratch-root", str(root), "--repo-root", str(root), "--clean", "--yes"]) == 0
    assert (d / "file.txt").exists()


def test_gc_never_follows_a_symlinked_scratch_dir_out_of_the_root(root, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.txt").write_text("keep\n")
    base = root / "gepa-scratch-1-deadbeef"
    base.symlink_to(outside)
    scratch_marker_path(base).write_text(json.dumps({
        "little_coder_self_improve_scratch": True, "layout": "private-repo-v1",
        "pid": 999999999, "created_at": 0,
    }))
    e = _entry(find_scratch_dirs(root), base)
    assert e["removable"] is False
    assert "symlink" in e["reason"]
    main(["--scratch-root", str(root), "--repo-root", str(root), "--clean", "--yes"])
    assert (outside / "precious.txt").read_text() == "keep\n"
    assert base.is_symlink()


def test_symlinked_git_dir_is_not_removable(source_repo, root, tmp_path):
    wt = _make_kept(source_repo, root)
    _forge_dead_marker(wt.path)
    outside = tmp_path / "outside-g"
    outside.mkdir()
    shutil.rmtree(wt.git_dir)
    wt.git_dir.symlink_to(outside)
    e = _entry(find_scratch_dirs(root), wt.path)
    assert e["removable"] is False
    assert "symlink" in e["reason"]


def test_marker_present_tree_missing_is_removable_and_clean_removes_all_siblings(source_repo, root, capsys):
    """A crash mid-setup: the marker is written first, so it can outlive a
    tree that never got created (or a git dir that did)."""
    wt = _make_kept(source_repo, root)
    _forge_dead_marker(wt.path)
    shutil.rmtree(wt.path)
    scratch_lock_path(wt.path).touch()
    e = _entry(find_scratch_dirs(root), wt.path)
    assert e["removable"] is True
    assert "partially present" in e["reason"]
    assert main(["--scratch-root", str(root), "--repo-root", str(source_repo), "--clean", "--yes"]) == 0
    assert not any(os.path.lexists(p) for p in _siblings(wt.path))


def test_gc_scans_only_the_given_root(source_repo, root, tmp_path):
    other_root = tmp_path / "other-root"
    other_root.mkdir()
    other = _make_kept(source_repo, other_root)
    _forge_dead_marker(other.path)
    target = _make_kept(source_repo, root)
    _forge_dead_marker(target.path)
    assert main(["--scratch-root", str(root), "--repo-root", str(source_repo), "--clean", "--yes"]) == 0
    assert not target.path.exists()
    assert all(os.path.lexists(p) for p in _siblings(other.path)[:3])


def test_gc_default_root_is_gettempdir(source_repo, tmp_path, monkeypatch, capsys):
    fake_tmp = tmp_path / "fake-tmp"
    fake_tmp.mkdir()
    wt = _make_kept(source_repo, fake_tmp)
    _forge_dead_marker(wt.path)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_tmp))
    assert main(["--repo-root", str(source_repo), "--clean", "--yes"]) == 0
    out = capsys.readouterr().out
    assert f"Scanned scratch root: {fake_tmp.resolve()}" in out
    assert not wt.path.exists()


def test_forged_dead_marker_does_not_make_a_locked_scratch_dir_removable(source_repo, root, capsys):
    """The lock held by the live scratch_worktree() owner must win over a
    marker that says everything is dead."""
    with scratch_worktree(source_repo, parent_dir=root, pi_bin=root / "pi", keep=True) as wt:
        _forge_dead_marker(wt.path)
        e = _entry(find_scratch_dirs(root), wt.path)
        assert e["removable"] is False
        assert "lock" in e["reason"]
        assert main(["--scratch-root", str(root), "--repo-root", str(source_repo), "--clean", "--yes"]) == 0
        assert "Nothing to clean" in capsys.readouterr().out
        assert wt.path.is_dir()
    # Lock released on exit: the same forged marker now reads as an orphan.
    assert _entry(find_scratch_dirs(root), wt.path)["removable"] is True


def test_leftover_unheld_lock_file_does_not_block_removal_and_is_cleaned(source_repo, root):
    wt = _make_kept(source_repo, root)
    lock_path = scratch_lock_path(wt.path)
    lock_path.touch()
    _forge_dead_marker(wt.path)
    assert main(["--scratch-root", str(root), "--repo-root", str(source_repo), "--clean", "--yes"]) == 0
    assert not any(os.path.lexists(p) for p in _siblings(wt.path))


def test_cli_clean_without_yes_prompts_and_respects_no(source_repo, root, monkeypatch, capsys):
    wt = _make_kept(source_repo, root)
    _forge_dead_marker(wt.path)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert main(["--scratch-root", str(root), "--repo-root", str(source_repo), "--clean"]) == 2
    assert "Aborted" in capsys.readouterr().out
    assert all(os.path.lexists(p) for p in _siblings(wt.path)[:3])


def test_cli_clean_with_yes_removes_tree_git_dir_marker_and_lock(source_repo, root, capsys):
    wt = _make_kept(source_repo, root)
    _forge_dead_marker(wt.path)
    assert main(["--scratch-root", str(root), "--repo-root", str(source_repo), "--clean", "--yes"]) == 0
    assert "Removed 1 scratch" in capsys.readouterr().out
    assert not any(os.path.lexists(p) for p in _siblings(wt.path))


def test_cli_clean_older_than_hours_filters_out_recent_orphans(source_repo, root):
    wt = _make_kept(source_repo, root)
    _forge_dead_marker(wt.path)
    assert main(["--scratch-root", str(root), "--repo-root", str(source_repo),
                 "--clean", "--yes", "--older-than-hours", "999"]) == 0
    assert wt.path.exists()
    _edit_marker(wt.path, created_at=time.time() - 1000 * 3600)
    assert main(["--scratch-root", str(root), "--repo-root", str(source_repo),
                 "--clean", "--yes", "--older-than-hours", "999"]) == 0
    assert not wt.path.exists()


def test_teardown_orphan_that_kept_its_marker_is_later_cleaned_by_the_gc(source_repo, root, monkeypatch):
    """scratch_worktree() keeps the marker when it cannot remove the tree;
    that is what lets the GC find and remove it afterwards."""
    def _cannot_remove(path):
        raise PermissionError(13, "simulated", str(path))

    with scratch_worktree(source_repo, parent_dir=root, pi_bin=root / "pi") as wt:
        hid = wt.path / "skills-hid"
        hid.mkdir()
        (hid / "x").write_text("x\n")
        hid.chmod(0)
        monkeypatch.setattr(sw, "_force_rmtree", _cannot_remove)
    monkeypatch.undo()
    assert wt.path.exists() and wt.marker_path.exists()
    # The owner (this process) is alive; once it is gone the GC removes it,
    # chmod 000 directory included.
    _forge_dead_marker(wt.path)
    assert _entry(find_scratch_dirs(root), wt.path)["removable"] is True
    assert main(["--scratch-root", str(root), "--repo-root", str(source_repo), "--clean", "--yes"]) == 0
    assert not any(os.path.lexists(p) for p in _siblings(wt.path))


def test_gc_continues_past_an_entry_it_cannot_remove_and_keeps_its_marker(source_repo, root, monkeypatch, capsys):
    stuck = _make_kept(source_repo, root)
    fine = _make_kept(source_repo, root)
    _forge_dead_marker(stuck.path)
    _forge_dead_marker(fine.path)
    real = sw._force_rmtree

    def _flaky(path):
        if os.path.basename(str(path)).startswith(stuck.path.name):
            raise PermissionError(13, "simulated", str(path))
        return real(path)

    monkeypatch.setattr(gc, "_force_rmtree", _flaky)
    code = main(["--scratch-root", str(root), "--repo-root", str(source_repo), "--clean", "--yes"])
    assert code == 1
    out = capsys.readouterr()
    assert "could not remove" in (out.out + out.err)
    assert not any(os.path.lexists(p) for p in _siblings(fine.path))
    assert stuck.path.exists() and stuck.marker_path.exists()
    monkeypatch.undo()
    assert main(["--scratch-root", str(root), "--repo-root", str(source_repo), "--clean", "--yes"]) == 0
    assert not any(os.path.lexists(p) for p in _siblings(stuck.path))


# --------------------------------------------------------------------------
# Legacy `git worktree` layout
# --------------------------------------------------------------------------


def _make_legacy_scratch(source_repo, root, *, pid=999999999):
    """A worktree as the old scratch_worktree() made it: detached, named
    gepa-scratch-<pid>-<hex8>, with the marker INSIDE the tree."""
    path = root / f"gepa-scratch-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    _git(source_repo, "worktree", "add", "-q", "--detach", str(path))
    (path / LEGACY_SCRATCH_MARKER_NAME).write_text(json.dumps({
        "little_coder_self_improve_scratch": True, "pid": pid, "created_at": time.time(),
    }))
    return path


def _make_branch_worktree(source_repo, parent, name):
    path = parent / f"branch-worktree-{name}"
    _git(source_repo, "worktree", "add", "-q", "-b", name, str(path))
    return path


def _registered(source_repo, path):
    listing = _git(source_repo, "worktree", "list", "--porcelain").stdout
    return str(path) in listing or os.path.realpath(path) in listing


def test_legacy_branch_worktree_is_never_touched(source_repo, root):
    branch_wt = _make_branch_worktree(source_repo, root, "feature-x")
    (branch_wt / LEGACY_SCRATCH_MARKER_NAME).write_text(json.dumps({
        "little_coder_self_improve_scratch": True, "pid": 999999999,
    }))
    e = _entry(find_legacy_worktrees(source_repo, scratch_root=root), branch_wt)
    assert e["removable"] is False
    assert "branch" in e["reason"]
    assert main(["--repo-root", str(source_repo), "--scratch-root", str(root), "--clean", "--yes"]) == 0
    assert _registered(source_repo, branch_wt)


def test_legacy_dead_marker_worktree_is_removable_and_cleaned(source_repo, root, capsys):
    path = _make_legacy_scratch(source_repo, root)
    e = _entry(find_legacy_worktrees(source_repo, scratch_root=root), path)
    assert e["removable"] is True
    assert e["kind"] == "legacy-worktree"
    assert find_scratch_worktrees is find_legacy_worktrees
    assert main(["--repo-root", str(source_repo), "--scratch-root", str(root), "--clean", "--yes"]) == 0
    assert not path.exists()
    assert not _registered(source_repo, path)


def test_legacy_prunable_gone_worktree_removable_when_scoped_and_name_matches(source_repo, root):
    path = _make_legacy_scratch(source_repo, root)
    shutil.rmtree(path)
    e = _entry(find_legacy_worktrees(source_repo, scratch_root=root), path)
    assert e["removable"] is True
    assert "prunable" in e["reason"]


def test_legacy_prunable_gone_worktree_not_removable_without_scratch_root(source_repo, root):
    path = _make_legacy_scratch(source_repo, root)
    shutil.rmtree(path)
    e = _entry(find_legacy_worktrees(source_repo, scratch_root=None), path)
    assert e["removable"] is False
    _git(source_repo, "worktree", "prune")


def test_legacy_prunable_gone_worktree_with_mismatched_name_not_removable(source_repo, root):
    foreign = root / "not-ours-at-all"
    _git(source_repo, "worktree", "add", "-q", "--detach", str(foreign))
    shutil.rmtree(foreign)
    e = _entry(find_legacy_worktrees(source_repo, scratch_root=root), foreign)
    assert e["removable"] is False
    assert "no marker" in e["reason"]
    _git(source_repo, "worktree", "prune")


def test_legacy_clean_does_not_prune_entries_outside_its_scratch_root(source_repo, root, tmp_path):
    other_root = tmp_path / "other-legacy-root"
    other_root.mkdir()
    untouched = _make_legacy_scratch(source_repo, other_root)
    target = _make_legacy_scratch(source_repo, root)
    shutil.rmtree(untouched)
    shutil.rmtree(target)
    try:
        assert main(["--repo-root", str(source_repo), "--scratch-root", str(root), "--clean", "--yes"]) == 0
        assert not _registered(source_repo, target)
        assert _registered(source_repo, untouched)
    finally:
        _git(source_repo, "worktree", "prune")


def test_legacy_worktree_listed_once_not_duplicated_by_dir_scan(source_repo, root, capsys):
    path = _make_legacy_scratch(source_repo, root)
    assert all(os.path.realpath(e["path"]) != os.path.realpath(path) for e in find_scratch_dirs(root))
    main(["--repo-root", str(source_repo), "--scratch-root", str(root), "--list"])
    out = capsys.readouterr().out
    assert sum(1 for line in out.splitlines() if path.name in line) == 1


def test_legacy_pass_skipped_when_repo_root_is_not_a_git_repo(source_repo, root, tmp_path, capsys):
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()
    wt = _make_kept(source_repo, root)
    _forge_dead_marker(wt.path)
    assert main(["--repo-root", str(not_a_repo), "--scratch-root", str(root), "--clean", "--yes"]) == 0
    assert not wt.path.exists()


def test_cli_list_prints_every_registered_worktree_and_scratch_dir(source_repo, root, capsys):
    branch_wt = _make_branch_worktree(source_repo, root, "feature-list")
    wt = _make_kept(source_repo, root)
    main(["--repo-root", str(source_repo), "--scratch-root", str(root), "--list"])
    out = capsys.readouterr().out
    assert str(source_repo) in out or os.path.realpath(source_repo) in out
    assert branch_wt.name in out
    assert wt.path.name in out
    assert "private-repo" in out and "legacy-worktree" in out
