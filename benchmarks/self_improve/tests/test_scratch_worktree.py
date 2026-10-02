"""scratch_worktree.py: all tests run against a THROWAWAY `git init` repo in
tmp_path -- NEVER the real checkout. conftest.py's _forbid_scratch_of_real_repo
fails any test that points scratch_worktree() at the real repo, and the
session-scoped _no_stray_scratch_artifacts fixture is the backstop for leaks."""
import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time

import pytest

import benchmarks.self_improve.scratch_worktree as sw
from benchmarks.self_improve.scratch_worktree import (
    LEGACY_SCRATCH_MARKER_NAME,
    ScratchWorktreeCorrupted,
    ScratchWorktreeError,
    prune_stale,
    resolve_pi_bin,
    scratch_git_dir,
    scratch_lock_path,
    scratch_marker_path,
    scratch_worktree,
)


def _git(cwd, *args, check=True):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=check)


def _init_repo(repo):
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "AGENTS.md").write_text("# little-coder\n\nBody text.\n")
    (repo / "skills").mkdir()
    (repo / "skills" / "bash.md").write_text("---\nname: bash\n---\nBash guidance.\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo


@pytest.fixture
def source_repo(tmp_path):
    return _init_repo(tmp_path / "source")


@pytest.fixture
def linked_source(tmp_path):
    """A linked worktree of a throwaway repo, like the real source checkout
    (whose .git is a gitfile pointing into another repo's .git)."""
    main = _init_repo(tmp_path / "main-source")
    linked = tmp_path / "linked-source"
    _git(main, "worktree", "add", "-q", "--detach", str(linked))
    (linked / "AGENTS.md").write_text("linked content\n")
    _git(linked, "commit", "-q", "-am", "commit on the linked worktree's detached HEAD")
    return linked


def _current_branch(repo_root):
    return _git(repo_root, "symbolic-ref", "-q", "--short", "HEAD", check=False)


def _worktree_list(repo_root):
    return _git(repo_root, "worktree", "list", "--porcelain").stdout


def _purge(wt):
    """Cleanup after keep=True: there is no registration to undo any more,
    only the four sibling paths."""
    shutil.rmtree(wt.path, ignore_errors=True)
    shutil.rmtree(wt.git_dir, ignore_errors=True)
    for p in (wt.marker_path, scratch_lock_path(wt.path)):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def _hash_tree(root):
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            p = os.path.join(dirpath, name)
            if os.path.islink(p):
                out[p] = "link:" + os.readlink(p)
            else:
                with open(p, "rb") as fh:
                    out[p] = hashlib.sha256(fh.read()).hexdigest()
    return out


def _git_common_dir(repo):
    return _git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir").stdout.strip()


def _write_exec(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)


# --------------------------------------------------------------------------
# resolve_pi_bin
# --------------------------------------------------------------------------


def test_resolve_pi_bin_prefers_explicit_argument(source_repo, tmp_path):
    fake = tmp_path / "explicit_pi.py"
    fake.write_text("#!/usr/bin/env python3\n")
    assert resolve_pi_bin(source_repo, explicit=fake) == fake.resolve()


def test_resolve_pi_bin_uses_env_override_when_set(source_repo, tmp_path, monkeypatch):
    fake = tmp_path / "fake_pi.py"
    fake.write_text("#!/usr/bin/env python3\n")
    monkeypatch.setenv("LITTLE_CODER_PI_BIN_OVERRIDE", str(fake))
    assert resolve_pi_bin(source_repo) == fake.resolve()


def test_resolve_pi_bin_falls_back_to_node_modules(source_repo, monkeypatch):
    monkeypatch.delenv("LITTLE_CODER_PI_BIN_OVERRIDE", raising=False)
    pi_path = source_repo / "node_modules" / ".bin" / "pi"
    pi_path.parent.mkdir(parents=True)
    pi_path.write_text("#!/usr/bin/env node\n")
    assert resolve_pi_bin(source_repo) == pi_path.resolve()


def test_resolve_pi_bin_raises_clear_error_when_nothing_found(source_repo, monkeypatch):
    monkeypatch.delenv("LITTLE_CODER_PI_BIN_OVERRIDE", raising=False)
    with pytest.raises(FileNotFoundError, match="pi CLI not found"):
        resolve_pi_bin(source_repo)


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


def test_scratch_tree_is_detached_and_creates_no_branch(source_repo, tmp_path):
    branches_before = _git(source_repo, "branch", "--list").stdout
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        assert wt.git(["symbolic-ref", "-q", "HEAD"], check=False).returncode != 0
    assert _git(source_repo, "branch", "--list").stdout == branches_before


def test_scratch_tree_contains_the_committed_files(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        assert (wt.path / "AGENTS.md").read_text() == "# little-coder\n\nBody text.\n"
        assert (wt.path / "skills" / "bash.md").exists()


def test_two_managers_get_distinct_paths_and_coexist(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt1:
        with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt2:
            assert wt1.path != wt2.path
            assert wt1.git_dir != wt2.git_dir
            assert wt1.path.is_dir()
            assert wt2.path.is_dir()


def _siblings(path):
    return [path, scratch_git_dir(path), scratch_marker_path(path), scratch_lock_path(path)]


def test_everything_removed_on_normal_exit(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        scratch_path = wt.path
        assert all(os.path.lexists(p) for p in _siblings(scratch_path))
    assert not any(os.path.lexists(p) for p in _siblings(scratch_path))
    assert str(scratch_path) not in _worktree_list(source_repo)


def test_everything_removed_when_body_raises(source_repo, tmp_path):
    holder = {}
    with pytest.raises(RuntimeError, match="boom"):
        with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
            holder["path"] = wt.path
            raise RuntimeError("boom")
    assert not any(os.path.lexists(p) for p in _siblings(holder["path"]))


def test_everything_removed_when_body_raises_keyboardinterrupt(source_repo, tmp_path):
    holder = {}
    with pytest.raises(KeyboardInterrupt):
        with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
            holder["path"] = wt.path
            raise KeyboardInterrupt()
    assert not any(os.path.lexists(p) for p in _siblings(holder["path"]))


def test_keep_preserves_tree_on_normal_exit(source_repo, tmp_path, capsys):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi", keep=True) as wt:
        pass
    assert wt.path.exists() and wt.git_dir.exists() and wt.marker_path.exists()
    assert "PRESERVED FOR POST-MORTEM" in capsys.readouterr().out
    _purge(wt)


def test_keep_preserves_tree_on_exception_without_masking_it(source_repo, tmp_path, capsys):
    """keep=True cleanup never masks an exception raised inside the block."""
    holder = {}

    class _Boom(Exception):
        pass

    with pytest.raises(_Boom):
        with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi", keep=True) as wt:
            holder["wt"] = wt
            raise _Boom("deliberate failure inside the with block")

    assert holder["wt"].path.exists()
    assert "PRESERVED FOR POST-MORTEM" in capsys.readouterr().out
    _purge(holder["wt"])


def test_keep_true_prints_hardened_inspect_command_and_gc_hint(source_repo, tmp_path, capsys):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi", keep=True) as wt:
        pass
    out = capsys.readouterr().out
    assert f"SCRATCH WORKTREE PRESERVED FOR POST-MORTEM: {wt.path}" in out
    inspect_line = next(line for line in out.splitlines() if line.startswith("Inspect with:"))
    assert f"--git-dir={wt.git_dir}" in inspect_line
    assert f"--work-tree={wt.path}" in inspect_line
    # The human's post-mortem must not run what the agent planted either.
    for flag in ("GIT_CONFIG_GLOBAL=/dev/null", "GIT_CONFIG_NOSYSTEM=1", f"GIT_COMMON_DIR={wt.git_dir}",
                 "core.hooksPath=/dev/null", "core.fsmonitor=false",
                 "core.attributesFile=/dev/null", "core.excludesFile=/dev/null", "--no-optional-locks",
                 "--no-pager", "status --porcelain --ignore-submodules=all"):
        assert flag in inspect_line
    assert f"--scratch-root {tmp_path.resolve()}" in out
    _purge(wt)


@pytest.mark.parametrize("keep", [False, True])
def test_lock_marker_and_git_dir_are_siblings_outside_the_tree(source_repo, tmp_path, keep):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi", keep=keep) as wt:
        lock_path = scratch_lock_path(wt.path)
        for sibling in (lock_path, wt.marker_path, wt.git_dir):
            assert sibling.parent == wt.path.parent
        assert wt.marker_path == scratch_marker_path(wt.path)
        assert wt.git_dir == scratch_git_dir(wt.path)
        fd = os.open(lock_path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
    assert not lock_path.exists()
    if keep:
        assert wt.path.exists() and wt.git_dir.exists() and wt.marker_path.exists()
        _purge(wt)
    else:
        assert not any(os.path.lexists(p) for p in _siblings(wt.path))


@pytest.mark.parametrize("suffix", ["", ".marker.json", ".git", ".lock"])
def test_scratch_path_refuses_to_reuse_an_existing_sibling(source_repo, tmp_path, monkeypatch, suffix):
    import uuid
    fixed_uuid = uuid.UUID(int=0)
    monkeypatch.setattr("benchmarks.self_improve.scratch_worktree.uuid.uuid4", lambda: fixed_uuid)
    base = tmp_path / f"gepa-scratch-{os.getpid()}-{fixed_uuid.hex[:8]}"
    planted = base.with_name(base.name + suffix)
    if suffix in ("", ".git"):
        planted.mkdir()
    else:
        planted.write_text("not ours\n")
    with pytest.raises(ScratchWorktreeError, match="already exists"):
        with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi"):
            pass
    assert planted.exists()  # never destroyed: not ours


# --------------------------------------------------------------------------
# Marker
# --------------------------------------------------------------------------


def test_mark_spawn_pending_records_a_recent_timestamp_without_clobbering_other_marker_fields(
    source_repo, tmp_path,
):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        before = json.loads(wt.marker_path.read_text())
        wt.mark_spawn_pending()
        after = json.loads(wt.marker_path.read_text())
        assert after["pid"] == before["pid"]
        assert isinstance(after["spawn_pending_at"], (int, float))
        assert abs(after["spawn_pending_at"] - time.time()) < 5


def test_set_active_pid_clears_spawn_pending_at_on_both_the_pid_and_none_paths(source_repo, tmp_path):
    """set_active_pid() must clear spawn_pending_at on every path, so a
    lingering one can only mean a crash mid-spawn."""
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        marker_path = wt.marker_path
        wt.mark_spawn_pending()
        assert "spawn_pending_at" in json.loads(marker_path.read_text())

        wt.set_active_pid(12345)
        after_pid = json.loads(marker_path.read_text())
        assert "spawn_pending_at" not in after_pid
        assert after_pid["active_pid"] == 12345

        wt.mark_spawn_pending()
        assert "spawn_pending_at" in json.loads(marker_path.read_text())

        wt.set_active_pid(None)
        after_none = json.loads(marker_path.read_text())
        assert "spawn_pending_at" not in after_none
        assert after_none["active_pid"] is None


def test_set_active_pid_does_not_follow_a_symlink_planted_at_the_marker_path(source_repo, tmp_path):
    """The marker sits in a shared parent dir (by default the system temp
    dir), so a symlink can be planted at its predictable name. The
    orchestrator's bookkeeping write must swap the entry, never write
    through it."""
    outside_target = tmp_path / "outside_target.txt"
    outside_target.write_text("do not overwrite me\n")

    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        wt.marker_path.unlink()
        wt.marker_path.symlink_to(outside_target)

        wt.set_active_pid(12345)

        assert outside_target.read_text() == "do not overwrite me\n"
        assert not wt.marker_path.is_symlink()
        assert json.loads(wt.marker_path.read_text())["active_pid"] == 12345


def test_marker_is_outside_the_tree_and_nothing_in_the_tree_names_the_orchestrator(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        wt.mark_spawn_pending()
        wt.set_active_pid(4242)
        assert not (wt.path / LEGACY_SCRATCH_MARKER_NAME).exists()
        marker = json.loads(wt.marker_path.read_text())
        assert marker["pid"] == os.getpid()
        assert marker["little_coder_self_improve_scratch"] is True
        assert marker["layout"] == "private-repo-v1"
        for p in wt.path.rglob("*"):
            if p.is_file() and not p.is_symlink():
                text = p.read_bytes().decode("utf-8", "replace")
                assert str(source_repo) not in text, p
                assert f'"pid": {os.getpid()}' not in text, p


def test_reset_leaves_the_external_marker_intact(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        before = wt.marker_path.read_bytes()
        wt.reset()
        assert wt.marker_path.read_bytes() == before


# --------------------------------------------------------------------------
# Private repo layout
# --------------------------------------------------------------------------


def test_scratch_tree_has_no_git_dir_and_git_admin_is_a_sibling(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        assert not os.path.lexists(wt.path / ".git")
        assert wt.git_dir == scratch_git_dir(wt.path)
        assert wt.git_dir.is_dir() and not wt.git_dir.is_symlink()
        assert stat.S_IMODE(os.lstat(wt.git_dir).st_mode) == 0o700
        assert stat.S_IMODE(os.lstat(wt.path).st_mode) == 0o700
        assert not (wt.git_dir / "hooks").exists()
        assert not (wt.git_dir / "info").exists()


def test_scratch_repo_has_no_remote_and_no_history_beyond_base(source_repo, tmp_path):
    (source_repo / "AGENTS.md").write_text("second\n")
    _git(source_repo, "commit", "-q", "-am", "second")
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        assert wt.git(["remote"]).stdout == ""
        assert wt.git(["rev-list", "--count", "HEAD"]).stdout.strip() == "1"
        assert str(source_repo) not in (wt.git_dir / "config").read_text()
        # FETCH_HEAD would name the source path; it is removed after checkout.
        assert not (wt.git_dir / "FETCH_HEAD").exists()


def test_base_commit_reachable_from_no_ref_is_fetchable(source_repo, tmp_path):
    branch = _current_branch(source_repo).stdout.strip()
    _git(source_repo, "checkout", "-q", "--detach")
    (source_repo / "AGENTS.md").write_text("only on a dangling commit\n")
    _git(source_repo, "commit", "-q", "-am", "dangling")
    dangling = _git(source_repo, "rev-parse", "HEAD").stdout.strip()
    _git(source_repo, "checkout", "-q", branch)
    with scratch_worktree(source_repo, commit=dangling, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        assert wt.base_commit == dangling
        assert (wt.path / "AGENTS.md").read_text() == "only on a dangling commit\n"


def test_source_that_is_a_linked_worktree_works(linked_source, tmp_path):
    with scratch_worktree(linked_source, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        assert (wt.path / "AGENTS.md").read_text() == "linked content\n"
        wt.reset()
        assert wt.git(["status", "--porcelain"]).stdout == ""


@pytest.mark.parametrize("variant", ["plain", "linked"])
def test_source_repo_git_admin_is_byte_identical_after_a_full_cycle(
    source_repo, linked_source, tmp_path, variant,
):
    source = source_repo if variant == "plain" else linked_source
    common = _git_common_dir(source)
    before = _hash_tree(common)
    worktrees_before = _worktree_list(source)

    with scratch_worktree(source, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        (wt.path / "AGENTS.md").write_text("mutated\n")
        (wt.path / "junk.txt").write_text("junk\n")
        wt.reset()
        target = wt.path / "skills" / "bash.md"
        target.write_text("candidate\n")
        wt.assert_only_expected_dirty([target])
        wt.reset()

    assert _hash_tree(common) == before
    assert _worktree_list(source) == worktrees_before


# --------------------------------------------------------------------------
# reset() / assert_only_expected_dirty()
# --------------------------------------------------------------------------


def test_reset_restores_tree_after_arbitrary_mutation(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        (wt.path / "AGENTS.md").write_text("mutated\n")
        (wt.path / "skills" / "bash.md").unlink()
        (wt.path / "untracked_junk.txt").write_text("junk\n")
        wt.reset()
        assert wt.git(["status", "--porcelain"]).stdout == ""
        assert (wt.path / "AGENTS.md").read_text() == "# little-coder\n\nBody text.\n"
        assert (wt.path / "skills" / "bash.md").exists()
        assert not (wt.path / "untracked_junk.txt").exists()


def test_reset_preserves_node_modules(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        (wt.path / ".gitignore").write_text("node_modules/\n")
        (wt.path / "node_modules").mkdir()
        (wt.path / "node_modules" / "marker.txt").write_text("still here\n")
        wt.reset()
        assert (wt.path / "node_modules" / "marker.txt").exists()


def test_repeated_reset_cycles_keep_the_git_dir_check_passing(source_repo, tmp_path):
    """The G/config byte check is fail-closed, so anything git itself writes
    there during normal operation would kill a run at candidate 2."""
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        target = wt.path / "skills" / "bash.md"
        for i in range(4):
            wt.reset()
            target.write_text(f"candidate {i}\n")
            wt.assert_only_expected_dirty([target])
            assert not (wt.git_dir / "hooks").exists()
            assert not (wt.git_dir / "info").exists()
        wt.reset()
        assert wt.git(["status", "--porcelain"]).stdout == ""


def test_assert_only_expected_dirty_passes_when_only_expected_files_changed(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        target = wt.path / "skills" / "bash.md"
        target.write_text("---\nname: bash\n---\nRevised.\n")
        wt.assert_only_expected_dirty([target])


def test_assert_only_expected_dirty_raises_on_an_unexpected_modification(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        (wt.path / "skills" / "bash.md").write_text("expected change\n")
        (wt.path / "AGENTS.md").write_text("UNEXPECTED change\n")
        with pytest.raises(ScratchWorktreeCorrupted, match="AGENTS.md"):
            wt.assert_only_expected_dirty([wt.path / "skills" / "bash.md"])


def test_base_commit_is_pinned_even_if_source_head_moves(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        pinned = wt.base_commit
        (source_repo / "AGENTS.md").write_text("new content\n")
        _git(source_repo, "add", "-A")
        _git(source_repo, "commit", "-q", "-m", "advance")
        wt.reset()
        assert wt.base_commit == pinned
        assert (wt.path / "AGENTS.md").read_text() == "# little-coder\n\nBody text.\n"


# --------------------------------------------------------------------------
# What the agent can plant must not run, and must not reach the source
# --------------------------------------------------------------------------


def _config_hook_block(sentinel):
    return (
        '[hook "x"]\n'
        f"\tcommand = touch {sentinel}\n"
        "\tevent = reference-transaction\n"
        "\tevent = post-index-change\n"
        "\tevent = post-checkout\n"
    )


def test_hook_planted_via_the_scratch_tree_never_reaches_the_source(source_repo, tmp_path):
    sentinel = tmp_path / "hook-fired"
    hook = f"#!/bin/sh\ntouch {sentinel}\n"
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        _write_exec(wt.path / ".git" / "hooks" / "post-checkout", hook)
        _write_exec(wt.path / ".git" / "hooks" / "reference-transaction", hook)
        (wt.path / "AGENTS.md").write_text("mutated\n")
        wt.reset()
        assert not os.path.lexists(wt.path / ".git")
    _git(source_repo, "checkout", "-q", "--detach", "HEAD")
    _git(source_repo, "commit", "-q", "--allow-empty", "-m", "x")
    assert not sentinel.exists()
    hooks_dir = source_repo / ".git" / "hooks"
    assert [p.name for p in hooks_dir.iterdir() if not p.name.endswith(".sample")] == []


@pytest.mark.parametrize("plant", ["hooks-dir", "info-dir", "config-hook", "config-hookspath", "config-fsmonitor"])
def test_tampering_with_the_git_dir_makes_reset_raise_and_nothing_fires(source_repo, tmp_path, plant):
    sentinel = tmp_path / "fired"
    script = tmp_path / "evil.sh"
    _write_exec(script, f"#!/bin/sh\ntouch {sentinel}\n")
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        g = wt.git_dir
        if plant == "hooks-dir":
            _write_exec(g / "hooks" / "post-checkout", f"#!/bin/sh\ntouch {sentinel}\n")
            _write_exec(g / "hooks" / "reference-transaction", f"#!/bin/sh\ntouch {sentinel}\n")
        elif plant == "info-dir":
            (g / "info").mkdir()
            (g / "info" / "attributes").write_text("* filter=x\n")
        elif plant == "config-hook":
            with open(g / "config", "a") as fh:
                fh.write(_config_hook_block(sentinel))
        elif plant == "config-hookspath":
            _write_exec(tmp_path / "evil-hooks" / "post-checkout", f"#!/bin/sh\ntouch {sentinel}\n")
            with open(g / "config", "a") as fh:
                fh.write(f"[core]\n\thooksPath = {tmp_path / 'evil-hooks'}\n")
        elif plant == "config-fsmonitor":
            with open(g / "config", "a") as fh:
                fh.write(f"[core]\n\tfsmonitor = {script}\n")
        (wt.path / "AGENTS.md").write_text("mutated\n")
        with pytest.raises(ScratchWorktreeCorrupted):
            wt.reset()
        with pytest.raises(ScratchWorktreeCorrupted):
            wt.assert_only_expected_dirty([])
    assert not sentinel.exists()


def test_git_config_replaced_by_a_symlink_to_identical_bytes_is_rejected(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        copy = tmp_path / "config-copy"
        copy.write_bytes((wt.git_dir / "config").read_bytes())
        (wt.git_dir / "config").unlink()
        (wt.git_dir / "config").symlink_to(copy)
        with pytest.raises(ScratchWorktreeCorrupted):
            wt.reset()


def test_config_hooks_in_global_config_or_env_do_not_fire_during_reset_or_status(
    source_repo, tmp_path, monkeypatch,
):
    sentinel = tmp_path / "fired"
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        evil_global = tmp_path / "evil-gitconfig"
        evil_global.write_text(_config_hook_block(sentinel))
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(evil_global))
        monkeypatch.setenv("GIT_CONFIG_COUNT", "2")
        monkeypatch.setenv("GIT_CONFIG_KEY_0", "hook.y.command")
        monkeypatch.setenv("GIT_CONFIG_VALUE_0", f"touch {sentinel}")
        monkeypatch.setenv("GIT_CONFIG_KEY_1", "hook.y.event")
        monkeypatch.setenv("GIT_CONFIG_VALUE_1", "reference-transaction")
        monkeypatch.setenv("GIT_CONFIG_PARAMETERS", f"'hook.z.command'='touch {sentinel}' 'hook.z.event'='post-index-change'")
        (wt.path / "AGENTS.md").write_text("mutated\n")
        wt.reset()
        target = wt.path / "skills" / "bash.md"
        target.write_text("candidate\n")
        wt.assert_only_expected_dirty([target])
        wt.reset()
    assert not sentinel.exists()


def test_in_tree_gitattributes_filter_from_global_config_does_not_run_during_reset(
    source_repo, tmp_path, monkeypatch,
):
    """`* filter=x` written into the tree plus a filter.x driver in the
    user's global config (stands in for the real ~/.gitconfig's filter.lfs):
    neither the tree's attributes nor the global driver may reach reset()."""
    sentinel = tmp_path / "filter-ran"
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        driver = tmp_path / "filter-driver.sh"
        _write_exec(driver, f"#!/bin/sh\ntouch {sentinel}\nexec cat\n")
        global_cfg = tmp_path / "filter-gitconfig"
        # A script path, not an inline `sh -c '...; cat'`: an unquoted `;`
        # starts a comment in git config and would silently break the driver.
        global_cfg.write_text(f'[filter "x"]\n\tsmudge = {driver}\n\tclean = {driver}\n')
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_cfg))
        (wt.path / ".gitattributes").write_text("* filter=x\n")
        (wt.path / "AGENTS.md").write_text("mutated, and a different size\n")
        (wt.path / "skills" / "bash.md").write_text("mutated too, also a different size\n")
        wt.reset()
        assert wt.git(["status", "--porcelain"]).stdout == ""
        assert not (wt.path / ".gitattributes").exists()
    assert not sentinel.exists()


def test_planted_gitfile_pointing_at_the_source_is_ignored_and_removed_by_reset(source_repo, tmp_path):
    src_git = source_repo / ".git"
    head_before = (src_git / "HEAD").read_bytes()
    index_before = (src_git / "index").read_bytes()
    branch_before = _current_branch(source_repo).stdout
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        (wt.path / ".git").write_text(f"gitdir: {src_git}\n")
        (wt.path / "AGENTS.md").write_text("mutated\n")
        wt.reset()
        assert not os.path.lexists(wt.path / ".git")
    assert (src_git / "HEAD").read_bytes() == head_before
    assert (src_git / "index").read_bytes() == index_before
    assert _current_branch(source_repo).stdout == branch_before


def test_planted_replace_ref_does_not_change_what_reset_checks_out(source_repo, tmp_path):
    """refs/replace/<base> in the sibling git dir would otherwise make every
    reset() check out the agent's substitute commit instead of the base."""
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        raw = ["git", "--git-dir", str(wt.git_dir)]
        run = lambda *a, **kw: subprocess.run([*raw, *a], capture_output=True, text=True, check=True, **kw)
        blob = run("hash-object", "-w", "--stdin", input="TAMPERED\n").stdout.strip()
        skills_tree = run("rev-parse", f"{wt.base_commit}:skills").stdout.strip()
        tree = run("mktree", input=f"100644 blob {blob}\tAGENTS.md\n040000 tree {skills_tree}\tskills\n").stdout.strip()
        fake = run("commit-tree", tree, "-m", "substitute",
                   env={**os.environ, "GIT_AUTHOR_NAME": "x", "GIT_AUTHOR_EMAIL": "x@x",
                        "GIT_COMMITTER_NAME": "x", "GIT_COMMITTER_EMAIL": "x@x"}).stdout.strip()
        run("update-ref", f"refs/replace/{wt.base_commit}", fake)
        (wt.path / "AGENTS.md").write_text("mutated\n")
        wt.reset()
        assert (wt.path / "AGENTS.md").read_text() == "# little-coder\n\nBody text.\n"


def test_nested_dot_git_entries_anywhere_in_the_tree_are_removed_by_reset(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        (wt.path / "skills" / ".git").mkdir()
        (wt.path / "skills" / ".git" / "SKILL.md").write_text("hidden store\n")
        (wt.path / "skills" / "sub").mkdir()
        (wt.path / "skills" / "sub" / ".GIT").write_text("gitdir: /elsewhere\n")
        (wt.path / "deep" / "er").mkdir(parents=True)
        (wt.path / "deep" / "er" / ".Git").symlink_to(tmp_path)
        wt.reset()
        leftovers = [
            os.path.join(d, n)
            for d, dirs, files in os.walk(wt.path, followlinks=False)
            for n in dirs + files if n.lower() == ".git"
        ]
        assert leftovers == []
        assert tmp_path.exists()  # the symlink was removed, not followed
        assert (wt.path / "skills" / "bash.md").exists()


def test_attr_source_fallback_cleans_before_checkout_on_older_git(source_repo, tmp_path, monkeypatch):
    """git < 2.40 has no --attr-source: the fallback removes untracked files
    (an agent-written .gitattributes) BEFORE checkout reads them."""
    monkeypatch.setattr(sw, "_git_supports_attr_source", lambda: False)
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        calls = []
        real_run = subprocess.run

        def _spy(cmd, *a, **kw):
            if cmd and cmd[0] == "git":
                calls.append(list(cmd))
            return real_run(cmd, *a, **kw)

        monkeypatch.setattr(sw.subprocess, "run", _spy)
        wt.reset()
        monkeypatch.setattr(sw.subprocess, "run", real_run)
    subcommands = [next(a for a in c[1:] if a in ("clean", "checkout", "reset")) for c in calls
                   if any(a in ("clean", "checkout", "reset") for a in c)]
    assert subcommands[0] == "clean"
    assert subcommands.index("clean") < subcommands.index("checkout")
    assert not any(a.startswith("--attr-source") for c in calls for a in c)


def test_attr_source_is_passed_on_git_that_supports_it(source_repo, tmp_path, monkeypatch):
    if not sw._git_supports_attr_source():
        pytest.skip("this git has no --attr-source")
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        calls = []
        real_run = subprocess.run

        def _spy(cmd, *a, **kw):
            if cmd and cmd[0] == "git":
                calls.append(list(cmd))
            return real_run(cmd, *a, **kw)

        monkeypatch.setattr(sw.subprocess, "run", _spy)
        wt.reset()
        wt.assert_only_expected_dirty([])
        monkeypatch.setattr(sw.subprocess, "run", real_run)
    assert calls
    for c in calls:
        assert f"--attr-source={wt.base_commit}" in c
        assert "core.attributesFile=/dev/null" in c
    status_calls = [c for c in calls if "status" in c]
    assert status_calls and all("--no-optional-locks" in c for c in status_calls)
    assert all("--ignore-submodules=all" in c for c in status_calls)
    assert all("core.excludesFile=/dev/null" in c for c in calls)


# --------------------------------------------------------------------------
# Environment of the orchestrator's own git calls
# --------------------------------------------------------------------------


def test_git_env_is_scrubbed_and_config_is_pinned(monkeypatch):
    monkeypatch.setenv("REFLECTION_LM_API_KEY", "sk-sentinel")
    monkeypatch.setenv("OTHER_SECRET", "x")
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT",
                 "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0", "GIT_ATTR_SOURCE", "GIT_SHALLOW_FILE",
                 "GIT_GRAFT_FILE", "GIT_REPLACE_REF_BASE", "GIT_OBJECT_DIRECTORY", "GIT_COMMON_DIR"):
        monkeypatch.setenv(name, "planted")
    env = sw._git_env(frozenset({"OTHER_SECRET"}))
    assert "REFLECTION_LM_API_KEY" not in env
    assert "SELF_IMPROVE_DOTENV" not in env
    assert "OTHER_SECRET" not in env
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT",
                 "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0", "GIT_ATTR_SOURCE", "GIT_SHALLOW_FILE",
                 "GIT_GRAFT_FILE", "GIT_REPLACE_REF_BASE", "GIT_OBJECT_DIRECTORY", "GIT_COMMON_DIR"):
        assert name not in env, name
    assert env["GIT_CONFIG_GLOBAL"] == os.devnull
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert "LITTLE_CODER_PI_BIN_OVERRIDE" in os.environ
    assert env["PATH"] == os.environ["PATH"]


def test_every_git_call_including_creation_runs_without_secrets(source_repo, tmp_path, monkeypatch):
    monkeypatch.setenv("REFLECTION_LM_API_KEY", "sk-sentinel")
    monkeypatch.setenv("OTHER_SECRET", "other-sentinel")
    calls = []
    real_run = subprocess.run

    def _spy(cmd, *a, **kw):
        # --git-common-dir is conftest's own real-repo guard, not ours.
        if cmd and cmd[0] == "git" and "--git-common-dir" not in cmd:
            calls.append((list(cmd), kw.get("env")))
        return real_run(cmd, *a, **kw)

    sw._git_version.cache_clear()
    sw._git_local_env_vars.cache_clear()
    monkeypatch.setattr(sw.subprocess, "run", _spy)
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi",
                          orchestrator_only_env={"OTHER_SECRET"}) as wt:
        (wt.path / "AGENTS.md").write_text("mutated\n")
        wt.reset()
        wt.assert_only_expected_dirty([])
    assert len(calls) >= 6
    for cmd, env in calls:
        assert env is not None, cmd  # never inherits os.environ wholesale
        assert "REFLECTION_LM_API_KEY" not in env, cmd
        assert "OTHER_SECRET" not in env, cmd
        if "--git-dir" in cmd or any(a.startswith("--git-dir=") for a in cmd):
            assert env["GIT_CONFIG_GLOBAL"] == os.devnull, cmd
            assert env["GIT_CONFIG_NOSYSTEM"] == "1", cmd


# --------------------------------------------------------------------------
# Setup failure and teardown robustness
# --------------------------------------------------------------------------


def test_setup_failure_leaves_nothing_behind(source_repo, tmp_path, monkeypatch):
    real = sw._run_git

    def _failing(args, **kw):
        if "fetch" in args:
            raise ScratchWorktreeError("injected fetch failure")
        return real(args, **kw)

    monkeypatch.setattr(sw, "_run_git", _failing)
    with pytest.raises(ScratchWorktreeError, match="injected"):
        with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi"):
            pass
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith("gepa-scratch-")] == []


def test_setup_never_destroys_a_git_dir_it_did_not_create(source_repo, tmp_path, monkeypatch):
    """If something else creates <tree>.git between the existence check and
    our mkdir, that directory is not ours and must survive the rollback."""
    real_mkdir = os.mkdir

    def _racing_mkdir(path, *a, **kw):
        p = str(path)
        if p.endswith(".git") and os.path.basename(p).startswith("gepa-scratch-"):
            real_mkdir(p)
            with open(os.path.join(p, "foreign.txt"), "w") as fh:
                fh.write("not yours\n")
        return real_mkdir(path, *a, **kw)

    monkeypatch.setattr(sw.os, "mkdir", _racing_mkdir)
    with pytest.raises(FileExistsError):
        with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi"):
            pass
    monkeypatch.setattr(sw.os, "mkdir", real_mkdir)
    left = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("gepa-scratch-"))
    assert len(left) == 1 and left[0].endswith(".git")
    assert (tmp_path / left[0] / "foreign.txt").read_text() == "not yours\n"


def test_teardown_removes_chmod_000_dirs_and_immutable_files(source_repo, tmp_path):
    """An agent can chmod 000 a directory or chflags uchg a file; the
    teardown's retrying remover clears both and then drops the marker."""
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        hid = wt.path / "skills" / "hid"
        hid.mkdir()
        (hid / "x.txt").write_text("x\n")
        locked = wt.path / "skills" / "locked.txt"
        locked.write_text("locked\n")
        if hasattr(os, "chflags"):
            os.chflags(locked, stat.UF_IMMUTABLE)
        hid.chmod(0)
    assert not any(os.path.lexists(p) for p in _siblings(wt.path))


def test_teardown_that_cannot_remove_the_tree_keeps_marker_and_lock(source_repo, tmp_path, monkeypatch, capsys):
    """If the tree survives teardown, the marker (and the unheld lock file)
    must stay, or the GC would later see an unmarked dir and never touch it."""
    def _cannot_remove(*args, **kwargs):
        raise PermissionError(13, "simulated: cannot remove", str(args))

    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        (wt.path / "skills" / "hid").mkdir()
        (wt.path / "skills" / "hid").chmod(0)
        monkeypatch.setattr(sw, "_force_rmtree", _cannot_remove)
        monkeypatch.setattr(sw, "_force_rmtree_at", _cannot_remove)
    err = capsys.readouterr()
    assert "could not remove" in (err.out + err.err)
    assert wt.path.exists() and wt.git_dir.exists()
    assert wt.marker_path.exists()
    assert scratch_lock_path(wt.path).exists()
    # The lock file is left but not held.
    fd = os.open(scratch_lock_path(wt.path), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)
    monkeypatch.undo()
    (wt.path / "skills" / "hid").chmod(0o700)
    _purge(wt)


# --------------------------------------------------------------------------
# Legacy worktrees
# --------------------------------------------------------------------------


def test_prune_stale_deregisters_a_manually_deleted_legacy_worktree(source_repo, tmp_path):
    legacy = tmp_path / f"gepa-scratch-{os.getpid()}-0badf00d"
    _git(source_repo, "worktree", "add", "-q", "--detach", str(legacy))
    shutil.rmtree(legacy)
    assert str(legacy) in _worktree_list(source_repo) or str(legacy.resolve()) in _worktree_list(source_repo)
    prune_stale(source_repo)
    listing = _worktree_list(source_repo)
    assert str(legacy) not in listing and str(legacy.resolve()) not in listing


# --------------------------------------------------------------------------
# env()
# --------------------------------------------------------------------------


_ENV_BASE = {
    "REFLECTION_LM_API_KEY": "sk-sentinel",
    "SELF_IMPROVE_DOTENV": "/x/.env",
    "OMLX_API_KEY": "model-sentinel",
    "PATH": "/usr/bin",
}


def test_env_always_withholds_the_reflection_key_and_dotenv_path(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        env = wt.env(dict(_ENV_BASE))
    assert "REFLECTION_LM_API_KEY" not in env
    assert "SELF_IMPROVE_DOTENV" not in env
    assert env["OMLX_API_KEY"] == "model-sentinel"
    assert env["PATH"] == "/usr/bin"
    assert env["LITTLE_CODER_PI_BIN_OVERRIDE"] == str((tmp_path / "pi").resolve())


def test_env_withholds_the_names_passed_as_orchestrator_only(source_repo, tmp_path):
    base = {**_ENV_BASE, "OTHER_SECRET": "other-sentinel"}
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi",
                          orchestrator_only_env={"OTHER_SECRET"}) as wt:
        env = wt.env(base)
    assert "OTHER_SECRET" not in env
    assert env["OMLX_API_KEY"] == "model-sentinel"
    assert base["OTHER_SECRET"] == "other-sentinel"


def test_env_still_sets_the_pi_bin_override_when_it_is_named_orchestrator_only(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi",
                          orchestrator_only_env={"LITTLE_CODER_PI_BIN_OVERRIDE"}) as wt:
        env = wt.env(dict(_ENV_BASE))
    assert env["LITTLE_CODER_PI_BIN_OVERRIDE"] == str((tmp_path / "pi").resolve())


def test_env_from_os_environ_strips_the_key_without_touching_os_environ(source_repo, tmp_path, monkeypatch):
    monkeypatch.setenv("REFLECTION_LM_API_KEY", "sk-sentinel")
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        env = wt.env()
    assert "REFLECTION_LM_API_KEY" not in sorted(env)
    assert os.environ["REFLECTION_LM_API_KEY"] == "sk-sentinel"


def test_python_is_new_enough_for_safe_rmtree():
    assert shutil.rmtree.avoids_symlink_attacks, sys.platform


def test_conftest_guard_rejects_any_worktree_of_the_protected_repo(source_repo, linked_source, tmp_path, monkeypatch):
    """conftest's _forbid_scratch_of_real_repo compares git common dirs, so
    a sibling worktree of the protected repo is caught too. Exercised with a
    throwaway repo standing in for the real one."""
    import benchmarks.self_improve.tests.conftest as cf

    protected_common = cf._git_common_dir(linked_source)
    assert protected_common == os.path.realpath(_git_common_dir(tmp_path / "main-source"))
    monkeypatch.setattr(cf, "_REAL_GIT_COMMON_DIR", protected_common)
    for source in (linked_source, tmp_path / "main-source"):
        with pytest.raises(pytest.fail.Exception, match="real repo"):
            with scratch_worktree(source, parent_dir=tmp_path, pi_bin=tmp_path / "pi"):
                pass
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi"):
        pass  # an unrelated repo is still allowed
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith("gepa-scratch-")] == []


# --------------------------------------------------------------------------
# The scratch git dir: GIT_COMMON_DIR pin, top-level allowlist, index rebuild
# --------------------------------------------------------------------------


def _raw_git(wt, *args, **kw):
    """What the agent can run: plain git on G, no hardening."""
    return subprocess.run(["git", f"--git-dir={wt.git_dir}", f"--work-tree={wt.path}", *args],
                          capture_output=True, text=True, check=True, **kw)


def _plant_commondir(wt, tmp_path, sentinel):
    evil = tmp_path / "evil-common"
    (evil / "objects" / "info").mkdir(parents=True)
    (evil / "refs").mkdir()
    (evil / "config").write_text(
        "[core]\n\trepositoryformatversion = 0\n\tbare = false\n"
        f'[hook "evil"]\n\tevent = post-checkout\n\tevent = reference-transaction\n\tcommand = touch {sentinel}\n'
    )
    (evil / "objects" / "info" / "alternates").write_text(f"{wt.git_dir / 'objects'}\n")
    shutil.copy(wt.git_dir / "HEAD", evil / "HEAD")
    (wt.git_dir / "commondir").write_text(f"{evil}\n")


def test_planted_commondir_is_rejected_and_its_config_hook_never_fires(source_repo, tmp_path):
    sentinel = tmp_path / "fired"
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        _plant_commondir(wt, tmp_path, sentinel)
        (wt.path / "AGENTS.md").write_text("mutated\n")
        with pytest.raises(ScratchWorktreeCorrupted, match="commondir"):
            wt.reset()
        with pytest.raises(ScratchWorktreeCorrupted):
            wt.assert_only_expected_dirty([])
    assert not sentinel.exists()


def test_git_common_dir_is_pinned_even_if_the_allowlist_check_is_bypassed(source_repo, tmp_path, monkeypatch):
    sentinel = tmp_path / "fired"
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        assert wt._git_env()["GIT_COMMON_DIR"] == str(wt.git_dir)
        _plant_commondir(wt, tmp_path, sentinel)
        monkeypatch.setattr(type(wt), "_verify_git_dir", lambda self: None)
        (wt.path / "AGENTS.md").write_text("mutated\n")
        wt.reset()
        out = wt.git(["rev-parse", "--git-common-dir"]).stdout.strip()
        assert os.path.realpath(out) == os.path.realpath(wt.git_dir)
        monkeypatch.undo()
        os.unlink(wt.git_dir / "commondir")
    assert not sentinel.exists()


def test_skip_worktree_bit_does_not_survive_reset(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        target = wt.path / "skills" / "bash.md"
        _raw_git(wt, "update-index", "--skip-worktree", "skills/bash.md")
        target.write_text("POISONED\n")
        wt.reset()
        assert target.read_text() == "---\nname: bash\n---\nBash guidance.\n"
        wt.assert_only_expected_dirty([])
        assert "S " not in wt.git(["ls-files", "-v"]).stdout


@pytest.mark.parametrize("flag", ["--skip-worktree", "--assume-unchanged"])
def test_assert_only_expected_dirty_rejects_flagged_index_entries(source_repo, tmp_path, flag):
    """Set after reset() (an agent process left running), the bit hides the
    edit from status; ls-files -v still shows it."""
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        wt.reset()
        _raw_git(wt, "update-index", flag, "skills/bash.md")
        (wt.path / "skills" / "bash.md").write_text("POISONED\n")
        with pytest.raises(ScratchWorktreeCorrupted, match="skills/bash.md"):
            wt.assert_only_expected_dirty([])


def _make_other_checkout(tmp_path):
    other = _init_repo(tmp_path / "other-checkout")
    (other / ".env").write_text("SECRET=1\n")
    (other / "untracked.txt").write_text("keep me\n")
    (other / "AGENTS.md").write_text("uncommitted work\n")
    return other


def _snapshot_other(other):
    return {
        "git": sorted(p.name for p in (other / ".git").iterdir()),
        "index": (other / ".git" / "index").read_bytes(),
        "env": (other / ".env").read_text(),
        "untracked": (other / "untracked.txt").read_text(),
        "edit": (other / "AGENTS.md").read_text(),
    }


@pytest.mark.parametrize("when", ["reset", "git", "assert"])
def test_tree_swapped_for_a_symlink_to_another_checkout_is_never_followed(source_repo, tmp_path, when):
    other = _make_other_checkout(tmp_path)
    before = _snapshot_other(other)
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        moved = wt.path.with_name(wt.path.name + "-moved")
        os.rename(wt.path, moved)
        os.symlink(other, wt.path)
        with pytest.raises(ScratchWorktreeCorrupted, match="tree .* is not the directory this run created"):
            if when == "reset":
                wt.reset()
            elif when == "git":
                wt.git(["status", "--porcelain"])
            else:
                wt.assert_only_expected_dirty([])
    shutil.rmtree(moved)
    if os.path.lexists(wt.path):
        os.unlink(wt.path)
    assert _snapshot_other(other) == before


def test_tree_replaced_by_a_fresh_real_directory_is_rejected(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        shutil.rmtree(wt.path)
        os.mkdir(wt.path)
        with pytest.raises(ScratchWorktreeCorrupted, match="tree .* is not the directory this run created"):
            wt.reset()
    if os.path.lexists(wt.path):
        os.rmdir(wt.path)


def test_git_dir_replaced_by_a_copy_is_rejected(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        copy = tmp_path / "g-copy"
        shutil.copytree(wt.git_dir, copy, symlinks=True)
        shutil.rmtree(wt.git_dir)
        os.rename(copy, wt.git_dir)
        with pytest.raises(ScratchWorktreeCorrupted, match="git dir .* is not the directory this run created"):
            wt.reset()
    if os.path.lexists(wt.git_dir):
        shutil.rmtree(wt.git_dir)


@pytest.mark.parametrize("entry,kind", [
    ("commondir", "file"), ("gitdir", "file"), ("config.worktree", "file"), ("worktrees", "dir"),
    ("hooks", "dir"), ("info", "dir"), ("modules", "dir"), ("sharedindex.0123", "file"),
    ("index.lock", "file"), ("packed-refs", "file"), ("description", "file"), ("COMMIT_EDITMSG", "file"),
    ("FETCH_HEAD", "file"), ("MERGE_HEAD", "file"), ("Config", "file"),
])
def test_unexpected_git_dir_entry_fails_closed(source_repo, tmp_path, entry, kind):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        p = wt.git_dir / entry
        if kind == "dir":
            p.mkdir()
        elif not os.path.lexists(p):
            p.write_text("x\n")
        else:
            pytest.skip("case-insensitive filesystem: aliases an existing entry")
        with pytest.raises(ScratchWorktreeCorrupted, match="unexpected"):
            wt.reset()


@pytest.mark.parametrize("entry", ["HEAD", "objects", "refs", "index", "logs", "shallow", "ORIG_HEAD"])
def test_allowed_git_dir_entry_of_the_wrong_type_fails_closed(source_repo, tmp_path, entry):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        wt.reset()  # creates ORIG_HEAD
        p = wt.git_dir / entry
        target = tmp_path / "elsewhere"
        if os.path.isdir(p) and not os.path.islink(p):
            shutil.copytree(p, target)
        else:
            shutil.copy(p, target)
        sw._force_rmtree(p)
        os.symlink(target, p)
        with pytest.raises(ScratchWorktreeCorrupted, match="wrong file type"):
            wt.reset()


@pytest.mark.parametrize("entry", ["HEAD", "objects", "refs", "config"])
def test_missing_required_git_dir_entry_fails_closed(source_repo, tmp_path, entry):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        sw._force_rmtree(wt.git_dir / entry)
        with pytest.raises(ScratchWorktreeCorrupted, match="is missing"):
            wt.reset()


def test_git_dir_holds_only_allowlisted_entries_through_many_cycles(source_repo, tmp_path):
    allowed = set(sw._GIT_DIR_REQUIRED) | set(sw._GIT_DIR_OPTIONAL)
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        assert set(os.listdir(wt.git_dir)) <= allowed
        target = wt.path / "skills" / "bash.md"
        for i in range(3):
            (wt.path / "junk").write_text("x")
            wt.reset()
            target.write_text(f"candidate {i}\n")
            wt.assert_only_expected_dirty([target])
            assert set(os.listdir(wt.git_dir)) <= allowed, sorted(os.listdir(wt.git_dir))


def test_unknown_entry_left_by_creation_fails_at_startup_and_cleans_up(source_repo, tmp_path, monkeypatch):
    real = sw._create_private_repo

    def _leaves_extra(source, base, tree, g, env):
        real(source, base, tree, g, env)
        (g / "description").write_text("from a future git\n")

    monkeypatch.setattr(sw, "_create_private_repo", _leaves_extra)
    with pytest.raises(ScratchWorktreeCorrupted, match="description"):
        with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi"):
            pass
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith("gepa-scratch-")] == []


def test_creation_pins_git_common_dir_but_the_source_rev_parse_does_not(source_repo, tmp_path, monkeypatch):
    calls = []
    real_run = subprocess.run

    def _spy(cmd, *a, **kw):
        if cmd and cmd[0] == "git" and "--git-common-dir" not in cmd:
            calls.append((list(cmd), dict(kw.get("env") or {})))
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(sw.subprocess, "run", _spy)
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        wt.reset()
    monkeypatch.setattr(sw.subprocess, "run", real_run)
    assert any("rev-parse" in cmd and "--verify" in cmd for cmd, _ in calls)
    for cmd, env in calls:
        if any(a.startswith("--git-dir=") for a in cmd):
            assert env.get("GIT_COMMON_DIR") == str(wt.git_dir), cmd
        elif "rev-parse" in cmd and "--verify" in cmd:
            assert "GIT_COMMON_DIR" not in env, cmd


def test_global_excludes_file_cannot_hide_an_untracked_file_from_status(source_repo, tmp_path, monkeypatch):
    """GIT_CONFIG_GLOBAL=/dev/null does not stop git reading the default
    $XDG_CONFIG_HOME/git/ignore; core.excludesFile=/dev/null does."""
    xdg = tmp_path / "xdg"
    (xdg / "git").mkdir(parents=True)
    (xdg / "git" / "ignore").write_text("hidden.txt\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        wt.reset()
        (wt.path / "hidden.txt").write_text("planted\n")
        with pytest.raises(ScratchWorktreeCorrupted, match="hidden.txt"):
            wt.assert_only_expected_dirty([])


# --------------------------------------------------------------------------
# The orchestrator's own file operations go through the held directory fds
# --------------------------------------------------------------------------


def _swap_after_first_check(monkeypatch, wt, swap):
    """Run the real _verify_git_dir once, then `swap()`: the swap lands
    between the check and the file operations reset() does next. Later
    calls run the real check (and so raise)."""
    real = type(wt)._verify_git_dir
    state = {"done": False}

    def _check_then_swap(self):
        real(self)
        if not state["done"]:
            state["done"] = True
            swap()

    monkeypatch.setattr(type(wt), "_verify_git_dir", _check_then_swap)


def test_tree_swapped_after_the_check_is_not_followed_by_the_dot_git_removal(source_repo, tmp_path, monkeypatch):
    other = _make_other_checkout(tmp_path)
    before = _snapshot_other(other)
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        moved = wt.path.with_name(wt.path.name + "-moved")
        (wt.path / ".git").mkdir()
        (wt.path / ".git" / "planted").write_text("x\n")

        def _swap():
            os.rename(wt.path, moved)
            os.symlink(other, wt.path)

        _swap_after_first_check(monkeypatch, wt, _swap)
        with pytest.raises(ScratchWorktreeCorrupted, match="tree .* is not the directory this run created"):
            wt.reset()
        monkeypatch.undo()
        # The removal went through the held fd: it hit our tree, not theirs.
        assert not os.path.lexists(moved / ".git")
    shutil.rmtree(moved)
    os.unlink(wt.path)
    assert _snapshot_other(other) == before


def test_git_dir_swapped_after_the_check_is_not_followed_by_the_index_unlink(source_repo, tmp_path, monkeypatch):
    other = _make_other_checkout(tmp_path)
    before = _snapshot_other(other)
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        moved = wt.git_dir.with_name(wt.git_dir.name + "-moved")

        def _swap():
            os.rename(wt.git_dir, moved)
            os.symlink(other / ".git", wt.git_dir)

        _swap_after_first_check(monkeypatch, wt, _swap)
        with pytest.raises(ScratchWorktreeCorrupted, match="git dir .* is not the directory this run created"):
            wt.reset()
        monkeypatch.undo()
        assert not os.path.lexists(moved / "index")
    shutil.rmtree(moved)
    os.unlink(wt.git_dir)
    assert _snapshot_other(other) == before


def test_tree_swapped_before_the_nested_dot_git_walk_is_not_followed(source_repo, tmp_path, monkeypatch):
    """A swap after the last git call and undone later is not detected (the
    checks are point-in-time), but the walk itself never leaves our tree."""
    other = _make_other_checkout(tmp_path)
    (other / "sub").mkdir()
    (other / "sub" / ".git").write_text("gitdir: elsewhere\n")
    before = _snapshot_other(other)
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        moved = wt.path.with_name(wt.path.name + "-moved")
        (wt.path / "skills" / ".git").mkdir()
        real_remove = type(wt)._remove_dot_git_entries

        def _swap_then_remove(self, top_level_only):
            if not top_level_only:
                os.rename(self.path, moved)
                os.symlink(other, self.path)
            try:
                return real_remove(self, top_level_only)
            finally:
                if not top_level_only:
                    os.unlink(self.path)
                    os.rename(moved, self.path)

        monkeypatch.setattr(type(wt), "_remove_dot_git_entries", _swap_then_remove)
        wt.reset()
        monkeypatch.undo()
        assert not os.path.lexists(wt.path / "skills" / ".git")
    assert (other / "sub" / ".git").read_text() == "gitdir: elsewhere\n"
    assert _snapshot_other(other) == before


@pytest.mark.parametrize("what", ["tree", "git dir"])
def test_teardown_never_removes_a_directory_renamed_into_place(source_repo, tmp_path, what, capsys):
    other = _make_other_checkout(tmp_path)
    before = _snapshot_other(other)
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        ours = wt.path if what == "tree" else wt.git_dir
        moved = ours.with_name(ours.name + "-moved")
        os.rename(ours, moved)
        os.rename(other, ours)
    assert "is not the directory this run created" in capsys.readouterr().err
    assert _snapshot_other(ours) == before
    os.rename(ours, other)
    shutil.rmtree(moved)
    for p in _siblings(wt.path):
        if os.path.lexists(p):
            sw._force_rmtree(p)


def test_teardown_still_removes_everything_if_closing_a_held_fd_fails(source_repo, tmp_path, monkeypatch):
    real_close = os.close
    held = []

    def _close(fd):
        real_close(fd)
        if fd in held:
            raise OSError(9, "simulated close failure")

    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        held[:] = [wt.tree_fd, wt.git_dir_fd]
        monkeypatch.setattr(sw.os, "close", _close)
    monkeypatch.undo()
    assert not any(os.path.lexists(p) for p in _siblings(wt.path))


def test_held_fds_are_closed_after_exit(source_repo, tmp_path):
    before = set(os.listdir("/dev/fd"))
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        assert {str(wt.tree_fd), str(wt.git_dir_fd)} <= set(os.listdir("/dev/fd"))
    assert set(os.listdir("/dev/fd")) == before


# --------------------------------------------------------------------------
# reset() verifies what it checked out against the base commit's tree
# --------------------------------------------------------------------------


def _loose_object_path(wt, oid):
    return wt.git_dir / "objects" / oid[:2] / oid[2:]


def _write_loose_object(path, content):
    import zlib
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        os.chmod(path, 0o644)
        path.unlink()
    path.write_bytes(zlib.compress(b"blob %d\0" % len(content) + content))


def test_rewritten_loose_object_makes_reset_raise(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        oid = wt.git(["rev-parse", f"{wt.base_commit}:skills/bash.md"]).stdout.strip()
        obj = _loose_object_path(wt, oid)
        assert obj.is_file(), "the depth-1 fetch of a tiny repo should unpack to loose objects"
        _write_loose_object(obj, b"POISONED\n")
        (wt.path / "skills" / "bash.md").write_text("mutated\n")
        with pytest.raises(ScratchWorktreeCorrupted, match="skills/bash.md"):
            wt.reset()
        # Without the check, reset() would have left the poisoned blob in place.
        assert (wt.path / "skills" / "bash.md").read_text() == "POISONED\n"


def test_object_served_from_planted_alternates_makes_reset_raise(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        oid = wt.git(["rev-parse", f"{wt.base_commit}:skills/bash.md"]).stdout.strip()
        alt = tmp_path / "alt-objects"
        _write_loose_object(alt / oid[:2] / oid[2:], b"POISONED VIA ALTERNATES\n")
        (wt.git_dir / "objects" / "info").mkdir(exist_ok=True)
        (wt.git_dir / "objects" / "info" / "alternates").write_text(f"{alt}\n")
        obj = _loose_object_path(wt, oid)
        os.chmod(obj, 0o644)
        obj.unlink()
        (wt.path / "skills" / "bash.md").write_text("mutated\n")
        with pytest.raises(ScratchWorktreeCorrupted, match="skills/bash.md"):
            wt.reset()
        assert (wt.path / "skills" / "bash.md").read_text() == "POISONED VIA ALTERNATES\n"


@pytest.mark.parametrize("tamper", ["content", "exec-bit", "symlink", "missing", "dir"])
def test_checkout_check_rejects_each_kind_of_mismatch(source_repo, tmp_path, tamper):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        wt._verify_checkout()
        target = wt.path / "skills" / "bash.md"
        if tamper == "content":
            target.write_text("---\nname: bash\n---\nBash guidancE.\n")
        elif tamper == "exec-bit":
            target.chmod(0o755)
        elif tamper == "symlink":
            copy = tmp_path / "bash-copy.md"
            shutil.copy(target, copy)
            target.unlink()
            target.symlink_to(copy)
        elif tamper == "missing":
            target.unlink()
        else:
            target.unlink()
            target.mkdir()
        with pytest.raises(ScratchWorktreeCorrupted, match="skills/bash.md"):
            wt._verify_checkout()


def test_checkout_check_rejects_a_symlinked_parent_directory(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        copy = tmp_path / "skills-copy"
        shutil.copytree(wt.path / "skills", copy)
        shutil.rmtree(wt.path / "skills")
        (wt.path / "skills").symlink_to(copy)
        with pytest.raises(ScratchWorktreeCorrupted, match="skills"):
            wt._verify_checkout()


def test_checkout_check_covers_exec_bits_and_symlinks_in_the_base(tmp_path):
    repo = tmp_path / "rich-source"
    _init_repo(repo)
    _write_exec(repo / "bin" / "run.sh", "#!/bin/sh\necho hi\n")
    (repo / "link.md").symlink_to("AGENTS.md")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "exec and symlink")
    with scratch_worktree(repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        modes = {path: mode for mode, _, path in wt.tracked}
        assert modes[b"bin/run.sh"] == b"100755" and modes[b"link.md"] == b"120000"
        (wt.path / "bin" / "run.sh").chmod(0o644)
        (wt.path / "link.md").unlink()
        (wt.path / "link.md").symlink_to("skills/bash.md")
        wt.reset()
        assert os.access(wt.path / "bin" / "run.sh", os.X_OK)
        assert os.readlink(wt.path / "link.md") == "AGENTS.md"
        (wt.path / "link.md").unlink()
        (wt.path / "link.md").symlink_to("skills/bash.md")
        with pytest.raises(ScratchWorktreeCorrupted, match="link.md"):
            wt._verify_checkout()


def test_base_whose_checkout_differs_from_its_blobs_fails_at_startup(tmp_path):
    """An eol conversion makes every reset differ from the recorded blobs;
    that must fail at creation, not at candidate 2."""
    repo = tmp_path / "crlf-source"
    _init_repo(repo)
    (repo / ".gitattributes").write_text("*.md text eol=crlf\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "crlf")
    with pytest.raises(ScratchWorktreeCorrupted, match="AGENTS.md"):
        with scratch_worktree(repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi"):
            pass
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith("gepa-scratch-")] == []
