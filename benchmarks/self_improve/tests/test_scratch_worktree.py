"""scratch_worktree.py: all tests run against a THROWAWAY `git init` repo in
tmp_path -- NEVER the real checkout. The session-scoped _no_stray_real_worktrees
fixture in conftest.py is the backstop that fails the whole session if that
invariant is ever violated."""
import subprocess

import pytest

from benchmarks.self_improve.scratch_worktree import (
    ScratchWorktree,
    ScratchWorktreeCorrupted,
    ScratchWorktreeError,
    SCRATCH_MARKER_NAME,
    prune_stale,
    resolve_pi_bin,
    scratch_worktree,
)


@pytest.fixture
def source_repo(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "AGENTS.md").write_text("# little-coder\n\nBody text.\n")
    (repo / "skills").mkdir()
    (repo / "skills" / "bash.md").write_text("---\nname: bash\n---\nBash guidance.\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo, check=True)
    return repo


def _current_branch(repo_root):
    return subprocess.run(
        ["git", "symbolic-ref", "-q", "--short", "HEAD"],
        cwd=repo_root, capture_output=True, text=True,
    )


def _worktree_list(repo_root):
    return subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=repo_root, capture_output=True, text=True, check=True,
    ).stdout


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


def test_worktree_is_detached_and_creates_no_branch(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        result = _current_branch(wt.path)
        assert result.returncode != 0  # symbolic-ref fails on a detached HEAD
    branches = subprocess.run(["git", "branch", "--list"], cwd=source_repo,
                               capture_output=True, text=True, check=True).stdout
    assert branches.strip() in ("* main", "* master", "")  # only the original branch, untouched


def test_worktree_contains_the_committed_files(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        assert (wt.path / "AGENTS.md").read_text() == "# little-coder\n\nBody text.\n"
        assert (wt.path / "skills" / "bash.md").exists()


def test_two_managers_get_distinct_paths_and_coexist(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt1:
        with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt2:
            assert wt1.path != wt2.path
            assert wt1.path.is_dir()
            assert wt2.path.is_dir()


def test_worktree_removed_and_deregistered_on_normal_exit(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        scratch_path = wt.path
        assert scratch_path.is_dir()
    assert not scratch_path.exists()
    assert str(scratch_path) not in _worktree_list(source_repo)


def test_worktree_removed_when_body_raises(source_repo, tmp_path):
    scratch_path_holder = {}
    with pytest.raises(RuntimeError, match="boom"):
        with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
            scratch_path_holder["path"] = wt.path
            raise RuntimeError("boom")
    assert not scratch_path_holder["path"].exists()
    assert str(scratch_path_holder["path"]) not in _worktree_list(source_repo)


def test_worktree_removed_when_body_raises_keyboardinterrupt(source_repo, tmp_path):
    scratch_path_holder = {}
    with pytest.raises(KeyboardInterrupt):
        with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
            scratch_path_holder["path"] = wt.path
            raise KeyboardInterrupt()
    assert not scratch_path_holder["path"].exists()


def test_keep_on_error_preserves_the_worktree(source_repo, tmp_path, capsys):
    scratch_path_holder = {}
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi", keep=True) as wt:
        scratch_path_holder["path"] = wt.path
    assert scratch_path_holder["path"].exists()
    assert "PRESERVED FOR POST-MORTEM" in capsys.readouterr().out
    # manual cleanup so this test doesn't itself leak a registration
    subprocess.run(["git", "worktree", "remove", "--force", str(scratch_path_holder["path"])],
                    cwd=source_repo, check=True)


def test_reset_restores_tree_after_arbitrary_mutation(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        (wt.path / "AGENTS.md").write_text("mutated\n")
        (wt.path / "skills" / "bash.md").unlink()
        (wt.path / "untracked_junk.txt").write_text("junk\n")
        wt.reset()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=wt.path,
                                 capture_output=True, text=True, check=True).stdout
        # SCRATCH_MARKER_NAME is never committed to the source repo, so it
        # always shows as untracked (`??`) -- that's the marker doing its
        # job (surviving `reset()`'s clean -fdx), not leftover corruption.
        assert status.strip() == f"?? {SCRATCH_MARKER_NAME}"
        assert (wt.path / "AGENTS.md").read_text() == "# little-coder\n\nBody text.\n"
        assert (wt.path / "skills" / "bash.md").exists()
        assert not (wt.path / "untracked_junk.txt").exists()


def test_reset_preserves_node_modules_symlink(source_repo, tmp_path):
    """A bare `git clean -fdx` would delete a gitignored node_modules symlink
    -- would otherwise regress into an opaque "pi CLI not found" on the NEXT
    candidate evaluation within the same worktree."""
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        (wt.path / ".gitignore").write_text("node_modules/\n")
        (wt.path / "node_modules").mkdir()
        (wt.path / "node_modules" / "marker.txt").write_text("still here\n")
        wt.reset()
        assert (wt.path / "node_modules" / "marker.txt").exists()


def test_reset_preserves_the_scratch_marker_file(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        assert (wt.path / SCRATCH_MARKER_NAME).exists()
        wt.reset()
        assert (wt.path / SCRATCH_MARKER_NAME).exists()


def test_assert_only_expected_dirty_passes_when_only_expected_files_changed(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        target = wt.path / "skills" / "bash.md"
        target.write_text("---\nname: bash\n---\nRevised.\n")
        wt.assert_only_expected_dirty([target])  # must not raise


def test_assert_only_expected_dirty_raises_on_an_unexpected_modification(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        (wt.path / "skills" / "bash.md").write_text("expected change\n")
        (wt.path / "AGENTS.md").write_text("UNEXPECTED change\n")
        with pytest.raises(ScratchWorktreeCorrupted, match="AGENTS.md"):
            wt.assert_only_expected_dirty([wt.path / "skills" / "bash.md"])


def test_base_commit_is_pinned_even_if_source_head_moves(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi") as wt:
        pinned = wt.base_commit
        # Source repo advances after the scratch worktree was created.
        (source_repo / "AGENTS.md").write_text("new content\n")
        subprocess.run(["git", "add", "-A"], cwd=source_repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "advance"], cwd=source_repo, check=True)
        wt.reset()
        assert wt.base_commit == pinned
        assert (wt.path / "AGENTS.md").read_text() == "# little-coder\n\nBody text.\n"


def test_scratch_path_refuses_to_reuse_an_existing_directory(source_repo, tmp_path, monkeypatch):
    import uuid
    fixed_uuid = uuid.UUID(int=0)
    monkeypatch.setattr("benchmarks.self_improve.scratch_worktree.uuid.uuid4", lambda: fixed_uuid)
    import os
    scratch_path = tmp_path / f"gepa-scratch-{os.getpid()}-{fixed_uuid.hex[:8]}"
    scratch_path.mkdir()
    with pytest.raises(ScratchWorktreeError, match="already exists"):
        with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi"):
            pass


def test_prune_stale_deregisters_a_manually_deleted_worktree(source_repo, tmp_path):
    with scratch_worktree(source_repo, parent_dir=tmp_path, pi_bin=tmp_path / "pi", keep=True) as wt:
        scratch_path = wt.path
    # Simulate a SIGKILL: the directory disappears without `worktree remove`.
    import shutil
    shutil.rmtree(scratch_path)
    assert str(scratch_path) in _worktree_list(source_repo)
    prune_stale(source_repo)
    assert str(scratch_path) not in _worktree_list(source_repo)
