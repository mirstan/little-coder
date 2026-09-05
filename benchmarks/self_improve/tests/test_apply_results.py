"""apply_results.py: local branch+commit logic only. The push/PR step
(create_pull_request) is NEVER exercised here -- it performs real,
visible-to-others actions (git push, gh pr create) and must only run when a
human explicitly invokes the real CLI with intent, never as a side effect of
running the test suite."""
import subprocess

import pytest

from benchmarks.self_improve.apply_results import (
    build_commit_message,
    create_branch_and_commit,
)


@pytest.fixture
def scratch_repo(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "file.md").write_text("original\n")
    subprocess.run(["git", "add", "file.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=tmp_path, check=True)
    return tmp_path


def _current_branch(repo_root):
    return subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo_root, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _log_messages(repo_root):
    return subprocess.run(
        ["git", "log", "--format=%s"],
        cwd=repo_root, check=True, capture_output=True, text=True,
    ).stdout.splitlines()


def test_create_branch_and_commit_creates_new_branch(scratch_repo):
    (scratch_repo / "file.md").write_text("changed\n")
    create_branch_and_commit(
        repo_root=scratch_repo,
        branch_name="self-improve/gepa-test",
        changed_files=[scratch_repo / "file.md"],
        commit_message="test commit",
    )
    assert _current_branch(scratch_repo) == "self-improve/gepa-test"


def test_create_branch_and_commit_commits_only_changed_files(scratch_repo):
    (scratch_repo / "file.md").write_text("changed\n")
    (scratch_repo / "untouched.md").write_text("should not be committed\n")
    create_branch_and_commit(
        repo_root=scratch_repo,
        branch_name="self-improve/gepa-test2",
        changed_files=[scratch_repo / "file.md"],
        commit_message="test commit",
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=scratch_repo,
        check=True, capture_output=True, text=True,
    ).stdout
    assert "untouched.md" in status  # still untracked, not committed
    assert "file.md" not in status   # committed, working tree clean for it


def test_create_branch_and_commit_message_is_used(scratch_repo):
    (scratch_repo / "file.md").write_text("changed\n")
    create_branch_and_commit(
        repo_root=scratch_repo,
        branch_name="self-improve/gepa-test3",
        changed_files=[scratch_repo / "file.md"],
        commit_message="a distinctive commit message",
    )
    assert "a distinctive commit message" in _log_messages(scratch_repo)[0]


def test_create_branch_and_commit_raises_when_no_changed_files(scratch_repo):
    """Never create an empty commit / empty branch -- caller must check
    write_components_back()'s return value first."""
    with pytest.raises(ValueError):
        create_branch_and_commit(
            repo_root=scratch_repo,
            branch_name="self-improve/gepa-empty",
            changed_files=[],
            commit_message="nothing changed",
        )


def test_build_commit_message_cites_component_score_deltas():
    msg = build_commit_message(
        changed_components=["skills_tools_bash"],
        score_deltas={"skills_tools_bash": (0.4, 0.7)},
    )
    assert "skills_tools_bash" in msg
    assert "0.4" in msg
    assert "0.7" in msg
