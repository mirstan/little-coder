"""apply_results.py: local branch+commit logic only. The push/PR step
(create_pull_request) is NEVER exercised here -- it performs real,
visible-to-others actions (git push, gh pr create) and must only run when a
human explicitly invokes the real CLI with intent, never as a side effect of
running the test suite."""
import subprocess

import pytest
import yaml

from benchmarks.self_improve.apply_results import (
    apply_and_open_pr,
    build_commit_message,
    create_branch_and_commit,
    create_pull_request,
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


def test_create_branch_and_commit_rejects_malformed_branch_name(scratch_repo):
    """Real hardening gap, confirmed by review: an unvalidated branch_name
    passed straight to `git checkout -b` could be misparsed as a git option
    (e.g. a name starting with '-'). check-ref-format catches this before
    any git subprocess that could mutate the repo runs."""
    (scratch_repo / "file.md").write_text("changed\n")
    # Real brittleness, confirmed by review: git init's default branch name
    # inherits init.defaultBranch from the environment (global/CI config
    # could set e.g. "trunk"), not always "main" or "master" -- the real
    # intent is only "the branch was not switched," so capture and compare
    # against the ACTUAL starting branch instead of assuming its name.
    original_branch = _current_branch(scratch_repo)
    with pytest.raises(ValueError, match="invalid branch name"):
        create_branch_and_commit(
            repo_root=scratch_repo,
            branch_name="-not-a-real-branch",
            changed_files=[scratch_repo / "file.md"],
            commit_message="test commit",
        )
    # rejected before any git state changed
    assert _current_branch(scratch_repo) == original_branch


def test_create_branch_and_commit_handles_path_starting_with_dash(scratch_repo):
    """Real hardening gap, confirmed by review: `git add` without a `--`
    path-terminator would parse a changed file name starting with '-' as a
    git option instead of a path."""
    dash_file = scratch_repo / "-weird-name.md"
    dash_file.write_text("changed\n")
    create_branch_and_commit(
        repo_root=scratch_repo,
        branch_name="self-improve/gepa-dash-test",
        changed_files=[dash_file],
        commit_message="test commit",
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=scratch_repo,
        check=True, capture_output=True, text=True,
    ).stdout
    assert status == ""  # committed cleanly, not left untracked/misparsed


def test_build_commit_message_cites_component_score_deltas():
    msg = build_commit_message(
        changed_components=["skills_tools_bash"],
        score_deltas={"skills_tools_bash": (0.4, 0.7)},
    )
    assert "skills_tools_bash" in msg
    assert "0.4" in msg
    assert "0.7" in msg


# ── apply_and_open_pr end-to-end (push_and_open_pr=False, the safe default)
# The full orchestrator (write_components_back + create_branch_and_commit)
# had never been tested as a whole before -- only its two pieces separately.
# push_and_open_pr stays False throughout: create_pull_request() (git push +
# gh pr create) is never reached by these tests.

FRONTMATTER_FIXTURE = """---
name: bash-guidance
type: tool-guidance
target_tool: bash
priority: 10
token_cost: 120
user-invocable: false
---
## `bash` Tool
Execute a shell command and return stdout+stderr.
"""


def _make_component_repo(tmp_path):
    (tmp_path / "AGENTS.md").write_text("# little-coder\n\nBody text.\n")
    skills_dir = tmp_path / "skills" / "tools"
    skills_dir.mkdir(parents=True)
    (skills_dir / "bash.md").write_text(FRONTMATTER_FIXTURE)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=tmp_path, check=True)

    components_yaml = tmp_path / "components.yaml"
    components_yaml.write_text(yaml.dump({
        "agents_md": "AGENTS.md",
        "skills_tools_bash": "skills/tools/bash.md",
    }))
    return components_yaml


def test_apply_and_open_pr_writes_commits_and_preserves_frontmatter(tmp_path):
    components_yaml = _make_component_repo(tmp_path)

    result = apply_and_open_pr(
        components_yaml_path=components_yaml,
        repo_root=tmp_path,
        optimized={"skills_tools_bash": "## `bash` Tool (v2)\nRevised guidance.\n"},
        score_deltas={"skills_tools_bash": (0.4, 0.8)},
        branch_name="self-improve/gepa-e2e-test",
        push_and_open_pr=False,
    )

    assert result is None  # no PR opened -- push_and_open_pr was False
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=tmp_path,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert branch == "self-improve/gepa-e2e-test"

    written = (tmp_path / "skills" / "tools" / "bash.md").read_text()
    assert "target_tool: bash" in written  # frontmatter preserved byte-for-byte
    assert "v2" in written
    assert "Execute a shell command" not in written  # old body actually replaced

    log = subprocess.run(
        ["git", "log", "--format=%B"], cwd=tmp_path,  # %B: full message (subject+body)
        check=True, capture_output=True, text=True,
    ).stdout
    assert "skills_tools_bash" in log  # commit message cites the changed component
    assert "0.4" in log and "0.8" in log  # and its score delta


def test_apply_and_open_pr_returns_none_and_touches_nothing_when_unchanged(tmp_path):
    """optimized text identical to current body -- no branch, no commit, no
    file write. Prevents an empty/no-op PR for components GEPA left alone."""
    components_yaml = _make_component_repo(tmp_path)
    original_branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=tmp_path,
        check=True, capture_output=True, text=True,
    ).stdout.strip()

    result = apply_and_open_pr(
        components_yaml_path=components_yaml,
        repo_root=tmp_path,
        optimized={"skills_tools_bash": "## `bash` Tool\nExecute a shell command and return stdout+stderr.\n"},
        score_deltas={},
        branch_name="self-improve/gepa-noop-test",
        push_and_open_pr=False,
    )

    assert result is None
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=tmp_path,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert branch == original_branch  # never switched branches -- nothing to commit


def test_apply_and_open_pr_aborts_on_optimized_pred_name_unknown_to_components_yaml(tmp_path):
    """Real gap, confirmed by review: write_components_back() only WARNS and
    skips an unrecognized pred_name (the right contract for its own direct
    callers), but apply_and_open_pr() is the highest-stakes caller -- it
    commits (and can open a real PR for) whatever DID get written as if the
    whole `optimized` set were applied. Must abort loudly on a scope
    mismatch instead of silently committing a partial subset, and must not
    write anything before it does."""
    components_yaml = _make_component_repo(tmp_path)
    with pytest.raises(ValueError, match="not in"):
        apply_and_open_pr(
            components_yaml_path=components_yaml,
            repo_root=tmp_path,
            optimized={
                "skills_tools_bash": "## `bash` Tool (v2)\nRevised guidance.\n",
                "not_a_real_pred_name": "some optimized text",
            },
            score_deltas={},
            branch_name="self-improve/gepa-scope-mismatch-test",
            push_and_open_pr=False,
        )
    written = (tmp_path / "skills" / "tools" / "bash.md").read_text()
    assert "v2" not in written  # nothing written -- aborted before any write


def test_apply_and_open_pr_ignores_unrelated_escaping_entry_in_components_yaml(tmp_path):
    """Real bug, confirmed by review: apply_and_open_pr() used to re-resolve
    EVERY components.yaml entry (not just the ones in `optimized`) to build
    its path->pred_name lookup, so an unrelated escaping/invalid entry
    elsewhere in the file raised AFTER write_components_back() already wrote
    the real, valid changes to disk -- leaving the working tree dirty with
    no branch/commit. An entry this call never touches must not block it."""
    components_yaml = _make_component_repo(tmp_path)
    mapping = yaml.safe_load(components_yaml.read_text())
    mapping["unrelated_escaping_entry"] = "../../etc/passwd"
    components_yaml.write_text(yaml.dump(mapping))

    result = apply_and_open_pr(
        components_yaml_path=components_yaml,
        repo_root=tmp_path,
        optimized={"skills_tools_bash": "## `bash` Tool (v2)\nRevised guidance.\n"},
        score_deltas={},
        branch_name="self-improve/gepa-unrelated-entry-test",
        push_and_open_pr=False,
    )
    assert result is None  # no PR opened -- push_and_open_pr was False, but no error either
    written = (tmp_path / "skills" / "tools" / "bash.md").read_text()
    assert "v2" in written


def test_apply_and_open_pr_rejects_component_path_that_escapes_repo_root(tmp_path):
    """Real hardening gap, confirmed by review: a components.yaml entry
    resolving outside repo_root must be rejected before any git operation
    runs, not silently written to disk elsewhere."""
    components_yaml = _make_component_repo(tmp_path)
    mapping = yaml.safe_load(components_yaml.read_text())
    mapping["escaping"] = "../../etc/passwd"
    components_yaml.write_text(yaml.dump(mapping))

    with pytest.raises(ValueError, match="escapes repo_root"):
        apply_and_open_pr(
            components_yaml_path=components_yaml,
            repo_root=tmp_path,
            optimized={"escaping": "pwned"},
            score_deltas={},
            branch_name="self-improve/gepa-escape-test",
            push_and_open_pr=False,
        )


# ── create_pull_request: STRUCTURAL check only, subprocess.run fully mocked.
# This never spawns a real process -- no git push, no gh pr create, no
# network. It only confirms the exact command shape (flags, order, args) is
# correct, since a typo here would otherwise only be caught by a human
# actually running it for real.

def test_create_pull_request_command_shape(monkeypatch, tmp_path):
    calls = []

    class _FakeCompletedProcess:
        stdout = "https://github.com/example/repo/pull/123\n"

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _FakeCompletedProcess()

    monkeypatch.setattr(subprocess, "run", fake_run)

    url = create_pull_request(
        repo_root=tmp_path,
        branch_name="self-improve/gepa-test",
        title="self-improve: GEPA-proposed rewrite (1 component)",
        body="commit body text",
    )

    assert url == "https://github.com/example/repo/pull/123"
    assert len(calls) == 2

    push_cmd, push_kwargs = calls[0]
    assert push_cmd == ["git", "push", "-u", "origin", "self-improve/gepa-test"]
    assert push_kwargs.get("cwd") == tmp_path
    assert push_kwargs.get("check") is True

    pr_cmd, pr_kwargs = calls[1]
    assert pr_cmd == [
        "gh", "pr", "create",
        "--title", "self-improve: GEPA-proposed rewrite (1 component)",
        "--body", "commit body text",
    ]
    assert pr_kwargs.get("cwd") == tmp_path
    assert pr_kwargs.get("check") is True
