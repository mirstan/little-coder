"""Apply a compiled GEPA candidate back to the real repo files and open a PR
for human review. Per the architecture (§7): PR only, never auto-commit to
main, never auto-merge.

Risk tiers, deliberately kept separate:
  - write_components_back() (components.py) + create_branch_and_commit(): pure
    local git operations, safe to test against a scratch repo.
  - create_pull_request(): pushes a branch and opens a REAL GitHub PR. This
    performs actions visible to others and must only be invoked by a human
    explicitly running the real CLI with intent -- never exercised by tests,
    never called automatically by apply_and_open_pr() without push_and_open_pr=True.
"""
import subprocess
from pathlib import Path


def build_commit_message(changed_components: list[str], score_deltas: dict[str, tuple[float, float]]) -> str:
    lines = ["self-improve: GEPA-proposed rewrite", ""]
    for name in changed_components:
        delta = score_deltas.get(name)
        if delta is not None:
            before, after = delta
            lines.append(f"- {name}: {before} -> {after}")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)


def create_branch_and_commit(
    repo_root: Path, branch_name: str, changed_files: list[Path], commit_message: str
) -> None:
    if not changed_files:
        raise ValueError("create_branch_and_commit called with no changed files -- "
                          "caller must check write_components_back()'s return value first")

    subprocess.run(["git", "checkout", "-b", branch_name], cwd=repo_root, check=True,
                    capture_output=True)
    subprocess.run(["git", "add", *[str(p) for p in changed_files]], cwd=repo_root, check=True,
                    capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", commit_message], cwd=repo_root, check=True,
                    capture_output=True)


def create_pull_request(repo_root: Path, branch_name: str, title: str, body: str) -> str:
    """Push branch_name and open a real GitHub PR via gh. NEVER call this
    without explicit human intent -- it is a real, visible-to-others action.
    Returns the PR URL."""
    subprocess.run(["git", "push", "-u", "origin", branch_name], cwd=repo_root, check=True)
    result = subprocess.run(
        ["gh", "pr", "create", "--title", title, "--body", body],
        cwd=repo_root, check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def apply_and_open_pr(
    components_yaml_path: Path,
    repo_root: Path,
    optimized: dict[str, str],
    score_deltas: dict[str, tuple[float, float]],
    branch_name: str,
    push_and_open_pr: bool = False,
) -> str | None:
    """Write optimized components back, commit locally, and -- ONLY if
    push_and_open_pr=True (default False) -- push and open a real PR.
    Returns the PR URL if one was opened, else None (including when nothing
    changed, or when push_and_open_pr is left False for local-only review)."""
    from benchmarks.self_improve.components import write_components_back

    changed = write_components_back(components_yaml_path, repo_root, optimized)
    if not changed:
        return None

    changed_names = [p.stem for p in changed]
    commit_message = build_commit_message(changed_names, score_deltas)
    create_branch_and_commit(repo_root, branch_name, changed, commit_message)

    if not push_and_open_pr:
        return None

    title = f"self-improve: GEPA-proposed rewrite ({len(changed)} component(s))"
    return create_pull_request(repo_root, branch_name, title, commit_message)
