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

    # write_components_back() already wrote changed_files to disk before this
    # function runs -- any failure below leaves those writes in place, so
    # name them in the error rather than leaving the caller to guess.
    ref_check = subprocess.run(
        ["git", "check-ref-format", "--branch", branch_name],
        cwd=repo_root, capture_output=True, text=True,
    )
    if ref_check.returncode != 0:
        raise ValueError(f"invalid branch name {branch_name!r}: {ref_check.stderr.strip()}")

    steps = [
        ["git", "checkout", "-b", branch_name],
        ["git", "add", "--", *[str(p) for p in changed_files]],
        ["git", "commit", "-q", "-m", commit_message],
    ]
    for step in steps:
        try:
            subprocess.run(step, cwd=repo_root, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"{' '.join(step)!r} failed: {e.stderr}\n"
                f"Files already written to disk but NOT committed: "
                f"{', '.join(str(p) for p in changed_files)}"
            ) from e


def create_pull_request(repo_root: Path, branch_name: str, title: str, body: str) -> str:
    """Push branch_name and open a real GitHub PR via gh. NEVER call this
    without explicit human intent -- it is a real, visible-to-others action.
    Returns the PR URL."""
    try:
        subprocess.run(["git", "push", "-u", "origin", branch_name], cwd=repo_root,
                        check=True, capture_output=True, text=True)
        result = subprocess.run(
            ["gh", "pr", "create", "--title", title, "--body", body],
            cwd=repo_root, check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"{' '.join(e.cmd)!r} failed: {e.stderr}") from e
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
    import yaml

    from benchmarks.self_improve.components import _resolve_component_path, write_components_back

    # Real gap, confirmed by review: write_components_back() only WARNS and
    # skips an `optimized` pred_name it doesn't recognize (the right contract
    # for its own direct callers/tests), but apply_and_open_pr() is the
    # highest-stakes caller -- it commits (and can open a real PR for)
    # whatever DID get written as if the whole `optimized` set were applied.
    # A scope mismatch (e.g. applying a full-scope run's output against a
    # scoped-down pilot components.yaml, per README's own documented
    # pilot-then-full-run workflow) would otherwise commit a silently partial
    # subset with no indication anything was dropped. Abort loudly here,
    # before any file is written.
    all_mapping = yaml.safe_load(Path(components_yaml_path).read_text()) or {}
    unknown = set(optimized) - set(all_mapping)
    if unknown:
        raise ValueError(
            f"apply_and_open_pr: {sorted(unknown)} not in {components_yaml_path} -- "
            "optimized and components.yaml were likely generated from different scopes"
        )

    changed = write_components_back(components_yaml_path, repo_root, optimized)
    if not changed:
        return None

    # Map changed file paths back to pred_name (NOT the file's own stem --
    # e.g. skills/tools/bash.md's stem is "bash", but score_deltas and the
    # caller's `optimized` dict are keyed by pred_name "skills_tools_bash";
    # a bare filename stem would silently never match either).
    # Resolved the same way write_components_back() resolved them, so a
    # components.yaml entry that would escape repo_root is rejected here too --
    # restricted to pred_names actually in `optimized` (real bug, confirmed by
    # review: resolving EVERY components.yaml entry here, including ones
    # unrelated to this call, meant an unrelated escaping/invalid entry could
    # raise AFTER write_components_back() already wrote the valid files above,
    # leaving the working tree dirty with no branch/commit and the PR flow
    # aborted -- entries write_components_back() never touches don't need to
    # be (re-)validated here at all).
    path_to_pred_name = {
        _resolve_component_path(repo_root, rel_path): pred_name
        for pred_name, rel_path in all_mapping.items()
        if pred_name in optimized
    }
    changed_names = [path_to_pred_name.get(p.resolve(), p.stem) for p in changed]
    commit_message = build_commit_message(changed_names, score_deltas)
    create_branch_and_commit(repo_root, branch_name, changed, commit_message)

    if not push_and_open_pr:
        return None

    title = f"self-improve: GEPA-proposed rewrite ({len(changed)} component(s))"
    return create_pull_request(repo_root, branch_name, title, commit_message)
