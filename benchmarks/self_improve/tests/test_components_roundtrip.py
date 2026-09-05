import yaml

from benchmarks.self_improve.components import (
    load_components,
    reattach_frontmatter,
    split_frontmatter,
    write_components_back,
)

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


def test_split_frontmatter_separates_yaml_and_body():
    frontmatter, body = split_frontmatter(FRONTMATTER_FIXTURE)
    assert "target_tool: bash" in frontmatter
    assert body.strip().startswith("## `bash` Tool")


def test_split_frontmatter_handles_file_with_no_frontmatter():
    """AGENTS.md and PRINCIPLES.md have no YAML frontmatter block at all --
    must return (None, full_text), not raise or misparse the first '#' line
    as a delimiter."""
    frontmatter, body = split_frontmatter("# little-coder\n\nSome text.\n")
    assert frontmatter is None
    assert body == "# little-coder\n\nSome text.\n"


def test_reattach_frontmatter_reconstructs_original_byte_for_byte():
    """The critical regression guard: split then reattach with an UNCHANGED
    body must reproduce the exact original bytes, including trailing
    newline conventions."""
    reconstructed = reattach_frontmatter(*split_frontmatter(FRONTMATTER_FIXTURE))
    assert reconstructed == FRONTMATTER_FIXTURE


def test_reattach_frontmatter_replaces_only_body_when_changed():
    frontmatter, _old_body = split_frontmatter(FRONTMATTER_FIXTURE)
    new_body = "## `bash` Tool (revised)\nNew guidance text.\n"
    reconstructed = reattach_frontmatter(frontmatter, new_body)
    assert "target_tool: bash" in reconstructed   # frontmatter untouched
    assert "revised" in reconstructed
    assert "Execute a shell command" not in reconstructed  # old body gone


def test_reattach_frontmatter_with_none_frontmatter_returns_body_as_is():
    assert reattach_frontmatter(None, "plain text\n") == "plain text\n"


def _make_repo(tmp_path):
    (tmp_path / "AGENTS.md").write_text("# little-coder\n\nBody text.\n")
    skills_dir = tmp_path / "skills" / "tools"
    skills_dir.mkdir(parents=True)
    (skills_dir / "bash.md").write_text(FRONTMATTER_FIXTURE)

    components_yaml = tmp_path / "components.yaml"
    components_yaml.write_text(yaml.dump({
        "agents_md": "AGENTS.md",
        "skills_tools_bash": "skills/tools/bash.md",
    }))
    return components_yaml


def test_load_components_from_real_repo_files_via_components_yaml(tmp_path):
    """End-to-end against REAL files (not fixtures) -- copy AGENTS.md and one
    real skill file into a temp repo layout, drive components.yaml, and
    confirm load_components() produces bodies with frontmatter stripped."""
    components_yaml = _make_repo(tmp_path)
    comps = load_components(components_yaml, repo_root=tmp_path)
    assert "target_tool: bash" not in comps["skills_tools_bash"]  # frontmatter stripped
    assert "## `bash` Tool" in comps["skills_tools_bash"]
    assert comps["agents_md"] == "# little-coder\n\nBody text.\n"


def test_write_components_back_preserves_untouched_files(tmp_path):
    """apply_results.py must skip writing any file whose optimized body is
    IDENTICAL to its current body -- prevents spurious git diffs / PR noise
    for components GEPA left alone."""
    components_yaml = _make_repo(tmp_path)
    changed = write_components_back(
        components_yaml, repo_root=tmp_path,
        optimized={"skills_tools_bash": "## `bash` Tool\nExecute a shell command and return stdout+stderr.\n"},
    )
    assert changed == []  # identical to current body -- no file touched


def test_write_components_back_reports_changed_files(tmp_path):
    components_yaml = _make_repo(tmp_path)
    changed = write_components_back(
        components_yaml, repo_root=tmp_path,
        optimized={"skills_tools_bash": "## `bash` Tool (v2)\nRevised.\n"},
    )
    assert changed == [tmp_path / "skills" / "tools" / "bash.md"]
    written = (tmp_path / "skills" / "tools" / "bash.md").read_text()
    assert "target_tool: bash" in written  # frontmatter still present
    assert "v2" in written
