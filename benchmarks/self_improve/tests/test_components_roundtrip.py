import pytest
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
    # .resolve(): real brittleness, confirmed by review -- write_components_back
    # returns paths resolved through resolve_contained_path (which .resolve()s
    # both base and result), so on a platform where pytest's tmp_path goes
    # through a symlink (e.g. macOS /var -> /private/var), an unresolved
    # comparison could differ even though the write genuinely succeeded.
    assert changed == [(tmp_path / "skills" / "tools" / "bash.md").resolve()]
    written = (tmp_path / "skills" / "tools" / "bash.md").read_text()
    assert "target_tool: bash" in written  # frontmatter still present
    assert "v2" in written


def _make_repo_with_escaping_entry(tmp_path, rel_path):
    (tmp_path / "AGENTS.md").write_text("# little-coder\n\nBody text.\n")
    components_yaml = tmp_path / "components.yaml"
    components_yaml.write_text(yaml.dump({"escaping": rel_path}))
    return components_yaml


@pytest.mark.parametrize("rel_path", ["../../etc/passwd", "/etc/passwd"])
def test_load_components_rejects_path_that_escapes_repo_root(tmp_path, rel_path):
    """Real hardening gap, confirmed by review: a components.yaml entry with
    an absolute path or '../' traversal must be rejected, not silently read
    from outside repo_root."""
    components_yaml = _make_repo_with_escaping_entry(tmp_path, rel_path)
    with pytest.raises(ValueError, match="escapes repo_root"):
        load_components(components_yaml, repo_root=tmp_path)


@pytest.mark.parametrize("rel_path", ["../../etc/passwd", "/etc/passwd"])
def test_write_components_back_rejects_path_that_escapes_repo_root(tmp_path, rel_path):
    """Same containment guard on the write path -- this is the side that
    could otherwise overwrite an arbitrary file outside repo_root."""
    components_yaml = _make_repo_with_escaping_entry(tmp_path, rel_path)
    with pytest.raises(ValueError, match="escapes repo_root"):
        write_components_back(components_yaml, repo_root=tmp_path, optimized={"escaping": "pwned"})


def test_write_components_back_warns_on_pred_name_missing_from_components_yaml(tmp_path, caplog):
    """Real gap, confirmed by review: applying an `optimized` dict against a
    components.yaml with a smaller/different scope (e.g. a full-scope run's
    output applied against a scoped-down pilot yaml, per README.md's
    documented pilot-then-full-run workflow) used to silently drop every
    unmatched pred_name with zero signal -- `changed` just came back shorter
    than expected, with no way to tell "nothing changed" from "scope
    mismatch" apart."""
    components_yaml = _make_repo(tmp_path)
    with caplog.at_level("WARNING"):
        changed = write_components_back(
            components_yaml, repo_root=tmp_path,
            optimized={"not_a_real_pred_name": "some optimized text"},
        )
    assert changed == []
    assert "not_a_real_pred_name" in caplog.text
