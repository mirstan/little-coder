import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rpc_client  # noqa: E402


def test_build_system_prompt_returns_agents_md_path_when_no_principles(tmp_path, monkeypatch):
    """Backward-compat guarantee: with no PRINCIPLES.md present, behavior is
    byte-identical to today -- the generated-file mechanism must not
    activate at all."""
    (tmp_path / "AGENTS.md").write_text("# little-coder\n\nBody.\n")
    monkeypatch.setattr(rpc_client, "REPO_ROOT", tmp_path)
    path = rpc_client._build_system_prompt()
    assert path == tmp_path / "AGENTS.md"


def test_build_system_prompt_concatenates_when_principles_present(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("# little-coder\n\nBody.\n")
    (tmp_path / "PRINCIPLES.md").write_text("Be concise.\n")
    monkeypatch.setattr(rpc_client, "REPO_ROOT", tmp_path)
    path = rpc_client._build_system_prompt()
    content = path.read_text()
    assert "# little-coder" in content
    assert "# Principles" in content
    assert "Be concise." in content
    # generated file lives under .pi/, not overwriting AGENTS.md itself
    assert path != tmp_path / "AGENTS.md"


def test_build_system_prompt_generated_file_is_gitignored_path(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("x")
    (tmp_path / "PRINCIPLES.md").write_text("y")
    monkeypatch.setattr(rpc_client, "REPO_ROOT", tmp_path)
    path = rpc_client._build_system_prompt()
    assert ".pi" in path.parts
    assert path.name.startswith(".system-prompt.generated")


def test_build_system_prompt_refreshes_on_each_call(tmp_path, monkeypatch):
    """Edits to PRINCIPLES.md between two PiRpc constructions must be picked
    up -- the generated file is rewritten every call, not cached."""
    (tmp_path / "AGENTS.md").write_text("A")
    (tmp_path / "PRINCIPLES.md").write_text("v1")
    monkeypatch.setattr(rpc_client, "REPO_ROOT", tmp_path)
    p1 = rpc_client._build_system_prompt()
    assert "v1" in p1.read_text()
    (tmp_path / "PRINCIPLES.md").write_text("v2")
    p2 = rpc_client._build_system_prompt()
    assert "v2" in p2.read_text()
    assert "v1" not in p2.read_text()
