import pytest

from benchmarks.self_improve.path_safety import resolve_contained_path


def test_resolve_contained_path_accepts_normal_relative_path(tmp_path):
    (tmp_path / "sub").mkdir()
    resolved = resolve_contained_path(tmp_path, "sub/file.md")
    assert resolved == (tmp_path / "sub" / "file.md").resolve()


def test_resolve_contained_path_rejects_traversal_outside_base(tmp_path):
    with pytest.raises(ValueError, match="escapes base directory"):
        resolve_contained_path(tmp_path, "../../etc/passwd")


def test_resolve_contained_path_rejects_absolute_path_outside_base(tmp_path):
    with pytest.raises(ValueError, match="must be relative"):
        resolve_contained_path(tmp_path, "/etc/passwd")


def test_resolve_contained_path_rejects_absolute_path_even_when_it_resolves_inside_base(tmp_path):
    """Real bug, confirmed by review: an absolute rel_path that HAPPENS to
    point inside base used to be accepted, even though a caller building a
    second path from the same (base, rel_path) inputs differently (e.g.
    separate path components instead of one combined string) can resolve
    somewhere else entirely -- silently causing detail files to go missing
    rather than a clean, early rejection. Absolute must always be rejected,
    regardless of where it happens to point."""
    inside = tmp_path / "sub" / "file.md"
    inside.parent.mkdir(parents=True)
    with pytest.raises(ValueError, match="must be relative"):
        resolve_contained_path(tmp_path, str(inside))
