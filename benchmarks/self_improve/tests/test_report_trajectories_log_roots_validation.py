"""report_trajectories.py main()'s --log-roots KEY=VALUE validation. Real
bug, confirmed by review: an unrecognized or malformed token was previously
silently absorbed and then silently never matched any of ingest_all's
known-source checks -- a real run could believe a source was ingested when
it was never attempted at all. Moved here from run_gepa.py, which no longer
has a --log-roots flag at all (see polyglot_adapter.py's module docstring
for why the frozen-historical-data path was replaced)."""
import sys
from pathlib import Path

import pytest

from benchmarks.self_improve.report_trajectories import _resolve_components_yaml, main, parse_log_roots


def _run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["report_trajectories.py", *argv])
    return main()


def test_main_rejects_unknown_log_roots_key(monkeypatch, capsys, tmp_path):
    code = _run_main(monkeypatch, ["--log-roots", f"gaea={tmp_path}"])
    assert code == 1
    assert "unknown key" in capsys.readouterr().err


def test_main_rejects_log_roots_token_missing_equals(monkeypatch, capsys, tmp_path):
    code = _run_main(monkeypatch, ["--log-roots", str(tmp_path)])
    assert code == 1
    assert "expected KEY=VALUE" in capsys.readouterr().err


def test_main_rejects_log_roots_token_with_empty_value(monkeypatch, capsys):
    code = _run_main(monkeypatch, ["--log-roots", "gaia="])
    assert code == 1
    assert "expected KEY=VALUE" in capsys.readouterr().err


def test_main_accepts_known_log_roots_key(monkeypatch, capsys, tmp_path):
    """A recognized key with a real value must proceed past validation (into
    the free report path, not exit 1 from the log-roots check itself)."""
    code = _run_main(monkeypatch, ["--log-roots", f"gaia={tmp_path}"])
    assert code == 0
    assert "unknown key" not in capsys.readouterr().err


def test_parse_log_roots_rejects_duplicate_key(tmp_path):
    """Real bug, confirmed by review: a duplicate key silently overwrote the
    earlier value with zero error, so requested data went missing from the
    report with no signal at all."""
    with pytest.raises(ValueError, match="duplicate key"):
        parse_log_roots([f"gaia={tmp_path}/a", f"gaia={tmp_path}/b"])


def test_main_rejects_duplicate_log_roots_key(monkeypatch, capsys, tmp_path):
    code = _run_main(monkeypatch, ["--log-roots", f"gaia={tmp_path}/a", f"gaia={tmp_path}/b"])
    assert code == 1
    assert "duplicate key" in capsys.readouterr().err


def test_resolve_components_yaml_relative_path_uses_repo_root():
    repo_root = Path("/repo")
    assert _resolve_components_yaml(repo_root, "config/components.yaml") == repo_root / "config" / "components.yaml"


def test_resolve_components_yaml_absolute_path_under_repo_root():
    repo_root = Path("/repo")
    result = _resolve_components_yaml(repo_root, "/repo/config/components.yaml")
    assert result == repo_root / "config" / "components.yaml"
