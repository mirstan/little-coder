"""The harbor adapter's opt-in pi session persistence (compaction A/B)."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("harbor")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "harbor_adapter"))
import little_coder_agent as LC  # noqa: E402


def test_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv(LC.ENV_PI_SESSION_IN_LOGS, raising=False)
    assert LC._pi_session_dir(tmp_path) is None
    assert not (tmp_path / "pi-session").exists()


def test_on_creates_dir_under_logs(tmp_path, monkeypatch):
    monkeypatch.setenv(LC.ENV_PI_SESSION_IN_LOGS, "1")
    got = LC._pi_session_dir(tmp_path)
    assert got == str(tmp_path / "pi-session")
    assert (tmp_path / "pi-session").is_dir()


def test_on_without_logs_dir_is_none(monkeypatch):
    monkeypatch.setenv(LC.ENV_PI_SESSION_IN_LOGS, "1")
    assert LC._pi_session_dir(None) is None
