"""main()'s --log-roots KEY=VALUE validation. Real bug, confirmed by review:
an unrecognized or malformed token was previously silently absorbed and then
silently never matched any of _ingest_all's known-source checks -- a real run
could believe a source was ingested when it was never attempted at all."""
import sys

import pytest

from benchmarks.self_improve.run_gepa import main


def _run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["run_gepa.py", *argv])
    return main()


def test_main_rejects_unknown_log_roots_key(monkeypatch, capsys, tmp_path):
    code = _run_main(monkeypatch, ["--log-roots", f"gaea={tmp_path}", "--dry-run"])
    assert code == 1
    assert "unknown key" in capsys.readouterr().err


def test_main_rejects_log_roots_token_missing_equals(monkeypatch, capsys, tmp_path):
    code = _run_main(monkeypatch, ["--log-roots", str(tmp_path), "--dry-run"])
    assert code == 1
    assert "expected KEY=VALUE" in capsys.readouterr().err


def test_main_rejects_log_roots_token_with_empty_value(monkeypatch, capsys):
    code = _run_main(monkeypatch, ["--log-roots", "gaia=", "--dry-run"])
    assert code == 1
    assert "expected KEY=VALUE" in capsys.readouterr().err


def test_main_accepts_known_log_roots_key(monkeypatch, capsys, tmp_path):
    """A recognized key with a real value must proceed past validation (into
    --dry-run's free path, not exit 1 from the log-roots check itself)."""
    code = _run_main(monkeypatch, ["--log-roots", f"gaia={tmp_path}", "--dry-run"])
    assert code == 0
    assert "unknown key" not in capsys.readouterr().err
