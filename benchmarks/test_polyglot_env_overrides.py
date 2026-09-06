"""POLYGLOT_BENCHMARK_ROOT / POLYGLOT_RESULTS_FILE / POLYGLOT_LOG_ROOT: let a
live-eval test harness point a subprocess invocation of aider_polyglot.py at a
synthetic benchmark root/results file/log dir, without which none of that
machinery is testable without a real paid model run (BENCHMARK_ROOT feeds
LANG_DESCRIPTORS at import time). Backward-compat: unset -> byte-identical to
the existing hardcoded defaults."""
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _reload_aider_polyglot():
    if "aider_polyglot" in sys.modules:
        del sys.modules["aider_polyglot"]
    import aider_polyglot
    return aider_polyglot


def test_benchmark_root_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("POLYGLOT_BENCHMARK_ROOT", raising=False)
    mod = _reload_aider_polyglot()
    assert mod.BENCHMARK_ROOT == Path.home() / "Documents" / "polyglot-benchmark"


def test_benchmark_root_uses_override_when_set(monkeypatch, tmp_path):
    monkeypatch.setenv("POLYGLOT_BENCHMARK_ROOT", str(tmp_path))
    mod = _reload_aider_polyglot()
    assert mod.BENCHMARK_ROOT == tmp_path


def test_benchmark_root_falls_back_to_default_when_override_is_empty_string(monkeypatch):
    """Real bug, confirmed by review: os.environ.get(name, default) returns ""
    (not the default) when the var is exported EMPTY. Path("") == Path("."),
    whose .exists() is True, silently pointing at the wrong directory."""
    monkeypatch.setenv("POLYGLOT_BENCHMARK_ROOT", "")
    mod = _reload_aider_polyglot()
    assert mod.BENCHMARK_ROOT == Path.home() / "Documents" / "polyglot-benchmark"


def test_benchmark_root_feeds_lang_descriptors_practice_dir(monkeypatch, tmp_path):
    """LANG_DESCRIPTORS is built at IMPORT time from BENCHMARK_ROOT -- this is
    the property that makes a subprocess re-exec (not just reading the
    constant) necessary for the override to actually change agent behavior."""
    monkeypatch.setenv("POLYGLOT_BENCHMARK_ROOT", str(tmp_path))
    mod = _reload_aider_polyglot()
    assert mod.LANG_DESCRIPTORS["python"]["practice_dir"] == tmp_path / "python" / "exercises" / "practice"


def test_results_file_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("POLYGLOT_RESULTS_FILE", raising=False)
    mod = _reload_aider_polyglot()
    assert mod.RESULTS_FILE == Path(mod.__file__).parent / "results_full_polyglot.json"


def test_results_file_uses_override_when_set(monkeypatch, tmp_path):
    target = tmp_path / "custom_results.json"
    monkeypatch.setenv("POLYGLOT_RESULTS_FILE", str(target))
    mod = _reload_aider_polyglot()
    assert mod.RESULTS_FILE == target


def test_results_file_falls_back_to_default_when_override_is_empty_string(monkeypatch):
    monkeypatch.setenv("POLYGLOT_RESULTS_FILE", "")
    mod = _reload_aider_polyglot()
    assert mod.RESULTS_FILE == Path(mod.__file__).parent / "results_full_polyglot.json"


def test_log_root_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("POLYGLOT_LOG_ROOT", raising=False)
    mod = _reload_aider_polyglot()
    assert mod.LOG_ROOT == Path(mod.__file__).parent / "full_polyglot_logs"


def test_log_root_uses_override_when_set(monkeypatch, tmp_path):
    target = tmp_path / "custom_logs"
    monkeypatch.setenv("POLYGLOT_LOG_ROOT", str(target))
    mod = _reload_aider_polyglot()
    assert mod.LOG_ROOT == target


def test_log_root_falls_back_to_default_when_override_is_empty_string(monkeypatch):
    monkeypatch.setenv("POLYGLOT_LOG_ROOT", "")
    mod = _reload_aider_polyglot()
    assert mod.LOG_ROOT == Path(mod.__file__).parent / "full_polyglot_logs"
