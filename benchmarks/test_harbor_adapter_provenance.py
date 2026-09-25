"""Tests for _derive_benchmark_label() and _build_environment_snapshot() in
little_coder_agent.py.

Requires the `harbor` package (only installed in harbor's own uv-tool venv,
not the plain system Python these other benchmark tests run under) purely
because little_coder_agent.py imports harbor.agents.base at module load
time; none of this file's actual assertions touch Harbor itself. See
test_harbor_adapter_timeout.py for the sibling test file covering
_resolve_trial_timeout_info()/_resolve_trial_timeout_sec().
"""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("harbor")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "harbor_adapter"))
import little_coder_agent as lca  # noqa: E402


def _write_trial_config(trial_dir: Path, task: dict, multiplier: float = 3.0):
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "config.json").write_text(json.dumps({
        "timeout_multiplier": multiplier,
        "task": task,
    }))


# ── _derive_benchmark_label ──────────────────────────────────────────────

def test_benchmark_label_none_logs_dir_falls_back_to_unknown():
    assert lca._derive_benchmark_label(None) == "terminal_bench_unknown"


def test_benchmark_label_missing_config_falls_back_to_unknown(tmp_path):
    trial_dir = tmp_path / "trial"
    logs_dir = trial_dir / "agent"
    logs_dir.mkdir(parents=True)
    # No config.json written at all.
    assert lca._derive_benchmark_label(logs_dir) == "terminal_bench_unknown"


def test_benchmark_label_package_shape_uses_source_alone(tmp_path):
    """Package-dataset config.json (task.name/ref/source, confirmed on disk
    against a real terminal-bench/terminal-bench-2-1 run) -- source alone is
    the dataset identity; the content-hash ref is a cache lookup key, not
    part of it."""
    trial_dir = tmp_path / "trial"
    _write_trial_config(trial_dir, {
        "name": "terminal-bench/train-fasttext",
        "ref": "sha256:deadbeef",
        "source": "terminal-bench/terminal-bench-2-1",
    })
    logs_dir = trial_dir / "agent"
    logs_dir.mkdir()
    assert lca._derive_benchmark_label(logs_dir) == "terminal-bench/terminal-bench-2-1"


def test_benchmark_label_legacy_shape_appends_short_commit_id(tmp_path):
    """Legacy name@version config.json (task.path/git_url/git_commit_id) --
    the same name@version string can be re-pinned to different commits, so
    a short git_commit_id disambiguates when present."""
    trial_dir = tmp_path / "trial"
    _write_trial_config(trial_dir, {
        "path": "my-task",
        "git_url": None,
        "git_commit_id": "abcdef0123456789",
        "source": "terminal-bench@2.0",
    })
    logs_dir = trial_dir / "agent"
    logs_dir.mkdir()
    assert lca._derive_benchmark_label(logs_dir) == "terminal-bench@2.0@abcdef012345"


def test_benchmark_label_legacy_shape_without_commit_id_uses_source_alone(tmp_path):
    trial_dir = tmp_path / "trial"
    _write_trial_config(trial_dir, {
        "path": "my-task",
        "git_url": None,
        "git_commit_id": None,
        "source": "terminal-bench@2.0",
    })
    logs_dir = trial_dir / "agent"
    logs_dir.mkdir()
    assert lca._derive_benchmark_label(logs_dir) == "terminal-bench@2.0"


def test_benchmark_label_missing_source_falls_back_to_unknown(tmp_path):
    trial_dir = tmp_path / "trial"
    _write_trial_config(trial_dir, {"name": "terminal-bench/train-fasttext", "ref": "sha256:x"})
    logs_dir = trial_dir / "agent"
    logs_dir.mkdir()
    assert lca._derive_benchmark_label(logs_dir) == "terminal_bench_unknown"


# ── _build_environment_snapshot ──────────────────────────────────────────

def test_environment_snapshot_includes_provenance_keys():
    timeout_info = {
        "cache_layout": "package",
        "resolution": "exact-ref",
        "selected_task_toml": "/fake/task.toml",
        "base_timeout_sec": 750.0,
        "multiplier": 15.0,
        "effective_timeout_sec": 10125.0,
    }
    snapshot = lca._build_environment_snapshot(
        "llamacpp/qwen3.6-35b-a3b",
        max_turns=0,
        thinking_level="high",
        ambient_max_turns_env="40",
        timeout_info=timeout_info,
    )
    assert snapshot["max_turns"] == 0
    assert snapshot["ambient_max_turns_env"] == "40"
    assert snapshot["little_coder_version"] == lca._AGENT_VERSION
    assert snapshot["code_sha"] == lca._CODE_SHA
    assert snapshot["adapter_file"].endswith("little_coder_agent.py")
    assert snapshot["timeout_provenance"] == timeout_info
    # capture_environment_snapshot()'s own keys are still present underneath.
    assert snapshot["model"] == "llamacpp/qwen3.6-35b-a3b"
    assert "max_tokens" in snapshot
    # The whole point of threading thinking_level through: the snapshot must
    # record the level this trial actually launches pi with, not the ambient
    # default. `"thinking" in snapshot` alone would pass even if the value
    # were dropped, because capture_environment_snapshot always emits the key.
    # Stable regardless of this machine's pi config -- cli_thinking wins over
    # every settings-file tier in _resolve_thinking.
    assert snapshot["thinking"]["resolved"] == "high"
    assert snapshot["thinking"]["source"] == "cli"


def test_environment_snapshot_is_json_serializable():
    """The adapter dumps this straight to JSON with `default=str` -- make
    sure that combination actually round-trips (guards against a future
    non-JSON-native value, e.g. a Path, sneaking into timeout_info or a
    capture_environment_snapshot() field, silently corrupting the file)."""
    timeout_info = lca._fallback_timeout_info()
    snapshot = lca._build_environment_snapshot(
        "llamacpp/qwen3.6-35b-a3b",
        max_turns=0,
        thinking_level="high",
        ambient_max_turns_env=None,
        timeout_info=timeout_info,
    )
    round_tripped = json.loads(json.dumps(snapshot, default=str))
    assert round_tripped["max_turns"] == 0
    assert round_tripped["timeout_provenance"] == timeout_info


# ── _confirm_live_thinking ──────────────────────────────────────────────


class _FakeRpc:
    def __init__(self, state=None, raises=None):
        self._state, self._raises = state or {}, raises

    def get_state(self):
        if self._raises:
            raise self._raises
        return self._state


class _FakeLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, msg):
        self.warnings.append(msg)


def _confirm(tmp_path, rpc, snapshot=None):
    snapshot = snapshot or {"thinking": {"resolved": "high", "confirmed_live": None}}
    path = tmp_path / "environment_snapshot.json"
    path.write_text(json.dumps(snapshot))
    logger = _FakeLogger()
    lca._confirm_live_thinking(rpc, snapshot, path, logger)
    return json.loads(path.read_text()), logger


def test_confirmed_live_records_the_level_pi_actually_resolved(tmp_path):
    """The clamp this exists to catch: pi degrades any level to "off" for a
    model registered reasoning=false, so a trial can run with no thinking at
    all under a snapshot whose `resolved` says "high"."""
    written, logger = _confirm(tmp_path, _FakeRpc({"thinkingLevel": "off"}))
    assert written["thinking"]["confirmed_live"] == "off"
    assert written["thinking"]["resolved"] == "high"  # the request, still recorded
    assert any("off" in w for w in logger.warnings)


def test_confirmed_live_agreeing_with_the_request_is_not_warned_about(tmp_path):
    written, logger = _confirm(tmp_path, _FakeRpc({"thinkingLevel": "high"}))
    assert written["thinking"]["confirmed_live"] == "high"
    assert logger.warnings == []


def test_a_failed_probe_is_recorded_rather_than_raised(tmp_path):
    """confirmed_live staying null must not be ambiguous between "pi was
    never asked" and "pi refused the request"."""
    written, _ = _confirm(tmp_path, _FakeRpc(raises=RuntimeError("pi rejected get_state")))
    assert written["thinking"]["confirmed_live"] is None
    assert written["errors"] == [
        {"source": "confirmed_thinking", "error": "RuntimeError: pi rejected get_state"}
    ]


def test_a_stateless_response_is_also_recorded(tmp_path):
    written, _ = _confirm(tmp_path, _FakeRpc({}))
    assert written["thinking"]["confirmed_live"] is None
    assert written["errors"][0]["source"] == "confirmed_thinking"
