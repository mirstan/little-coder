"""Plan 5 (config-provenance v2): _derive_benchmark_label() and
_build_environment_snapshot() in little_coder_agent.py.

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
    assert "thinking" in snapshot
    assert "max_tokens" in snapshot


def test_environment_snapshot_is_json_serializable():
    """The adapter dumps this straight to JSON with `default=str` -- make
    sure that combination actually round-trips (guards against a future
    non-JSON-native value, e.g. a Path, sneaking into timeout_info or a
    capture_environment_snapshot() field, silently corrupting the file)."""
    timeout_info = lca._fallback_timeout_info()
    snapshot = lca._build_environment_snapshot(
        "llamacpp/qwen3.6-35b-a3b",
        max_turns=0,
        ambient_max_turns_env=None,
        timeout_info=timeout_info,
    )
    round_tripped = json.loads(json.dumps(snapshot, default=str))
    assert round_tripped["max_turns"] == 0
    assert round_tripped["timeout_provenance"] == timeout_info
