"""_resolve_trial_timeout_sec() -- the Harbor task-cache glob lookup.

Requires the `harbor` package (only installed in harbor's own uv-tool venv,
not the plain system Python these other benchmark tests run under) purely
because little_coder_agent.py imports harbor.agents.base at module load time;
none of this file's actual assertions touch Harbor itself.
"""
import json
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("harbor")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "harbor_adapter"))
import little_coder_agent as lca  # noqa: E402


def _write_task_toml(path: Path, timeout_sec: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"[agent]\ntimeout_sec = {timeout_sec}\n")


def _write_trial_config(trial_dir: Path, task_name: str, multiplier: float = 3.0):
    """Legacy name@version trial config shape: task.path is a bare name."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "config.json").write_text(json.dumps({
        "timeout_multiplier": multiplier,
        "task": {"path": task_name},
    }))


def _write_trial_config_v21(trial_dir: Path, task_name: str, multiplier: float = 3.0):
    """Newer org/name package dataset trial config shape: task.name is
    namespaced (e.g. "terminal-bench/overfull-hbox"), and there is no "path"
    key at all -- confirmed against a real terminal-bench/terminal-bench-2-1
    run's on-disk config.json, not assumed."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "config.json").write_text(json.dumps({
        "timeout_multiplier": multiplier,
        "task": {"name": task_name, "ref": "sha256:deadbeef", "source": "terminal-bench/terminal-bench-2-1"},
    }))


def test_legacy_name_at_version_layout(tmp_path, monkeypatch):
    """<hash>/<task_name>/task.toml -- the name@version dataset cache shape."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(lca, "HARBOR_TASK_CACHE", cache)
    _write_task_toml(cache / "somehash123" / "my-task" / "task.toml", 900.0)

    trial_dir = tmp_path / "trial"
    _write_trial_config(trial_dir, "my-task")
    logs_dir = trial_dir / "agent"
    logs_dir.mkdir()

    assert lca._resolve_trial_timeout_sec(logs_dir) == pytest.approx(900.0 * 3.0 * 0.9)


def test_org_name_package_layout(tmp_path, monkeypatch):
    """packages/<org>/<task_name>/<content-hash>/task.toml -- the newer
    org/name package dataset cache shape (one level deeper than the legacy
    layout: task.toml sits under a further hash dir, not directly in
    <task_name>). Confirmed against a real `terminal-bench/terminal-bench-2-1`
    download, not assumed."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(lca, "HARBOR_TASK_CACHE", cache)
    _write_task_toml(
        cache / "packages" / "terminal-bench" / "my-task" / "contenthash456" / "task.toml",
        1200.0,
    )

    trial_dir = tmp_path / "trial"
    _write_trial_config(trial_dir, "my-task")
    logs_dir = trial_dir / "agent"
    logs_dir.mkdir()

    assert lca._resolve_trial_timeout_sec(logs_dir) == pytest.approx(1200.0 * 3.0 * 0.9)


def test_v21_config_shape_with_namespaced_task_name(tmp_path, monkeypatch):
    """Regression test: a real terminal-bench/terminal-bench-2-1 trial's
    config.json has task.name (namespaced), not task.path -- using the wrong
    key raised KeyError, silently swallowed by the function's own broad
    except-fallback, so every trial under a package dataset silently used
    DEFAULT_PROMPT_TIMEOUT_SEC instead of its real per-task budget. Confirmed
    directly: this caused overfull-hbox to be hard-killed by Harbor's own
    2250s enforcement while the agent's internal deadline tracking still
    thought it had 3600s left."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(lca, "HARBOR_TASK_CACHE", cache)
    _write_task_toml(
        cache / "packages" / "terminal-bench" / "overfull-hbox" / "contenthash789" / "task.toml",
        750.0,
    )

    trial_dir = tmp_path / "trial"
    _write_trial_config_v21(trial_dir, "terminal-bench/overfull-hbox")
    logs_dir = trial_dir / "agent"
    logs_dir.mkdir()

    assert lca._resolve_trial_timeout_sec(logs_dir) == pytest.approx(750.0 * 3.0 * 0.9)


def test_both_layouts_present_prefers_newest(tmp_path, monkeypatch):
    """When both an old and a new cache exist for the same task name (e.g.
    right after switching dataset versions), the most recently written
    task.toml wins over an arbitrary match."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(lca, "HARBOR_TASK_CACHE", cache)

    old_toml = cache / "oldhash" / "my-task" / "task.toml"
    _write_task_toml(old_toml, 900.0)
    time.sleep(0.05)
    new_toml = cache / "packages" / "terminal-bench" / "my-task" / "newhash" / "task.toml"
    _write_task_toml(new_toml, 1800.0)

    trial_dir = tmp_path / "trial"
    _write_trial_config(trial_dir, "my-task")
    logs_dir = trial_dir / "agent"
    logs_dir.mkdir()

    assert lca._resolve_trial_timeout_sec(logs_dir) == pytest.approx(1800.0 * 3.0 * 0.9)


def test_no_cache_match_falls_back_to_default(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(lca, "HARBOR_TASK_CACHE", cache)

    trial_dir = tmp_path / "trial"
    _write_trial_config(trial_dir, "nonexistent-task")
    logs_dir = trial_dir / "agent"
    logs_dir.mkdir()

    assert lca._resolve_trial_timeout_sec(logs_dir) == lca.DEFAULT_PROMPT_TIMEOUT_SEC


def test_none_logs_dir_falls_back_to_default():
    assert lca._resolve_trial_timeout_sec(None) == lca.DEFAULT_PROMPT_TIMEOUT_SEC
