"""_resolve_trial_timeout_sec() -- the Harbor task-cache glob lookup.

Requires the `harbor` package (only installed in harbor's own uv-tool venv,
not the plain system Python these other benchmark tests run under) purely
because little_coder_agent.py imports harbor.agents.base at module load time;
none of this file's actual assertions touch Harbor itself.
"""
import json
import os
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


def _write_trial_config_v21(trial_dir: Path, task_name: str, ref: str = "sha256:deadbeef", multiplier: float = 3.0):
    """Newer org/name package dataset trial config shape: task.name is
    namespaced (e.g. "terminal-bench/overfull-hbox"), and there is no "path"
    key at all -- confirmed against a real terminal-bench/terminal-bench-2-1
    run's on-disk config.json, not assumed. task.ref is a "sha256:<hex>"
    string that identifies the exact content-hash directory under the
    package cache."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "config.json").write_text(json.dumps({
        "timeout_multiplier": multiplier,
        "task": {"name": task_name, "ref": ref, "source": "terminal-bench/terminal-bench-2-1"},
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
    # No task.name in this config (legacy shape) -- exercised separately by
    # test_v21_config_shape_with_namespaced_task_name below. This test just
    # covers the on-disk package layout via a legacy-shaped config whose bare
    # name happens to only exist in the package cache, using the fallback
    # (ref-less) glob path.
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "config.json").write_text(json.dumps({
        "timeout_multiplier": 3.0,
        "task": {"name": "terminal-bench/my-task"},
    }))
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
    _write_trial_config_v21(trial_dir, "terminal-bench/overfull-hbox", ref="sha256:contenthash789")
    logs_dir = trial_dir / "agent"
    logs_dir.mkdir()

    assert lca._resolve_trial_timeout_sec(logs_dir) == pytest.approx(750.0 * 3.0 * 0.9)


def test_package_config_ignores_newer_legacy_file(tmp_path, monkeypatch):
    """Regression test for the cross-layout mtime bug: a package-shape
    config (task.name namespaced, task.ref a sha256 matching the package
    cache dir name) must resolve to the package task.toml even when a
    same-named legacy task.toml exists with a NEWER mtime. On the pre-fix
    code (mtime tie-break across a unioned glob of both layouts), the newer
    legacy file always won since every real package-cache task.toml has an
    epoch mtime (confirmed against ~/.cache/harbor/tasks) -- so this test
    forces the same epoch-vs-real-mtime skew here and asserts the PACKAGE
    value wins regardless."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(lca, "HARBOR_TASK_CACHE", cache)

    pkg_toml = cache / "packages" / "terminal-bench" / "caffe-cifar-10" / "abc123" / "task.toml"
    _write_task_toml(pkg_toml, 3600.0)
    os.utime(pkg_toml, (0, 0))  # real package caches restore no timestamp

    legacy_toml = cache / "somehash" / "caffe-cifar-10" / "task.toml"
    _write_task_toml(legacy_toml, 1200.0)
    # Give the legacy file a real, much-newer mtime (it's freshly written).
    assert legacy_toml.stat().st_mtime > 0

    trial_dir = tmp_path / "trial"
    _write_trial_config_v21(trial_dir, "terminal-bench/caffe-cifar-10", ref="sha256:abc123")
    logs_dir = trial_dir / "agent"
    logs_dir.mkdir()

    assert lca._resolve_trial_timeout_sec(logs_dir) == pytest.approx(3600.0 * 3.0 * 0.9)


def test_legacy_config_ignores_package_file(tmp_path, monkeypatch):
    """Mirror of the above: a legacy-shape config must resolve to the
    legacy task.toml even when a same-named package-layout file exists (and
    even if that package file happens to have a newer mtime)."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(lca, "HARBOR_TASK_CACHE", cache)

    legacy_toml = cache / "somehash" / "crack-7z-hash" / "task.toml"
    _write_task_toml(legacy_toml, 900.0)

    time.sleep(0.02)
    pkg_toml = cache / "packages" / "terminal-bench" / "crack-7z-hash" / "def456" / "task.toml"
    _write_task_toml(pkg_toml, 1800.0)
    assert pkg_toml.stat().st_mtime >= legacy_toml.stat().st_mtime

    trial_dir = tmp_path / "trial"
    _write_trial_config(trial_dir, "crack-7z-hash")
    logs_dir = trial_dir / "agent"
    logs_dir.mkdir()

    assert lca._resolve_trial_timeout_sec(logs_dir) == pytest.approx(900.0 * 3.0 * 0.9)


def test_package_ref_mismatch_falls_back_to_package_glob(tmp_path, monkeypatch):
    """task.ref points at a content-hash dir that doesn't exist (e.g. cache
    evicted and re-fetched under a different hash) -- must fall back to a
    glob restricted to the PACKAGE layout only, never the legacy one, even
    when a legacy file for the same task name also exists."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(lca, "HARBOR_TASK_CACHE", cache)

    other_pkg_toml = cache / "packages" / "terminal-bench" / "my-task" / "otherhash" / "task.toml"
    _write_task_toml(other_pkg_toml, 2400.0)

    legacy_toml = cache / "somehash" / "my-task" / "task.toml"
    _write_task_toml(legacy_toml, 900.0)

    trial_dir = tmp_path / "trial"
    _write_trial_config_v21(trial_dir, "terminal-bench/my-task", ref="sha256:nonexistenthash")
    logs_dir = trial_dir / "agent"
    logs_dir.mkdir()

    assert lca._resolve_trial_timeout_sec(logs_dir) == pytest.approx(2400.0 * 3.0 * 0.9)


def test_null_multiplier_falls_back_default(tmp_path, monkeypatch):
    """timeout_multiplier: null -> float(None) raises TypeError, caught by
    the broad except-fallback -- provenance dict must record
    fallback-default."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(lca, "HARBOR_TASK_CACHE", cache)
    _write_task_toml(cache / "somehash" / "my-task" / "task.toml", 900.0)

    trial_dir = tmp_path / "trial"
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "config.json").write_text(json.dumps({
        "timeout_multiplier": None,
        "task": {"path": "my-task"},
    }))
    logs_dir = trial_dir / "agent"
    logs_dir.mkdir()

    assert lca._resolve_trial_timeout_sec(logs_dir) == lca.DEFAULT_PROMPT_TIMEOUT_SEC
    info = lca._resolve_trial_timeout_info(logs_dir)
    assert info["resolution"] == "fallback-default"
    assert info["effective_timeout_sec"] == lca.DEFAULT_PROMPT_TIMEOUT_SEC


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


def test_provenance_dict_shapes(tmp_path, monkeypatch):
    """_resolve_trial_timeout_info exposes cache_layout/resolution/
    selected_task_toml for each of the exact-ref, glob-fallback, and legacy
    branches."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(lca, "HARBOR_TASK_CACHE", cache)

    # exact-ref (package)
    exact_toml = cache / "packages" / "terminal-bench" / "task-a" / "hashA" / "task.toml"
    _write_task_toml(exact_toml, 100.0)
    trial_a = tmp_path / "trial_a"
    _write_trial_config_v21(trial_a, "terminal-bench/task-a", ref="sha256:hashA")
    logs_a = trial_a / "agent"
    logs_a.mkdir()
    info_a = lca._resolve_trial_timeout_info(logs_a)
    assert info_a["cache_layout"] == "package"
    assert info_a["resolution"] == "exact-ref"
    assert info_a["selected_task_toml"] == str(exact_toml)

    # fallback-glob (package, ref missing)
    glob_toml = cache / "packages" / "terminal-bench" / "task-b" / "hashB" / "task.toml"
    _write_task_toml(glob_toml, 200.0)
    trial_b = tmp_path / "trial_b"
    trial_b.mkdir(parents=True, exist_ok=True)
    (trial_b / "config.json").write_text(json.dumps({
        "timeout_multiplier": 1.0,
        "task": {"name": "terminal-bench/task-b"},
    }))
    logs_b = trial_b / "agent"
    logs_b.mkdir()
    info_b = lca._resolve_trial_timeout_info(logs_b)
    assert info_b["cache_layout"] == "package"
    assert info_b["resolution"] == "fallback-glob"
    assert info_b["selected_task_toml"] == str(glob_toml)

    # legacy glob
    legacy_toml = cache / "somehash" / "task-c" / "task.toml"
    _write_task_toml(legacy_toml, 300.0)
    trial_c = tmp_path / "trial_c"
    _write_trial_config(trial_c, "task-c")
    logs_c = trial_c / "agent"
    logs_c.mkdir()
    info_c = lca._resolve_trial_timeout_info(logs_c)
    assert info_c["cache_layout"] == "legacy"
    assert info_c["resolution"] == "glob"
    assert info_c["selected_task_toml"] == str(legacy_toml)

    # fallback-default
    trial_d = tmp_path / "trial_d"
    _write_trial_config(trial_d, "nonexistent")
    logs_d = trial_d / "agent"
    logs_d.mkdir()
    info_d = lca._resolve_trial_timeout_info(logs_d)
    assert info_d["resolution"] == "fallback-default"
    assert info_d["cache_layout"] is None
    assert info_d["selected_task_toml"] is None


def test_bare_name_with_sha256_ref_resolves_via_exact_ref_glob(tmp_path, monkeypatch):
    """Regression test: a package-shape config whose task.name has NO
    "<org>/" prefix (e.g. "overfull-hbox") makes org.rpartition("/") yield
    org="*", which used to skip the exact-ref fast path entirely (gated on
    `org != "*"`) even though task.ref's content hash already identifies a
    real, unambiguous directory. Must resolve via the new wildcard-org
    exact-hash glob to that file's timeout_sec -- not the default, and not
    some other same-named org's file."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(lca, "HARBOR_TASK_CACHE", cache)

    real_toml = cache / "packages" / "orgA" / "overfull-hbox" / "realhash" / "task.toml"
    _write_task_toml(real_toml, 1500.0)
    # A decoy under a different org with the same bare name but a different
    # hash -- must NOT be picked.
    decoy_toml = cache / "packages" / "orgB" / "overfull-hbox" / "decoyhash" / "task.toml"
    _write_task_toml(decoy_toml, 60.0)

    trial_dir = tmp_path / "trial"
    _write_trial_config_v21(trial_dir, "overfull-hbox", ref="sha256:realhash")
    logs_dir = trial_dir / "agent"
    logs_dir.mkdir()

    info = lca._resolve_trial_timeout_info(logs_dir)
    assert info["resolution"] == "exact-ref-glob"
    assert info["selected_task_toml"] == str(real_toml)
    assert info["base_timeout_sec"] == pytest.approx(1500.0)
    assert info["ambiguous"] is False
    assert info["candidates_considered"] == 1
    assert lca._resolve_trial_timeout_sec(logs_dir) == pytest.approx(1500.0 * 3.0 * 0.9)


def test_bare_name_two_orgs_same_hash_equal_timeouts_agreed(tmp_path, monkeypatch):
    """Two different orgs happen to share both the bare name and the exact
    content hash (e.g. a task mirrored under two org namespaces). Since
    both candidates agree on timeout_sec, resolve to it and report
    ambiguous=False, with resolution suffixed "-agreed"."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(lca, "HARBOR_TASK_CACHE", cache)

    toml_a = cache / "packages" / "orgA" / "shared-task" / "samehash" / "task.toml"
    _write_task_toml(toml_a, 500.0)
    toml_b = cache / "packages" / "orgB" / "shared-task" / "samehash" / "task.toml"
    _write_task_toml(toml_b, 500.0)

    trial_dir = tmp_path / "trial"
    _write_trial_config_v21(trial_dir, "shared-task", ref="sha256:samehash")
    logs_dir = trial_dir / "agent"
    logs_dir.mkdir()

    info = lca._resolve_trial_timeout_info(logs_dir)
    assert info["resolution"] == "exact-ref-glob-agreed"
    assert info["ambiguous"] is False
    assert info["candidates_considered"] == 2
    assert info["base_timeout_sec"] == pytest.approx(500.0)
    assert "candidate_task_tomls" not in info
    assert "candidate_timeouts_sec" not in info


def test_bare_name_two_orgs_same_hash_different_timeouts_ambiguous(tmp_path, monkeypatch):
    """Two orgs share bare name + exact content hash but their task.toml
    files disagree on timeout_sec (a malformed/inconsistent cache). Must
    pick the minimum (an under-estimate merely finalizes early; an
    over-estimate risks a hard-kill mid-write) and flag the ambiguity in the
    provenance dict so it's visible in result.json."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(lca, "HARBOR_TASK_CACHE", cache)

    toml_a = cache / "packages" / "orgA" / "shared-task" / "samehash" / "task.toml"
    _write_task_toml(toml_a, 500.0)
    toml_b = cache / "packages" / "orgB" / "shared-task" / "samehash" / "task.toml"
    _write_task_toml(toml_b, 200.0)

    trial_dir = tmp_path / "trial"
    _write_trial_config_v21(trial_dir, "shared-task", ref="sha256:samehash")
    logs_dir = trial_dir / "agent"
    logs_dir.mkdir()

    info = lca._resolve_trial_timeout_info(logs_dir)
    assert info["resolution"] == "exact-ref-glob"
    assert info["ambiguous"] is True
    assert info["candidates_considered"] == 2
    assert info["base_timeout_sec"] == pytest.approx(200.0)
    assert sorted(info["candidate_task_tomls"]) == sorted([str(toml_a), str(toml_b)])
    assert sorted(info["candidate_timeouts_sec"]) == [200.0, 500.0]
    assert lca._resolve_trial_timeout_sec(logs_dir) == pytest.approx(200.0 * 3.0 * 0.9)
    # Round-trips through JSON cleanly (PR27 depends on this).
    assert json.loads(json.dumps(info)) == info


def test_namespaced_name_still_takes_exact_ref_fast_path(tmp_path, monkeypatch):
    """Guard against the new wildcard-org glob shadowing the original
    exact-ref fast path: a namespaced task.name with a valid sha256 ref
    whose exact org+hash directory exists must still resolve via plain
    "exact-ref", not "exact-ref-glob", even when another org's directory
    with the same bare name and hash also exists on disk."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(lca, "HARBOR_TASK_CACHE", cache)

    real_toml = cache / "packages" / "terminal-bench" / "some-task" / "hashX" / "task.toml"
    _write_task_toml(real_toml, 1234.0)
    # Another org, same bare name and hash -- exact-ref must still win
    # without ever consulting the wildcard-org glob.
    other_org_toml = cache / "packages" / "otherorg" / "some-task" / "hashX" / "task.toml"
    _write_task_toml(other_org_toml, 9999.0)

    trial_dir = tmp_path / "trial"
    _write_trial_config_v21(trial_dir, "terminal-bench/some-task", ref="sha256:hashX")
    logs_dir = trial_dir / "agent"
    logs_dir.mkdir()

    info = lca._resolve_trial_timeout_info(logs_dir)
    assert info["resolution"] == "exact-ref"
    assert info["selected_task_toml"] == str(real_toml)
    assert info["ambiguous"] is False
    assert info["candidates_considered"] == 1
    assert lca._resolve_trial_timeout_sec(logs_dir) == pytest.approx(1234.0 * 3.0 * 0.9)
