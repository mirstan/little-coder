"""Tests for rpc_client.capture_environment_snapshot().

Every path constant is monkeypatched to a tmp_path fixture -- never the real
machine's ~/.omlx, ~/.pi, or ~/.config/little-coder, so these tests are
reproducible on any machine regardless of what's actually installed there.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rpc_client as RC  # noqa: E402


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture(autouse=True)
def _isolate_paths(tmp_path, monkeypatch):
    """Point every config source somewhere that doesn't exist yet by default --
    individual tests create only the files they need."""
    monkeypatch.setattr(RC, "_PI_SETTINGS_PATH", tmp_path / "pi-settings.json")
    monkeypatch.setattr(RC, "_LC_MODELS_SHIPPED_DEFAULT", tmp_path / "shipped-models.json")
    monkeypatch.setattr(RC, "_OMLX_SETTINGS", tmp_path / "omlx-settings.json")
    monkeypatch.setattr(RC, "_OMLX_MODEL_SETTINGS", tmp_path / "omlx-model-settings.json")
    monkeypatch.setattr(RC, "_VENDOR_PATCH_TARGET", tmp_path / "openai-completions.js")
    # The real repo .cache/pi-bench-agent otherwise leaks in: _resolve_thinking
    # resolves its settings path through this constant, so an unpatched one
    # would make these assertions depend on what a local benchmark run left
    # behind.
    monkeypatch.setattr(RC, "_BENCH_AGENT_DIR", tmp_path / "bench-agent")
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
    monkeypatch.delenv("LITTLE_CODER_MODELS_FILE", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


# ── thinking resolution ──────────────────────────────────────────────────
#
# These pin the settings file pi ITSELF reads (<agent dir>/settings.json),
# not ~/.pi/agent/settings.json: PiRpc redirects PI_CODING_AGENT_DIR, so the
# home-directory file is one pi never opens during a benchmark run.
# _PI_SETTINGS_PATH still exists, but only for _load_little_coder_settings's
# mirror of the TS extension, which really does resolve from $HOME.

def test_thinking_cli_value_wins_over_agent_dir_settings(tmp_path, monkeypatch):
    agent_dir = tmp_path / "exported-agent"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
    _write_json(agent_dir / "settings.json", {"defaultThinkingLevel": "high"})
    out = RC._resolve_thinking("low")
    assert out == {
        "cli_value": "low", "pi_default_setting": "high",
        "pi_default_setting_file": str(agent_dir / "settings.json"),
        "resolved": "low", "source": "cli", "confirmed_live": None,
    }


def test_thinking_falls_back_to_agent_dir_settings_when_cli_unset(tmp_path, monkeypatch):
    agent_dir = tmp_path / "exported-agent"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
    _write_json(agent_dir / "settings.json", {"defaultThinkingLevel": "high"})
    out = RC._resolve_thinking(None)
    assert out["resolved"] == "high"
    assert out["source"] == "pi_default_settings"


def test_thinking_home_settings_are_not_consulted(tmp_path, monkeypatch):
    """The pre-isolation source must not leak back in. A machine whose
    ~/.pi/agent/settings.json says "off" -- the exact config that motivated
    the redirect -- must not have that reported as this run's level."""
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "exported-agent"))
    _write_json(RC._PI_SETTINGS_PATH, {"defaultThinkingLevel": "off"})
    out = RC._resolve_thinking(None)
    assert out["pi_default_setting"] is None
    assert out["resolved"] == RC.PI_BUILTIN_THINKING_LEVEL


def test_thinking_falls_back_to_pis_builtin_when_no_setting_applies(tmp_path):
    out = RC._resolve_thinking(None)
    # Not None/"unresolved": pi resolves this case itself, to its compiled-in
    # "medium". Claiming ignorance let a run at "medium" be recorded as though
    # the level were unknowable.
    assert out["resolved"] == "medium" == RC.PI_BUILTIN_THINKING_LEVEL
    assert out["source"] == "pi_builtin_default"
    # Still distinguishable from a file that explicitly said "medium".
    assert out["pi_default_setting"] is None


def test_thinking_ignores_a_stale_latch_in_the_bench_agent_dir(tmp_path):
    """thinking-budget's latch persists setThinkingLevel("off") into the
    scratch dir. _bench_agent_dir() deletes that file on every PiRpc, so pi
    never reads it -- and neither may the snapshot, whichever side of that
    delete the snapshot happens to be built on."""
    _write_json(RC._BENCH_AGENT_DIR / "settings.json", {"defaultThinkingLevel": "off"})
    out = RC._resolve_thinking(None)
    assert out["pi_default_setting"] is None
    assert out["resolved"] == RC.PI_BUILTIN_THINKING_LEVEL


# ── max_tokens resolution ────────────────────────────────────────────────

def _models_file(model_id="tiel-coder-oq4e", max_tokens=32768, provider="omlx"):
    return {"providers": {provider: {"models": [{"id": model_id, "maxTokens": max_tokens}]}}}


def test_max_tokens_env_var_wins(tmp_path, monkeypatch):
    env_path = tmp_path / "env-models.json"
    _write_json(env_path, _models_file(max_tokens=111))
    monkeypatch.setenv("LITTLE_CODER_MODELS_FILE", str(env_path))
    out = RC._find_model_max_tokens("omlx", "tiel-coder-oq4e")
    assert out == {"value": 111, "source_file": str(env_path), "resolution": "env:LITTLE_CODER_MODELS_FILE"}


def test_max_tokens_xdg_config_home_used_when_no_env_override(tmp_path, monkeypatch):
    xdg = tmp_path / "xdg"
    _write_json(xdg / "little-coder" / "models.json", _models_file(max_tokens=222))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    out = RC._find_model_max_tokens("omlx", "tiel-coder-oq4e")
    assert out["value"] == 222
    assert out["resolution"] == "env:XDG_CONFIG_HOME"


def test_max_tokens_home_default_when_no_env_vars(tmp_path):
    _write_json(tmp_path / ".config" / "little-coder" / "models.json", _models_file(max_tokens=333))
    out = RC._find_model_max_tokens("omlx", "tiel-coder-oq4e")
    assert out["value"] == 333
    assert out["resolution"] == "home_default"


def test_max_tokens_falls_back_to_shipped_default_when_override_lacks_model(tmp_path):
    _write_json(tmp_path / ".config" / "little-coder" / "models.json", _models_file(model_id="some-other-model"))
    _write_json(RC._LC_MODELS_SHIPPED_DEFAULT, _models_file(max_tokens=444))
    out = RC._find_model_max_tokens("omlx", "tiel-coder-oq4e")
    assert out["value"] == 444
    assert out["resolution"] == "shipped_default"


def test_max_tokens_none_when_nowhere_defines_the_model(tmp_path):
    out = RC._find_model_max_tokens("omlx", "tiel-coder-oq4e")
    assert out == {"value": None, "source_file": None, "resolution": None}


# ── server_sampling (omlx) ───────────────────────────────────────────────

def test_server_sampling_omlx_merges_global_and_per_model(tmp_path):
    _write_json(RC._OMLX_SETTINGS, {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "repetition_penalty": 1.0}})
    _write_json(RC._OMLX_MODEL_SETTINGS, {"models": {"tiel-coder-oq4e": {"temperature": 0.6, "top_k": 20}}})
    errors = []
    out = RC._capture_omlx_sampling("tiel-coder-oq4e", errors)
    assert out["global_default"] == {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "repetition_penalty": 1.0}
    assert out["per_model_override"] == {"temperature": 0.6, "top_k": 20}
    assert out["effective"] == {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0}
    assert errors == []


def test_server_sampling_omlx_no_override_for_this_model(tmp_path):
    _write_json(RC._OMLX_SETTINGS, {"sampling": {"temperature": 1.0}})
    _write_json(RC._OMLX_MODEL_SETTINGS, {"models": {"some-other-model": {"temperature": 0.6}}})
    errors = []
    out = RC._capture_omlx_sampling("tiel-coder-oq4e", errors)
    assert out["global_default"] == {"temperature": 1.0}
    assert out["per_model_override"] is None
    assert out["effective"] == {"temperature": 1.0}


def test_server_sampling_omlx_files_missing_records_absent_not_error(tmp_path):
    errors = []
    out = RC._capture_omlx_sampling("tiel-coder-oq4e", errors)
    assert out["global_default"] is None
    assert out["per_model_override"] is None
    assert out["effective"] is None
    assert len(errors) == 2  # both files missing


def test_server_sampling_none_note_for_non_omlx_provider(tmp_path):
    snap = RC.capture_environment_snapshot("rapidmlx/nail-qwen")
    assert snap["server_sampling"] == {"note": "not introspectable for this provider", "provider": "rapidmlx"}
    # No omlx file I/O should have been attempted -- errors list stays empty
    # since server_sampling isn't even looked at for this provider.
    assert snap["errors"] == []


# ── vendor patch ─────────────────────────────────────────────────────────

def test_vendor_patch_detected_when_marker_present(tmp_path):
    RC._VENDOR_PATCH_TARGET.parent.mkdir(parents=True, exist_ok=True)
    RC._VENDOR_PATCH_TARGET.write_text("...\nif (process.env.PI_REASONING_MAX_TOKENS) { ... }\n")
    snap = RC.capture_environment_snapshot("omlx/tiel-coder-oq4e")
    assert snap["vendor_patch"]["exists"] is True
    assert snap["vendor_patch"]["applied"] is True


def test_vendor_patch_absent_when_marker_missing(tmp_path):
    RC._VENDOR_PATCH_TARGET.parent.mkdir(parents=True, exist_ok=True)
    RC._VENDOR_PATCH_TARGET.write_text("// nothing relevant here\n")
    snap = RC.capture_environment_snapshot("omlx/tiel-coder-oq4e")
    assert snap["vendor_patch"]["exists"] is True
    assert snap["vendor_patch"]["applied"] is False


def test_vendor_patch_absent_when_file_missing(tmp_path):
    snap = RC.capture_environment_snapshot("omlx/tiel-coder-oq4e")
    assert snap["vendor_patch"]["exists"] is False
    assert snap["vendor_patch"]["applied"] is False


# ── full best-effort contract ────────────────────────────────────────────

def test_snapshot_never_raises_when_everything_is_missing(tmp_path):
    snap = RC.capture_environment_snapshot("omlx/tiel-coder-oq4e")
    for key in ("model", "provider", "thinking", "max_tokens", "server_sampling", "vendor_patch", "errors"):
        assert key in snap
    assert snap["errors"], "expected at least one breadcrumb for the missing omlx files"


def test_snapshot_records_error_for_malformed_model_string(tmp_path):
    snap = RC.capture_environment_snapshot("bare-model-id-no-slash")
    assert any(e["source"] == "model" for e in snap["errors"])
    assert snap["server_sampling"]["note"] == "not introspectable for this provider"
