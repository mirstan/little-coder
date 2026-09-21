"""Tests for rpc_client's thinking-level resolver.

The resolution cases below are ported from
.pi/extensions/benchmark-profiles/profiles.test.ts. The two implementations
are kept in sync by hand, so this file is the only thing that will catch a
change made to one side and not the other -- keep the shared cases matching
the TS ones when either moves.

Run with:
    python -m pytest benchmarks/test_thinking_level_resolver.py -v
"""
from __future__ import annotations

import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rpc_client as RC  # noqa: E402
from rpc_client import (  # noqa: E402
    DEFAULT_THINKING_LEVEL,
    REPO_ROOT,
    _norm_key,
    _resolve_profile_from,
    resolve_thinking_level,
)

SHIPPED = json.loads((REPO_ROOT / ".pi" / "settings.json").read_text())["little_coder"]

# A runtime id using ":" (e.g. "qwen3.6:35b-a3b") must still match a
# "-"-keyed profile -- ported from profiles.test.ts's colon-vs-hyphen fixture.
COLON_SETTINGS = {
    "default_model_profile": {"thinking_budget": 4096},
    "model_profiles": {
        "llamacpp/qwen3.6-35b-a3b": {"thinking_budget": 1234, "temperature": 0.3},
    },
}


# ── _resolve_profile_from vs resolveProfileFrom ─────────────────────────────


def test_resolves_base_profile():
    p = _resolve_profile_from(SHIPPED, "llamacpp/qwen3.6-35b-a3b")
    assert p["thinking_budget"] == 4096
    assert "context_limit" not in p
    assert "max_turns" not in p


def test_applies_terminal_bench_overrides():
    p = _resolve_profile_from(SHIPPED, "llamacpp/qwen3.6-35b-a3b", "terminal_bench")
    assert p["thinking_budget"] == 3000
    assert p["temperature"] == 0.2
    assert "max_turns" not in p
    assert "context_limit" not in p


def test_applies_gaia_overrides():
    p = _resolve_profile_from(SHIPPED, "llamacpp/qwen3.6-35b-a3b", "gaia")
    assert p["thinking_budget"] == 2000
    assert p["temperature"] == 0.4
    assert p["max_turns"] == 40
    assert p["context_limit"] == 65536


def test_unknown_model_falls_back_to_default_profile():
    p = _resolve_profile_from(SHIPPED, "fake-provider/fake-model")
    assert p["thinking_budget"] == 4096
    assert "context_limit" not in p


def test_unknown_benchmark_yields_base_profile_unchanged():
    p = _resolve_profile_from(SHIPPED, "llamacpp/qwen3.6-35b-a3b", "totally_made_up")
    assert p["thinking_budget"] == 4096
    assert "max_turns" not in p


def test_benchmark_overrides_key_never_leaks_into_the_result():
    for key in SHIPPED["model_profiles"]:
        assert "benchmark_overrides" not in _resolve_profile_from(SHIPPED, key)


# ── separator-insensitive matching (issue #8 quirk) ─────────────────────────


def test_norm_key_collapses_colon_to_hyphen():
    assert _norm_key("llamacpp/qwen3.6:35b-a3b") == "llamacpp/qwen3.6-35b-a3b"


def test_colon_runtime_id_matches_hyphenated_profile_key():
    p = _resolve_profile_from(COLON_SETTINGS, "llamacpp/qwen3.6:35b-a3b")
    assert p["thinking_budget"] == 1234


def test_exact_hyphenated_id_still_matches():
    p = _resolve_profile_from(COLON_SETTINGS, "llamacpp/qwen3.6-35b-a3b")
    assert p["thinking_budget"] == 1234


def test_prefix_match_when_runtime_id_has_a_tag_suffix():
    p = _resolve_profile_from(COLON_SETTINGS, "llamacpp/qwen3.6:35b-a3b:Q4_K_M")
    assert p["thinking_budget"] == 1234


def test_unrelated_model_falls_back_to_default():
    p = _resolve_profile_from(COLON_SETTINGS, "ollama/llama3")
    assert p["thinking_budget"] == 4096


# ── resolve_thinking_level ──────────────────────────────────────────────────


def test_every_shipped_profile_resolves_high():
    # No shipped profile names a thinking_level yet, so the whole fleet
    # falls through to default_model_profile's "high".
    for key in SHIPPED["model_profiles"]:
        assert resolve_thinking_level(key) == "high", key
    assert resolve_thinking_level("fake-provider/fake-model") == "high"


def test_a_matched_profile_does_not_inherit_the_default_profiles_level(monkeypatch):
    """Whole-profile fallback, mirroring resolveProfileFrom: a profile that
    matches but omits thinking_level falls to DEFAULT_THINKING_LEVEL, not to
    default_model_profile's value. Both are "high" as shipped, so this is a
    trap for a future edit of default_model_profile rather than a live bug."""
    settings = {
        "default_model_profile": {"thinking_level": "medium"},
        "model_profiles": {"omlx/m": {"thinking_budget": 4096}},
    }
    monkeypatch.setattr(RC, "_load_little_coder_settings", lambda: settings)
    assert resolve_thinking_level("omlx/m") == DEFAULT_THINKING_LEVEL
    assert resolve_thinking_level("omlx/unmatched") == "medium"


def test_shipped_default_profile_declares_the_level():
    assert SHIPPED["default_model_profile"]["thinking_level"] == "high"


def test_explicit_profile_level_wins(monkeypatch):
    settings = {
        "default_model_profile": {"thinking_level": "high"},
        "model_profiles": {"omlx/quiet": {"thinking_level": "off", "thinking_budget": 32768}},
    }
    monkeypatch.setattr(RC, "_load_little_coder_settings", lambda: settings)
    # "off" beats the profile's own nonzero thinking_budget: the budget caps
    # thinking that the level never turned on.
    assert resolve_thinking_level("omlx/quiet") == "off"
    assert resolve_thinking_level("omlx/other") == "high"


def test_benchmark_override_can_change_the_level(monkeypatch):
    settings = {
        "model_profiles": {
            "omlx/m": {
                "thinking_level": "high",
                "benchmark_overrides": {"gaia": {"thinking_level": "low"}},
            }
        }
    }
    monkeypatch.setattr(RC, "_load_little_coder_settings", lambda: settings)
    assert resolve_thinking_level("omlx/m") == "high"
    assert resolve_thinking_level("omlx/m", "gaia") == "low"
    assert resolve_thinking_level("omlx/m", "terminal_bench") == "high"


def test_missing_or_malformed_settings_fall_back_to_the_default(monkeypatch, tmp_path):
    missing = tmp_path / "nope.json"
    broken = tmp_path / "broken.json"
    broken.write_text("{ not json")
    for path in (missing, broken):
        monkeypatch.setattr(RC, "_PI_PROJECT_SETTINGS_PATH", path)
        monkeypatch.setattr(RC, "_PI_SETTINGS_PATH", path)
        assert resolve_thinking_level("omlx/anything") == DEFAULT_THINKING_LEVEL
    assert DEFAULT_THINKING_LEVEL == "high"


def test_non_string_thinking_level_falls_back(monkeypatch):
    settings = {"model_profiles": {"omlx/m": {"thinking_level": 3}}}
    monkeypatch.setattr(RC, "_load_little_coder_settings", lambda: settings)
    assert resolve_thinking_level("omlx/m") == DEFAULT_THINKING_LEVEL


def test_a_level_outside_pis_vocabulary_falls_back_and_warns(monkeypatch, capsys):
    """pi's cli/args.js demotes an unrecognized --thinking to a warning and
    DROPS the flag, so passing a typo through would run at pi's own default
    while the environment snapshot recorded the typo as source="cli".
    thinking_level is not schema-validated, so a typo is the realistic path."""
    settings = {"model_profiles": {"omlx/m": {"thinking_level": "hihg"}}}
    monkeypatch.setattr(RC, "_load_little_coder_settings", lambda: settings)
    assert resolve_thinking_level("omlx/m") == DEFAULT_THINKING_LEVEL
    assert "hihg" in capsys.readouterr().err


def test_the_resolver_accepts_every_level_pi_accepts(monkeypatch):
    for level in RC.PI_THINKING_LEVELS:
        settings = {"model_profiles": {"omlx/m": {"thinking_level": level}}}
        monkeypatch.setattr(RC, "_load_little_coder_settings", lambda: settings)
        assert resolve_thinking_level("omlx/m") == level


def test_the_vocabulary_matches_the_vendored_pi(monkeypatch):
    """PI_THINKING_LEVELS is a hand copy of pi's VALID_THINKING_LEVELS. pi is
    vendored here, so the copy can be checked against the real thing rather
    than trusted -- this fails on a pi bump that changes the vocabulary."""
    args_js = (
        RC.REPO_ROOT / "node_modules" / "@earendil-works" / "pi-coding-agent"
        / "dist" / "cli" / "args.js"
    )
    if not args_js.is_file():
        pytest.skip(f"vendored pi not installed at {args_js}")
    match = re.search(r"VALID_THINKING_LEVELS\s*=\s*\[([^\]]*)\]", args_js.read_text())
    assert match, "VALID_THINKING_LEVELS not found -- pi's args.js changed shape"
    assert set(re.findall(r'"([^"]+)"', match.group(1))) == set(RC.PI_THINKING_LEVELS)


def test_settings_load_prefers_the_project_file(monkeypatch, tmp_path):
    project = tmp_path / "project.json"
    home = tmp_path / "home.json"
    project.write_text(json.dumps({"little_coder": {"default_model_profile": {"thinking_level": "low"}}}))
    home.write_text(json.dumps({"little_coder": {"default_model_profile": {"thinking_level": "max"}}}))
    monkeypatch.setattr(RC, "_PI_PROJECT_SETTINGS_PATH", project)
    monkeypatch.setattr(RC, "_PI_SETTINGS_PATH", home)
    assert resolve_thinking_level("omlx/anything") == "low"

    # A project file without a little_coder block is skipped, not treated as
    # an empty one -- same as loadSettings()'s candidate loop.
    project.write_text(json.dumps({"quietStartup": True}))
    assert resolve_thinking_level("omlx/anything") == "max"
