"""_resolve_token_usage() -- Plan 7's token-source selection + Harbor
AgentContext field mapping.

Requires the `harbor` package (only installed in harbor's own uv-tool venv,
not the plain system Python these other benchmark tests run under) purely
because little_coder_agent.py imports harbor.agents.base at module load time;
none of this file's actual assertions touch Harbor itself. Same pattern as
test_harbor_adapter_timeout.py.
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("harbor")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "harbor_adapter"))
import little_coder_agent as lca  # noqa: E402


EVENT_SUM_USAGE = {"input": 100, "output": 20, "cache_read": 10, "cache_write": 0, "cost": 0.001}


def test_prefers_session_stats_when_available():
    """session_stats is the complete source (includes tool-result usage and
    compaction tokens) -- it must win over the event-summed fallback even
    when both are populated."""
    stats = {
        "tokens": {"input": 500, "output": 100, "cacheRead": 50, "cacheWrite": 10, "total": 660},
        "cost": 0.45,
    }
    out = lca._resolve_token_usage(EVENT_SUM_USAGE, turn_count=1, stats=stats)
    assert out["token_source"] == "session_stats"
    # n_input_tokens: field doc says "including cache" -> input + both cache directions.
    assert out["n_input_tokens"] == 500 + 50 + 10
    assert out["n_cache_tokens"] == 50
    assert out["n_output_tokens"] == 100
    assert out["cost_usd"] == pytest.approx(0.45)
    assert out["raw"] == {"input": 500, "output": 100, "cache_read": 50, "cache_write": 10, "cost": 0.45}


def test_falls_back_to_event_sum_when_stats_unavailable():
    """stats=None (pi already dead, or an older pi without get_session_stats)
    -- must fall back to the event-summed PromptResult.usage."""
    out = lca._resolve_token_usage(EVENT_SUM_USAGE, turn_count=1, stats=None)
    assert out["token_source"] == "event_sum"
    assert out["n_input_tokens"] == 100 + 10 + 0
    assert out["n_cache_tokens"] == 10
    assert out["n_output_tokens"] == 20
    assert out["cost_usd"] == pytest.approx(0.001)


def test_unavailable_when_no_turn_ever_completed():
    """No turn_end ever fired (turn_count == 0) and stats is None -- there is
    nothing meaningful summed, so this is "unavailable", not a real
    (zero-valued) event sum."""
    zero_usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "cost": 0.0}
    out = lca._resolve_token_usage(zero_usage, turn_count=0, stats=None)
    assert out["token_source"] == "unavailable"
    assert out["n_input_tokens"] == 0
    assert out["cost_usd"] is None


def test_zero_cost_reads_as_none_not_free():
    """Local providers (llama.cpp, omlx) report cost 0 -- cost_usd must be
    None ("not priced"), matching how other Harbor agents report an unpriced
    run, rather than reading as a genuinely free run."""
    stats = {"tokens": {"input": 1000, "output": 200, "cacheRead": 0, "cacheWrite": 0}, "cost": 0}
    out = lca._resolve_token_usage(EVENT_SUM_USAGE, turn_count=1, stats=stats)
    assert out["cost_usd"] is None
    assert out["n_input_tokens"] == 1000


def test_session_stats_present_but_malformed_falls_back():
    """A `stats` dict that lacks a "tokens" object (unexpected shape from a
    future/older pi build) must not be trusted -- fall back to event_sum
    rather than crash or silently report zeros with a misleading
    token_source."""
    out = lca._resolve_token_usage(EVENT_SUM_USAGE, turn_count=1, stats={"cost": 0.1})
    assert out["token_source"] == "event_sum"
    assert out["n_output_tokens"] == 20
