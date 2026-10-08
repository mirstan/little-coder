"""run_watch rules: every alert rule, config resolution and dedup, pure."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_watch as W  # noqa: E402

TRIAL = "fix-git__abc123"


def rec(turn, **kw):
    base = {"v": 1, "turn": turn, "ts_start": 1000.0 + turn * 10, "ts_end": 1005.0 + turn * 10,
            "iso_end": "", "headers_s": 0.5, "ttft_s": 1.0, "gen_s": 4.0, "turn_s": 5.0,
            "stop_reason": "toolUse", "error_message": None, "usage_reported": True,
            "input": 500, "cache_read": 0, "cache_write": 0, "output": 200, "reasoning": 0,
            "prompt_tokens": 500, "context_tokens": 700, "cache_hit": 0.0, "text_chars": 0,
            "thinking_chars": 0, "n_tool_calls": 1, "max_tool_arg_bytes": 100,
            "tool_calls": [{"name": "ShellSession", "arg_bytes": 100}], "tool_errors": 0,
            "retention": None, "demoted_new": None, "stubs": None, "stubbed_new": None,
            "guards": [], "compactions": [], "retries_before": 0}
    base.update(kw)
    return base


def retention(large, prefix, demoted, signed=0):
    return {"v": 1, "kind": "shell_retention", "pairs": large, "large": large, "due": prefix,
            "prefix": prefix, "demoted": demoted, "flushed": False, "signed": signed,
            "skippedNoShrink": prefix - demoted, "skippedArchive": 0, "bytesBefore": 0, "bytesAfter": 0}


def trial():
    return W.TrialState(name=TRIAL, start_ts=1000.0)


def rules(alerts):
    return [(a.level, a.rule) for a in alerts]


def test_config_defaults_env_and_flags():
    cfg = W.config_from_env({}, {})
    assert cfg == W.RuleConfig()
    assert (cfg.context_jump_tokens, cfg.max_output_tokens, cfg.no_demotion_large_pairs,
            cfg.no_demotion_turns, cfg.no_demotion_context_tokens, cfg.divergence_reprefill_tokens,
            cfg.divergence_gap_tokens, cfg.cache_min_prompt_tokens, cfg.cache_min_hit_ratio,
            cfg.ttft_max_s, cfg.stall_min, cfg.cooldown_s, cfg.telemetry_grace_turns) == (
        30000, 16000, 12, 40, 60000, 16384, 8192, 40000, 0.5, 120.0, 20.0, 600.0, 10)
    env = {"RUN_WATCH_CONTEXT_JUMP_TOKENS": "5000", "RUN_WATCH_TTFT_MAX_S": "90",
           "RUN_WATCH_MAX_OUTPUT_TOKENS": "lots"}
    cfg = W.config_from_env(env, {"ttft_max_s": 60.0, "stall_min": None})
    assert (cfg.context_jump_tokens, cfg.ttft_max_s, cfg.max_output_tokens, cfg.stall_min) == (5000, 60.0, 16000, 20.0)


def test_context_jump_fires_on_growth_and_resets_after_a_compaction():
    st, cfg = trial(), W.RuleConfig()
    assert W.evaluate_turn(st, rec(1, prompt_tokens=46_000), cfg) == []
    assert W.evaluate_turn(st, rec(2, prompt_tokens=60_000), cfg) == []
    jump = W.evaluate_turn(st, rec(3, prompt_tokens=125_000), cfg)
    assert rules(jump) == [("warn", "context_jump")]
    assert (jump[0].trial, jump[0].subject) == (TRIAL, "turn 3")
    assert "60,000 -> 125,000" in jump[0].message
    after = W.evaluate_turn(st, rec(4, prompt_tokens=200_000,
                                    compactions=[{"source": "harness", "reason": "manual", "ok": True}]), cfg)
    assert rules(after) == []
    assert st.compactions == 1


def test_a_turn_without_usage_is_not_a_baseline():
    st, cfg = trial(), W.RuleConfig()
    W.evaluate_turn(st, rec(1, prompt_tokens=46_000), cfg)
    W.evaluate_turn(st, rec(2, usage_reported=False, prompt_tokens=0, stop_reason="aborted"), cfg)
    assert rules(W.evaluate_turn(st, rec(3, prompt_tokens=50_000), cfg)) == []


def test_length_stop_big_output_and_ttft_spike():
    st, cfg = trial(), W.RuleConfig()
    got = W.evaluate_turn(st, rec(1, stop_reason="length", output=80_000, max_tool_arg_bytes=240_000, ttft_s=121.0), cfg)
    assert rules(got) == [("crit", "length_stop"), ("warn", "big_output"), ("warn", "ttft_spike")]
    assert rules(W.evaluate_turn(st, rec(2, output=16_000, ttft_s=120.0), cfg)) == []


def test_cache_collapse_needs_a_provider_that_reports_cache():
    st, cfg = trial(), W.RuleConfig()
    assert W.evaluate_turn(st, rec(1, prompt_tokens=50_000, cache_read=0, cache_hit=0.0), cfg) == []
    assert W.evaluate_turn(st, rec(2, prompt_tokens=50_000, cache_read=30_000, cache_hit=0.6), cfg) == []
    assert rules(W.evaluate_turn(st, rec(3, prompt_tokens=50_000, cache_read=15_000, cache_hit=0.3), cfg)) == [
        ("warn", "cache_collapse")]
    assert W.evaluate_turn(st, rec(4, prompt_tokens=30_000, cache_read=3_000, cache_hit=0.1), cfg) == []


def test_retention_stalled_and_signed_fire_once_per_trial():
    st, cfg = trial(), W.RuleConfig()
    got = W.evaluate_turn(st, rec(1, retention=retention(9, 4, 0, signed=4)), cfg)
    assert rules(got) == [("crit", "retention_stalled"), ("warn", "retention_signed")]
    assert got[0].subject == "once"
    assert W.evaluate_turn(st, rec(2, retention=retention(9, 4, 0, signed=4)), cfg) == []


def test_no_demotions_by_large_pairs_and_by_long_context():
    st, cfg = trial(), W.RuleConfig()
    assert W.evaluate_turn(st, rec(1, retention=retention(11, 0, 0)), cfg) == []
    assert rules(W.evaluate_turn(st, rec(2, retention=retention(12, 0, 0)), cfg)) == [("crit", "no_demotions")]
    assert W.evaluate_turn(st, rec(3, retention=retention(13, 0, 0)), cfg) == []

    st2 = trial()
    seen = []
    for n in range(1, 41):
        seen += [(n, r) for r in rules(W.evaluate_turn(st2, rec(n, prompt_tokens=61_000), cfg))]
    assert seen == [(10, ("info", "telemetry_missing")), (40, ("warn", "no_demotions"))]


def test_a_demotion_seen_once_silences_no_demotions():
    st, cfg = trial(), W.RuleConfig()
    W.evaluate_turn(st, rec(1, retention=retention(9, 4, 4)), cfg)
    assert W.evaluate_turn(st, rec(2, retention=retention(20, 0, 0)), cfg) == []


def test_guard_and_stub_telemetry_raise_info_alerts():
    st, cfg = trial(), W.RuleConfig()
    guard = {"v": 1, "kind": "guard_abort", "trigger": "toolcall_cap", "tool": "ShellSession",
             "argChars": 32001, "capChars": 32000}
    echo = {"v": 1, "kind": "echo_block", "source": "length_stub", "tool": "write"}
    got = W.evaluate_turn(st, rec(1, guards=[guard, echo], stubbed_new=1,
                                  stubs={"v": 1, "kind": "length_stub", "stubbed": 1}), cfg)
    assert rules(got) == [("info", "guard_abort"), ("info", "echo_block"), ("info", "stub_applied")]
    assert "toolcall_cap" in got[0].message and "argChars=32001" in got[0].message
    assert got[0].subject == "turn 1 guard_abort:toolcall_cap"
    assert st.telemetry_seen is True


def test_server_prefix_divergence_only_when_it_diverges_mid_history():
    cfg = W.RuleConfig()
    base = {"src": "omlx", "kind": "prefix_cache", "ts": 5000.0, "request": "r", "prompt": 71530, "reused": 31310}
    small = dict(base, reprefill=5220, shared=68144, comparable=69632)
    assert W.evaluate_server_event(small, TRIAL, cfg) == []
    # A big append after an ordinary tail divergence (1,488-token gap): not mid-history.
    big_append = dict(base, reprefill=40220, shared=68144, comparable=69632)
    assert W.evaluate_server_event(big_append, TRIAL, cfg) == []
    diverged = dict(base, reprefill=40220, shared=31310, comparable=69632)
    got = W.evaluate_server_event(diverged, TRIAL, cfg)
    assert rules(got) == [("warn", "prefix_divergence")]
    assert (got[0].trial, got[0].subject) == (TRIAL, "omlx 5000.000")
    assert "38,322 tokens before its end" in got[0].message
    deep_but_small = dict(base, reprefill=9000, shared=31310, comparable=69632)
    assert W.evaluate_server_event(deep_but_small, TRIAL, cfg) == []
    appended = dict(base, reprefill=40220, shared=69632, comparable=69632)
    assert W.evaluate_server_event(appended, TRIAL, cfg) == []


def test_a_failed_compaction_does_not_reset_the_jump_baseline():
    st, cfg = trial(), W.RuleConfig()
    W.evaluate_turn(st, rec(1, prompt_tokens=100_000), cfg)
    got = W.evaluate_turn(st, rec(2, prompt_tokens=165_000,
                                  compactions=[{"source": "harness", "reason": "manual", "ok": False}]), cfg)
    assert rules(got) == [("warn", "context_jump")]
    assert st.compactions == 0


def test_server_completion_rules_and_errors():
    cfg = W.RuleConfig()
    done = {"src": "splash", "kind": "completion", "ts": 6000.0, "status": "cancelled",
            "prompt": 50_000, "cached": 10_000, "output": 48_227, "ttft_s": 247.3, "tok_s": 30.0}
    assert rules(W.evaluate_server_event(done, None, cfg)) == [
        ("warn", "ttft_spike"), ("warn", "cache_collapse"), ("warn", "big_output")]
    ok = dict(done, cached=45_000, output=300, ttft_s=2.0)
    assert W.evaluate_server_event(ok, None, cfg) == []
    err = {"src": "splash", "kind": "server_error", "ts": 6001.0, "text": "not_found · GET /api/status"}
    assert rules(W.evaluate_server_event(err, TRIAL, cfg)) == [("info", "server_error")]
    assert W.evaluate_server_event({"kind": "completion"}, TRIAL, cfg) == []


def test_stall_says_hung_or_slow_and_ignores_finished_trials():
    cfg = W.RuleConfig()
    st = trial()
    hung = W.evaluate_stall(st, 1000.0 + 21 * 60, 1000.0, None, cfg)
    assert rules(hung) == [("warn", "stall")] and "hung" in hung[0].message
    slow = W.evaluate_stall(st, 1000.0 + 21 * 60, 1000.0, 1000.0 + 20 * 60, cfg)
    assert "slow" in slow[0].message
    assert W.evaluate_stall(st, 1000.0 + 19 * 60, 1000.0, None, cfg) == []
    st.finished = True
    assert W.evaluate_stall(st, 1000.0 + 60 * 60, 1000.0, None, cfg) == []


def alert(rule, ts, level="warn", subject=None):
    return W.Alert(ts=ts, level=level, rule=rule, trial=TRIAL, subject=subject or f"s{ts}", message="m")


def test_dedup_drops_repeats_rate_limits_warns_and_never_crits():
    d = W.DedupState()
    assert W.admit(d, alert("ttft_spike", 100.0, subject="turn 1"), 600.0) is True
    assert W.admit(d, alert("ttft_spike", 101.0, subject="turn 1"), 600.0) is False
    assert W.admit(d, alert("ttft_spike", 200.0), 600.0) is False
    late = alert("ttft_spike", 800.0)
    assert W.admit(d, late, 600.0) is True
    assert late.data["suppressed_before"] == 1
    assert W.admit(d, alert("length_stop", 100.0, level="crit"), 600.0) is True
    assert W.admit(d, alert("length_stop", 101.0, level="crit"), 600.0) is True


def test_seeded_dedup_refuses_an_alert_already_in_alerts_jsonl():
    d = W.DedupState()
    W.seed_dedup(d, [{"ts": 100.0, "rule": "length_stop", "trial": TRIAL, "subject": "turn 1"}, "garbage"])
    assert W.admit(d, alert("length_stop", 100.0, level="crit", subject="turn 1"), 600.0) is False
    assert W.admit(d, alert("length_stop", 120.0, level="crit", subject="turn 2"), 600.0) is True


def test_format_alert_and_row():
    a = W.Alert(ts=1000.0, level="crit", rule="length_stop", trial=TRIAL, subject="turn 3",
                message="response hit the output limit", data={"suppressed_before": 2})
    line = W.format_alert(a)
    assert line.startswith("[CRIT ")
    assert line.endswith("] fix-git length_stop: response hit the output limit (+2 similar suppressed)")
    row = W.alert_to_row(a)
    assert {k: row[k] for k in ("ts", "level", "rule", "trial", "subject", "message", "data")} == {
        "ts": 1000.0, "level": "crit", "rule": "length_stop", "trial": TRIAL, "subject": "turn 3",
        "message": "response hit the output limit", "data": {"suppressed_before": 2}}
    assert isinstance(row["iso"], str) and row["iso"]
