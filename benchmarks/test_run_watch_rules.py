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


def test_long_context_alone_does_not_page_when_too_few_pairs_were_ever_due():
    # The live adaptive-rejection-sampler trial: 40+ turns over 60K tokens with
    # one large pair out of 58, so the batch could never fill.
    st, cfg = trial(), W.RuleConfig()
    assert cfg.no_demotion_min_due == 4
    seen = []
    for n in range(1, 61):
        r = retention(1 if n < 50 else 3, 0, 0)
        r["due"] = 1 if n < 50 else 3
        seen += rules(W.evaluate_turn(st, rec(n, prompt_tokens=120_000, retention=r), cfg))
    assert seen == []

    st2 = trial()
    seen = []
    for n in range(1, 41):
        r = retention(6, 0, 0)
        r["due"] = 4
        seen += [(n, x) for x in rules(W.evaluate_turn(st2, rec(n, prompt_tokens=61_000, retention=r), cfg))]
    # prefix 0 with due 4 is not "stalled" (that needs prefix >= 1); the long
    # context with a full batch due and nothing demoted is still worth a warning.
    assert seen == [(40, ("warn", "no_demotions"))]


def test_long_context_still_warns_when_only_other_extensions_report_telemetry():
    # length-truncation-stub reports on every request; shell-retention silent
    # (crashed, not loaded) must still page as it did before the cost gate.
    st, cfg = trial(), W.RuleConfig()
    seen = []
    for n in range(1, 61):
        seen += [(n, x) for x in rules(W.evaluate_turn(
            st, rec(n, prompt_tokens=120_000, stubs={"v": 1, "kind": "length_stub", "stubbed": 0}), cfg))]
    assert seen == [(40, ("warn", "no_demotions"))]


def gated(large, due, skipped, est_s=3000, est_r=140_000, ctx=184_000, reason=None, est_secs=None):
    r = retention(large, due - skipped, 0)
    r.update({"due": due, "skippedNoShrink": 0, "gate": "deferred" if skipped else "none",
              "gateReason": reason, "skippedCost": skipped, "estSaveTokens": est_s,
              "estReprefillTokens": est_r, "estContextTokens": ctx})
    if est_secs is not None:
        r.update({"estReprefillSeconds": est_secs, "ceilingSeconds": 60})
    return r


def opened(reason="ratio", est_secs=400.0):
    r = retention(8, 4, 4)
    r.update({"due": 4, "skippedNoShrink": 0, "gate": "open", "gateReason": reason, "skippedCost": 0,
              "estSaveTokens": 9000, "estReprefillTokens": 80_000, "estContextTokens": 150_000,
              "estReprefillSeconds": est_secs, "ceilingSeconds": 60, "estReprefillFromToken": 36_864})
    return r


def test_a_cost_deferral_explains_no_demotions_and_is_reported_once():
    st, cfg = trial(), W.RuleConfig()
    got = W.evaluate_turn(st, rec(1, prompt_tokens=184_000, retention=gated(13, 8, 8)), cfg)
    assert rules(got) == [("info", "demotion_deferred")]
    assert got[0].data == {"skippedCost": 8, "estSaveTokens": 3000, "estReprefillTokens": 140_000,
                           "estContextTokens": 184_000, "gateReason": None, "estReprefillSeconds": 0.0}
    assert "3,000" in got[0].message and "140,000" in got[0].message
    assert "save ratio" in got[0].message
    seen = []
    for n in range(2, 60):
        seen += rules(W.evaluate_turn(st, rec(n, prompt_tokens=184_000, retention=gated(13, 8, 8)), cfg))
    assert seen == []


def test_a_compaction_reached_with_every_due_pair_still_deferred_is_noted_once():
    st, cfg = trial(), W.RuleConfig()
    W.evaluate_turn(st, rec(1, prompt_tokens=200_000, retention=gated(13, 8, 8)), cfg)
    comp = [{"source": "harness", "reason": "manual", "ok": True, "tokens_before": 221_000, "tokens_after": 30_000}]
    got = W.evaluate_turn(st, rec(2, prompt_tokens=30_000, compactions=comp, retention=gated(2, 0, 0)), cfg)
    assert rules(got) == [("info", "deferred_into_compaction")]
    assert W.evaluate_turn(st, rec(3, prompt_tokens=31_000, compactions=comp, retention=gated(2, 0, 0)), cfg) == []

    # A demotion before the compaction means the gate did open: nothing to say.
    st2 = trial()
    W.evaluate_turn(st2, rec(1, prompt_tokens=200_000, retention=gated(13, 8, 4)), cfg)
    W.evaluate_turn(st2, rec(2, prompt_tokens=205_000, retention={**retention(13, 8, 8), "skippedCost": 0}), cfg)
    assert "deferred_into_compaction" not in rules(
        W.evaluate_turn(st2, rec(3, prompt_tokens=30_000, compactions=comp), cfg))


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


# ── prefix-divergence classification against the turn ledger ──────────────

def div_event(ts, prompt, shared, comparable, reprefill):
    return {"src": "omlx", "kind": "prefix_cache", "ts": ts, "request": "r", "prompt": prompt,
            "reused": shared, "reprefill": reprefill, "shared": shared, "comparable": comparable}


def ledger(*recs):
    """Fold turn records into a TrialState the way the watcher does."""
    st, cfg = trial(), W.RuleConfig()
    for r in recs:
        W.evaluate_turn(st, r, cfg)
    return st


T1 = dict(ts_start=1000.0, ts_end=1100.0, prompt_tokens=50_000, output=900)


def test_a_divergence_on_a_turn_that_demoted_is_expected_not_an_alert():
    st = ledger(rec(1, **T1),
                rec(2, ts_start=1200.0, ts_end=1300.0, prompt_tokens=70_000, demoted_new=4,
                    retention=retention(12, 8, 8)))
    got = W.classify_divergence(div_event(1201.5, 70_000, 30_000, 60_000, 40_000), st, W.RuleConfig(), now=1400.0)
    assert rules(got) == [("info", "expected_divergence")]
    assert got[0].data["explained_by"] == ["demotion"] and got[0].data["turn"] == 2
    assert got[0].subject == "omlx 1201.500"


def test_a_divergence_after_a_compaction_or_a_stub_is_expected():
    for extra, why in (({"compactions": [{"source": "pi", "reason": "threshold", "ok": True}]}, "compaction"),
                       ({"stubbed_new": 1, "stubs": {"v": 1, "kind": "length_stub", "stubbed": 1}}, "stub")):
        st = ledger(rec(1, **T1), rec(2, ts_start=1200.0, ts_end=1300.0, prompt_tokens=40_000, **extra))
        got = W.classify_divergence(div_event(1202.0, 40_000, 5_000, 48_000, 35_000), st, W.RuleConfig(), now=1400.0)
        assert rules(got) == [("info", "expected_divergence")], why
        assert got[0].data["explained_by"] == [why]


def test_a_failed_compaction_explains_nothing():
    st = ledger(rec(1, **T1), rec(2, ts_start=1200.0, ts_end=1300.0, prompt_tokens=70_000,
                                  compactions=[{"source": "harness", "reason": "manual", "ok": False}]))
    got = W.classify_divergence(div_event(1201.0, 70_000, 30_000, 60_000, 40_000), st, W.RuleConfig(), now=1400.0)
    assert rules(got) == [("warn", "prefix_divergence")]


def test_an_unexplained_mid_history_divergence_stays_a_warning():
    st = ledger(rec(1, **T1), rec(2, ts_start=1200.0, ts_end=1300.0, prompt_tokens=70_000, demoted_new=0))
    got = W.classify_divergence(div_event(1201.0, 70_000, 30_000, 60_000, 40_000), st, W.RuleConfig(), now=1400.0)
    assert rules(got) == [("warn", "prefix_divergence")]
    assert got[0].data["turn"] == 2 and "30,000 tokens before its end" in got[0].message


def test_a_divergence_inside_the_previous_output_gets_its_own_label():
    # Turn 1 prompted 50,000 and generated 20,000; omlx re-tokenized that output
    # (jundot/omlx#4353), so turn 2's prompt shares 55,000 of the stored 70,000.
    st = ledger(rec(1, ts_start=1000.0, ts_end=1100.0, prompt_tokens=50_000, output=20_000),
                rec(2, ts_start=1200.0, ts_end=1300.0, prompt_tokens=71_000))
    got = W.classify_divergence(div_event(1201.0, 71_000, 55_000, 70_000, 16_500), st, W.RuleConfig(), now=1400.0)
    assert rules(got) == [("info", "divergence_in_output")]
    assert "previous output" in got[0].message and "omlx#4353" in got[0].message


def test_a_divergence_waits_for_its_turn_record_then_times_out_as_unmatched():
    st = ledger(rec(1, **T1))
    ev = div_event(1201.0, 70_000, 30_000, 60_000, 40_000)
    cfg = W.RuleConfig()
    assert W.classify_divergence(ev, st, cfg, now=1300.0) is None  # turn 2 still running
    late = W.classify_divergence(ev, st, cfg, now=1201.0 + cfg.divergence_match_wait_s + 1)
    assert rules(late) == [("warn", "prefix_divergence")] and "no matching turn" in late[0].message
    # A later turn already recorded means the window has passed: decide now.
    st2 = ledger(rec(1, **T1), rec(3, ts_start=1500.0, ts_end=1600.0, prompt_tokens=90_000))
    assert rules(W.classify_divergence(ev, st2, cfg, now=1600.0)) == [("warn", "prefix_divergence")]


def test_matching_needs_the_prompt_count_not_just_the_time_window():
    st = ledger(rec(1, **T1), rec(2, ts_start=1200.0, ts_end=1300.0, prompt_tokens=70_000, demoted_new=4))
    # Same window, different prompt size: another request (e.g. a subagent), not turn 2.
    assert W.classify_divergence(div_event(1201.0, 33_000, 1_000, 30_000, 32_000), st, W.RuleConfig(), now=1250.0) is None


ABORTED = dict(ts_start=1200.0, ts_end=1300.0, usage_reported=False, prompt_tokens=0, input=0, output=0,
               stop_reason="aborted")


def test_an_aborted_turn_without_usage_matches_on_its_window_alone():
    st = ledger(rec(1, **T1), rec(2, demoted_new=4, retention=retention(12, 8, 8), **ABORTED),
                rec(3, ts_start=1400.0, ts_end=1500.0, prompt_tokens=90_000))
    # now is inside the wait: decided because a later turn has started, not by timeout.
    got = W.classify_divergence(div_event(1201.0, 70_000, 30_000, 60_000, 40_000), st, W.RuleConfig(), now=1501.0)
    assert rules(got) == [("info", "expected_divergence")]
    assert got[0].data["turn"] == 2 and got[0].data["explained_by"] == ["demotion"]


def test_an_unexplained_divergence_on_an_aborted_turn_names_that_turn():
    st = ledger(rec(1, **T1), rec(2, **ABORTED), rec(3, ts_start=1400.0, ts_end=1500.0, prompt_tokens=90_000))
    got = W.classify_divergence(div_event(1201.0, 70_000, 30_000, 60_000, 40_000), st, W.RuleConfig(), now=1501.0)
    assert rules(got) == [("warn", "prefix_divergence")]
    assert got[0].data["turn"] == 2 and "no matching turn record" not in got[0].message


def test_a_prompt_size_match_beats_an_aborted_turn_in_the_same_window():
    st = ledger(rec(1, **T1), rec(2, demoted_new=4, **ABORTED),
                rec(3, ts_start=1250.0, ts_end=1300.0, prompt_tokens=70_000))
    got = W.classify_divergence(div_event(1260.0, 70_000, 30_000, 60_000, 40_000), st, W.RuleConfig(), now=1301.0)
    assert rules(got) == [("warn", "prefix_divergence")] and got[0].data["turn"] == 3


def test_the_in_output_check_skips_an_aborted_turn_for_the_previous_output():
    st = ledger(rec(1, ts_start=1000.0, ts_end=1100.0, prompt_tokens=50_000, output=20_000),
                rec(2, ts_start=1110.0, ts_end=1150.0, usage_reported=False, prompt_tokens=0, output=0,
                    stop_reason="aborted"),
                rec(3, ts_start=1200.0, ts_end=1300.0, prompt_tokens=71_000))
    got = W.classify_divergence(div_event(1201.0, 71_000, 55_000, 70_000, 16_500), st, W.RuleConfig(), now=1400.0)
    assert rules(got) == [("info", "divergence_in_output")] and got[0].data["turn"] == 3


def test_a_retry_divergence_in_an_aborted_turns_slack_waits_for_the_retry_record():
    # pi retries 2 s after turn 2 aborts; omlx logs the retry's prefix line before
    # its record exists, inside turn 2's trailing slack.
    st, cfg = ledger(rec(1, **T1), rec(2, **ABORTED)), W.RuleConfig()
    ev = div_event(1302.0, 70_000, 30_000, 60_000, 40_000)
    assert W.classify_divergence(ev, st, cfg, now=1310.0) is None
    W.evaluate_turn(st, rec(3, ts_start=1301.0, ts_end=1400.0, prompt_tokens=70_000, demoted_new=4), cfg)
    got = W.classify_divergence(ev, st, cfg, now=1401.0)
    assert rules(got) == [("info", "expected_divergence")] and got[0].data["turn"] == 3


def test_a_divergence_that_belongs_to_the_aborted_turn_matches_it_once_the_next_turn_lands():
    st, cfg = ledger(rec(1, **T1), rec(2, **ABORTED)), W.RuleConfig()
    ev = div_event(1201.0, 70_000, 30_000, 60_000, 40_000)
    assert W.classify_divergence(ev, st, cfg, now=1310.0) is None
    W.evaluate_turn(st, rec(3, ts_start=1303.0, ts_end=1400.0, prompt_tokens=90_000), cfg)
    got = W.classify_divergence(ev, st, cfg, now=1401.0)
    assert rules(got) == [("warn", "prefix_divergence")] and got[0].data["turn"] == 2
    assert "no matching turn record" not in got[0].message


def test_divergence_counts_by_class_are_kept_on_the_trial():
    st = ledger(rec(1, **T1), rec(2, ts_start=1200.0, ts_end=1300.0, prompt_tokens=70_000, demoted_new=4))
    W.classify_divergence(div_event(1201.0, 70_000, 30_000, 60_000, 40_000), st, W.RuleConfig(), now=1400.0)
    assert st.divergences == {"expected_divergence": 1}


def test_a_ceiling_deferral_names_its_reason_and_each_reason_is_reported_once():
    st, cfg = trial(), W.RuleConfig()
    got = W.evaluate_turn(st, rec(1, prompt_tokens=200_000,
                                  retention=gated(13, 8, 8, est_s=11_044, ctx=124_103, reason="ceiling", est_secs=357.2)), cfg)
    assert rules(got) == [("info", "demotion_deferred")]
    assert got[0].data["gateReason"] == "ceiling" and got[0].data["estReprefillSeconds"] == 357.2
    assert "~357 s" in got[0].message and "60 s ceiling" in got[0].message
    assert rules(W.evaluate_turn(st, rec(2, prompt_tokens=201_000,
                                         retention=gated(13, 8, 8, reason="ceiling", est_secs=360.0)), cfg)) == []
    near = W.evaluate_turn(st, rec(3, prompt_tokens=205_000,
                                   retention=gated(13, 8, 8, ctx=204_965, reason="near", est_secs=662.0)), cfg)
    assert rules(near) == [("info", "demotion_deferred")]
    assert "near compaction" in near[0].message


def test_a_spike_the_gate_paid_for_and_predicted_is_info_not_warn():
    st, cfg = trial(), W.RuleConfig()
    W.evaluate_turn(st, rec(1, prompt_tokens=148_000, cache_read=145_000, cache_hit=0.98), cfg)
    got = W.evaluate_turn(st, rec(2, prompt_tokens=150_000, cache_read=40_000, cache_hit=0.27, ttft_s=450.0,
                                  retention=opened(est_secs=400.0)), cfg)
    assert rules(got) == [("info", "cache_collapse"), ("info", "ttft_spike")]
    assert got[1].data["predicted_s"] == 400.0
    # Slower than the gate predicted: a surprise, so it still warns.
    got = W.evaluate_turn(st, rec(3, prompt_tokens=150_000, cache_read=40_000, cache_hit=0.27, ttft_s=700.0,
                                  retention=opened(est_secs=400.0)), cfg)
    assert rules(got) == [("warn", "cache_collapse"), ("warn", "ttft_spike")]
    # A cold-cache open predicts nothing; neither does a turn without a gate estimate.
    for ret in (opened(reason="cold", est_secs=400.0), retention(8, 4, 4)):
        got = W.evaluate_turn(st, rec(4, prompt_tokens=150_000, cache_read=40_000, cache_hit=0.27, ttft_s=300.0,
                                      retention=ret), cfg)
        assert rules(got) == [("warn", "cache_collapse"), ("warn", "ttft_spike")]
    # gcode-to-text 97: the server kept 14K where the gate planned a break at ~37K.
    got = W.evaluate_turn(st, rec(5, prompt_tokens=150_000, cache_read=14_121, cache_hit=0.09, ttft_s=450.0,
                                  retention=opened(est_secs=400.0)), cfg)
    assert rules(got) == [("warn", "cache_collapse"), ("warn", "ttft_spike")]
    # Snapshots from before the ceiling carry no estimate: never downgraded.
    old = opened(est_secs=400.0)
    for k in ("estReprefillSeconds", "ceilingSeconds", "estReprefillFromToken"):
        del old[k]
    got = W.evaluate_turn(st, rec(6, prompt_tokens=150_000, cache_read=40_000, cache_hit=0.27, ttft_s=300.0,
                                  retention=old), cfg)
    assert rules(got) == [("warn", "cache_collapse"), ("warn", "ttft_spike")]
