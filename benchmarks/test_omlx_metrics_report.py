"""Tests for omlx_metrics_report.

Everything runs against a synthetic trial directory and a hand-built server.log:
the real ~/.omlx/logs/server.log rotates daily and changes under us, so it is
never read here. The synthetic log lines are copied from real omlx output so the
regexes are tested against the format they actually have to survive.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import omlx_metrics_report as R  # noqa: E402


def _local(day: str, clock: str) -> datetime:
    """The UTC instant a server.log stamp of `day clock` refers to on this host."""
    return datetime.strptime(f"{day} {clock}", "%Y-%m-%d %H:%M:%S").astimezone(timezone.utc)


def _chat(clock, out_tokens, gen_s, tok_s, prompt, finish="tool_calls", day="2026-09-13"):
    return (
        f"{day} {clock},000 - omlx.server - INFO - [-] - Chat completion: model=tiel-coder-oq4e, "
        f"{out_tokens} tokens in {gen_s}s ({tok_s} tok/s), prompt: {prompt}, "
        f"finish_reason={finish}, max_tokens=80000, request_max_tokens=80000"
    )


def _prefix(clock, reprefill, total, reused, req="2da900bb", day="2026-09-13"):
    return (
        f"{day} {clock},000 - omlx.scheduler - INFO - [-] - prefix cache: request {req} "
        f"re-prefills {reprefill} of {total} tokens (reused {reused}); closest stored sequence "
        f"57411c69 shares the first {reused} of {reused} comparable tokens before diverging"
    )


def _throttle(clock, day="2026-09-13"):
    return (
        f"{day} {clock},000 - omlx.scheduler - INFO - [-] - Request 2da900bb needs prefill headroom "
        "before throttling (reason=adaptive_prefill_throttle, current=31.43GB, predicted=7.07GB, target=37.44GB)"
    )


def _reclaim(clock, gb, day="2026-09-13"):
    return (
        f"{day} {clock},000 - omlx.engine_pool - INFO - [-] - Reclaimed {gb}GB of pooled Metal buffers "
        "for prefill request 2da900bb (no idle model to evict)"
    )


def _reject(clock, kv_len=81920, chunk=4096, cap_gb="34.2", source="omlx.scheduler - ERROR", day="2026-09-13"):
    return (
        f"{day} {clock},000 - {source} - [-] - Chunked prefill capacity rejected for 87d801cb: "
        f"Prefill context too large for available memory (pre-chunk guard at {chunk} tokens, kv_len={kv_len}): "
        f"predicted peak would exceed prefill safety cap {cap_gb}GB (90% of static/metal_cap ceiling 38.0GB). "
        "Raise kernel iogpu.wired_limit_mb in Terminal, or reduce context length."
    )


def _mtp(clock, rounds=12, accepted=30, proposed=48, tpr=2.5, block=4, day="2026-09-13"):
    return (
        f"{day} {clock},000 - omlx.engine - INFO - [-] - vlm_mtp stats: request=2da900bb finish=stop "
        f"rounds={rounds} accepted={accepted}/{proposed} (62.5%) tokens_per_round={tpr} "
        f"emitted=60 block_size={block}"
    )


def _trial(
    tmp_path,
    lines,
    started="09:00:00",
    finished="09:30:00",
    day="2026-09-13",
    finished_day=None,
):
    trial_dir = tmp_path / "sometask__abc1234"
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "agent_execution": {
                    "started_at": _local(day, started).isoformat().replace("+00:00", "Z"),
                    "finished_at": _local(finished_day or day, finished).isoformat().replace("+00:00", "Z"),
                }
            }
        )
    )
    server_log = tmp_path / "server.log"
    server_log.write_text("\n".join(lines) + "\n")
    return trial_dir, server_log


def _rows(trial_dir):
    with (trial_dir / R.CSV_FILENAME).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_log_timestamps_are_localized_before_comparison():
    """A naive local stamp must land on the same instant result.json's Z stamp means.

    The expectation is built through time.mktime rather than _local's
    strptime().astimezone() chain: that chain is what parse_log_ts does
    internally, so a test using it passes even if the localization is dropped.
    """
    line = _chat("09:05:00", 100, 10.0, 10.0, 5000)
    epoch = time.mktime((2026, 9, 13, 9, 5, 0, 0, 0, -1))
    assert R.parse_log_ts(line) == datetime.fromtimestamp(epoch, timezone.utc)


def test_log_timestamps_carry_milliseconds():
    line = "2026-09-13 09:05:00,371 - omlx.server - INFO - [-] - Chat completion: x"
    assert R.parse_log_ts(line).microsecond == 371000


def test_non_timestamped_lines_are_skipped():
    assert R.parse_log_ts("    at some traceback frame\n") is None


def test_iso_z_timestamps_parse_as_utc():
    assert R.parse_iso_utc("2026-09-13T09:45:20.391135Z") == datetime(
        2026, 9, 13, 9, 45, 20, 391135, tzinfo=timezone.utc
    )


def test_lines_outside_the_trial_window_are_excluded(tmp_path):
    trial_dir, server_log = _trial(
        tmp_path,
        [
            _chat("08:00:00", 100, 10.0, 10.0, 1111),
            _chat("09:10:00", 200, 10.0, 20.0, 2222),
            _chat("10:00:00", 300, 10.0, 30.0, 3333),
        ],
    )
    R.report(trial_dir, server_log)
    assert [r["prompt_tokens"] for r in _rows(trial_dir)] == ["2222"]


def test_prefix_cache_is_paired_with_the_completion_whose_prompt_matches_its_total(tmp_path):
    trial_dir, server_log = _trial(
        tmp_path,
        [
            _prefix("09:01:00", 8280, 163928, 155648),
            _chat("09:02:00", 7825, 530.46, 21.6, 163928),
        ],
    )
    R.report(trial_dir, server_log)
    row = _rows(trial_dir)[0]
    assert row["reused_tokens"] == "155648"
    assert row["reprefill_tokens"] == "8280"


def test_a_prefix_line_whose_total_does_not_match_is_not_attributed(tmp_path):
    trial_dir, server_log = _trial(
        tmp_path,
        [
            _prefix("09:01:00", 8280, 163928, 155648),
            _chat("09:02:00", 7825, 530.46, 21.6, 99999),
        ],
    )
    R.report(trial_dir, server_log)
    row = _rows(trial_dir)[0]
    assert row["reused_tokens"] == ""
    assert row["reprefill_tokens"] == ""


def test_a_fully_cached_completion_with_no_prefix_line_has_blank_cache_columns(tmp_path):
    trial_dir, server_log = _trial(tmp_path, [_chat("09:02:00", 100, 10.0, 10.0, 5000)])
    R.report(trial_dir, server_log)
    assert _rows(trial_dir)[0]["reused_tokens"] == ""


def test_throttles_and_reclaims_are_attributed_to_the_turn_they_fall_in(tmp_path):
    trial_dir, server_log = _trial(
        tmp_path,
        [
            _throttle("09:01:00"),
            _reclaim("09:01:01", "4.01"),
            _throttle("09:01:30"),
            _reclaim("09:01:31", "2.45"),
            _chat("09:02:00", 100, 10.0, 10.0, 1000),
            _throttle("09:03:00"),
            _reclaim("09:03:01", "1.50"),
            _chat("09:04:00", 200, 10.0, 20.0, 2000),
            _chat("09:05:00", 300, 10.0, 30.0, 3000),
        ],
    )
    R.report(trial_dir, server_log)
    rows = _rows(trial_dir)
    assert [r["throttle_events"] for r in rows] == ["2", "1", "0"]
    assert [r["reclaimed_gb"] for r in rows] == ["6.46", "1.5", "0.0"]


def test_turn_wall_s_spans_from_the_previous_completion(tmp_path):
    trial_dir, server_log = _trial(
        tmp_path,
        [_chat("09:02:00", 100, 10.0, 10.0, 1000), _chat("09:05:30", 200, 10.0, 20.0, 2000)],
    )
    R.report(trial_dir, server_log)
    rows = _rows(trial_dir)
    assert rows[0]["turn_wall_s"] == "120.0"
    assert rows[1]["turn_wall_s"] == "210.0"


def test_decode_rate_and_finish_reason_come_from_the_completion_line(tmp_path):
    trial_dir, server_log = _trial(tmp_path, [_chat("09:02:00", 7825, 530.46, 21.6, 156023, finish="length")])
    R.report(trial_dir, server_log)
    row = _rows(trial_dir)[0]
    assert row["decode_tok_s"] == "21.6"
    assert row["finish_reason"] == "length"
    assert row["prompt_tokens"] == "156023"


def test_mtp_columns_are_blank_not_zero_when_mtp_never_ran(tmp_path):
    """Zero would read as 'MTP ran and accepted nothing', which is a different claim."""
    trial_dir, server_log = _trial(tmp_path, [_chat("09:02:00", 100, 10.0, 10.0, 1000)])
    R.report(trial_dir, server_log)
    row = _rows(trial_dir)[0]
    assert row["mtp_rounds"] == ""
    assert row["mtp_accepted"] == ""
    assert row["mtp_proposed"] == ""
    assert row["mtp_tokens_per_round"] == ""
    assert row["mtp_block_size"] == ""


def test_mtp_stats_are_attached_to_the_completion_that_follows_them(tmp_path):
    trial_dir, server_log = _trial(
        tmp_path,
        [
            _mtp("09:01:59", rounds=12, accepted=30, tpr=2.5, block=4),
            _chat("09:02:00", 100, 10.0, 10.0, 1000),
            _chat("09:03:00", 100, 10.0, 10.0, 2000),
        ],
    )
    R.report(trial_dir, server_log)
    rows = _rows(trial_dir)
    assert rows[0]["mtp_rounds"] == "12"
    assert rows[0]["mtp_accepted"] == "30"
    assert rows[0]["mtp_tokens_per_round"] == "2.5"
    assert rows[0]["mtp_block_size"] == "4"
    assert rows[1]["mtp_rounds"] == ""


def test_prefill_rejections_are_summarized_and_kept_out_of_the_turn_rows(tmp_path, capsys):
    trial_dir, server_log = _trial(
        tmp_path,
        [_chat("09:02:00", 100, 10.0, 10.0, 1000), _reject("09:03:00", kv_len=118784)],
    )
    R.report(trial_dir, server_log)
    assert len(_rows(trial_dir)) == 1
    out = capsys.readouterr().out
    assert "prefill rejections: 1" in out
    assert "kv_len=118784" in out
    assert "ceiling=38.0GB" in out


def test_the_scheduler_and_server_copies_of_one_rejection_count_once(tmp_path, capsys):
    trial_dir, server_log = _trial(
        tmp_path,
        [
            _reject("09:03:00", source="omlx.scheduler - ERROR"),
            _reject("09:03:00", source="omlx.server - WARNING"),
        ],
    )
    R.report(trial_dir, server_log)
    assert "prefill rejections: 1" in capsys.readouterr().out


def test_no_rejection_is_reported_as_none(tmp_path, capsys):
    trial_dir, server_log = _trial(tmp_path, [_chat("09:02:00", 100, 10.0, 10.0, 1000)])
    R.report(trial_dir, server_log)
    assert "prefill rejections: none" in capsys.readouterr().out


def test_turn_idx_numbers_completions_by_their_position_in_the_window(tmp_path):
    """One agent-loop turn can drive several completions; turn_idx counts rows, not turns."""
    trial_dir, server_log = _trial(
        tmp_path,
        [
            _chat("09:02:00", 100, 10.0, 10.0, 1000),
            _chat("09:03:00", 100, 10.0, 10.0, 2000),
            _chat("09:04:00", 100, 10.0, 10.0, 3000),
        ],
    )
    R.report(trial_dir, server_log)
    assert [r["turn_idx"] for r in _rows(trial_dir)] == ["1", "2", "3"]


def test_csv_has_the_agreed_columns_in_order(tmp_path):
    trial_dir, server_log = _trial(tmp_path, [_chat("09:02:00", 100, 10.0, 10.0, 1000)])
    R.report(trial_dir, server_log)
    expected = [
        "turn_idx",
        "prompt_tokens",
        "output_tokens",
        "reused_tokens",
        "reprefill_tokens",
        "decode_tok_s",
        "turn_wall_s",
        "throttle_events",
        "reclaimed_gb",
        "finish_reason",
        "mtp_rounds",
        "mtp_accepted",
        "mtp_proposed",
        "mtp_tokens_per_round",
        "mtp_block_size",
    ]
    assert R.CSV_COLUMNS == expected
    with (trial_dir / R.CSV_FILENAME).open(newline="") as handle:
        assert next(csv.reader(handle)) == expected


def test_output_tokens_reach_the_csv(tmp_path):
    """build_turns has always counted them; only CSV_COLUMNS kept them out of the file."""
    trial_dir, server_log = _trial(tmp_path, [_chat("09:02:00", 7825, 530.46, 21.6, 1000)])
    R.report(trial_dir, server_log)
    assert _rows(trial_dir)[0]["output_tokens"] == "7825"


def test_mtp_accepted_is_reported_with_the_proposed_denominator(tmp_path):
    trial_dir, server_log = _trial(
        tmp_path,
        [
            _mtp("09:01:59", accepted=30, proposed=48),
            _chat("09:02:00", 100, 10.0, 10.0, 1000),
            _chat("09:03:00", 100, 10.0, 10.0, 2000),
        ],
    )
    R.report(trial_dir, server_log)
    rows = _rows(trial_dir)
    assert rows[0]["mtp_accepted"] == "30"
    assert rows[0]["mtp_proposed"] == "48"
    assert rows[1]["mtp_proposed"] == ""


def test_summary_reports_median_decode_and_max_prompt(tmp_path, capsys):
    trial_dir, server_log = _trial(
        tmp_path,
        [
            _chat("09:01:00", 100, 10.0, 10.0, 1000),
            _chat("09:02:00", 100, 10.0, 30.0, 9000),
            _chat("09:03:00", 100, 10.0, 20.0, 5000),
        ],
    )
    R.report(trial_dir, server_log)
    out = capsys.readouterr().out
    assert "turns: 3" in out
    assert "median decode: 20.0 tok/s" in out
    assert "max prompt: 9000 tokens" in out


def test_an_empty_window_still_writes_a_header_only_csv(tmp_path, capsys):
    trial_dir, server_log = _trial(tmp_path, [_chat("03:00:00", 100, 10.0, 10.0, 1000)])
    R.report(trial_dir, server_log)
    assert _rows(trial_dir) == []
    assert "turns: 0" in capsys.readouterr().out


def test_cli_rejects_a_server_log_that_does_not_exist(tmp_path):
    with pytest.raises(SystemExit):
        R.main(["--trial-dir", str(tmp_path), "--server-log", str(tmp_path / "nope.log")])


def test_cli_writes_the_csv_for_an_explicit_rotated_server_log(tmp_path):
    trial_dir, server_log = _trial(tmp_path, [_chat("09:02:00", 100, 10.0, 10.0, 1000)])
    rotated = tmp_path / "server.log.2026-09-13"
    rotated.write_text(server_log.read_text())
    assert R.main(["--trial-dir", str(trial_dir), "--server-log", str(rotated)]) == 0
    assert (trial_dir / R.CSV_FILENAME).exists()


def test_throttles_after_the_last_completion_are_reported_not_dropped(tmp_path, capsys):
    """A window ending in a rejection or a kill leaves events with no turn row to land in."""
    trial_dir, server_log = _trial(
        tmp_path,
        [
            _chat("09:02:00", 100, 10.0, 10.0, 1000),
            _throttle("09:03:00"),
            _reclaim("09:03:01", "4.01"),
            _throttle("09:04:00"),
            _reject("09:05:00"),
        ],
    )
    R.report(trial_dir, server_log)
    assert [r["throttle_events"] for r in _rows(trial_dir)] == ["0"]
    out = capsys.readouterr().out
    assert "after the last completion: 2 throttle events, 4.01GB reclaimed" in out


def test_trailing_events_with_no_turns_do_not_claim_a_last_completion(tmp_path, capsys):
    """Zero turns is zero completions, so there is no "last completion" to be after."""
    trial_dir, server_log = _trial(tmp_path, [_throttle("09:01:00"), _reclaim("09:01:01", "4.01")])
    R.report(trial_dir, server_log)
    out = capsys.readouterr().out
    assert "after the last completion" not in out
    assert "1 throttle events, 4.01GB reclaimed (no completions in this window at all)" in out


def test_nothing_trailing_is_reported_when_every_event_landed_in_a_turn(tmp_path, capsys):
    trial_dir, server_log = _trial(
        tmp_path,
        [_throttle("09:01:00"), _chat("09:02:00", 100, 10.0, 10.0, 1000)],
    )
    R.report(trial_dir, server_log)
    assert "after the last completion" not in capsys.readouterr().out


def test_build_turns_returns_trailing_counts_alongside_turns_and_rejections(tmp_path):
    started = _local("2026-09-13", "09:00:00")
    events = [
        (_local("2026-09-13", "09:03:00"), "throttle", {}),
        (_local("2026-09-13", "09:03:30"), "reclaim", {"gb": "2.5"}),
    ]
    turns, rejections, trailing = R.build_turns(events, started)
    assert (turns, rejections) == ([], [])
    assert trailing == {"throttle_events": 1, "reclaimed_gb": 2.5}


def test_a_rejected_requests_cache_stats_do_not_attach_to_a_later_completion(tmp_path):
    """The rejected request never completes; a later turn can share its prompt length by chance."""
    trial_dir, server_log = _trial(
        tmp_path,
        [
            _prefix("09:01:00", 8280, 163928, 155648),
            _mtp("09:01:05", rounds=12, accepted=30, proposed=48),
            _reject("09:01:10", kv_len=163928),
            _chat("09:02:00", 100, 10.0, 10.0, 163928),
        ],
    )
    R.report(trial_dir, server_log)
    row = _rows(trial_dir)[0]
    assert row["reused_tokens"] == ""
    assert row["reprefill_tokens"] == ""
    assert row["mtp_rounds"] == ""
    assert row["mtp_accepted"] == ""


def test_two_rejections_differing_only_in_chunk_size_are_both_counted(tmp_path, capsys):
    trial_dir, server_log = _trial(
        tmp_path,
        [
            _reject("09:03:00", kv_len=81920, chunk=4096),
            _reject("09:03:00", kv_len=81920, chunk=8192),
        ],
    )
    R.report(trial_dir, server_log)
    assert "prefill rejections: 2" in capsys.readouterr().out


def test_two_rejections_differing_only_in_the_safety_cap_are_both_counted(tmp_path, capsys):
    trial_dir, server_log = _trial(
        tmp_path,
        [
            _reject("09:03:00", kv_len=81920, cap_gb="34.2"),
            _reject("09:03:00", kv_len=81920, cap_gb="30.0"),
        ],
    )
    R.report(trial_dir, server_log)
    assert "prefill rejections: 2" in capsys.readouterr().out


def _midnight_trial(tmp_path, live_lines, first_day="2026-09-12"):
    """A trial whose window runs from 23:30 on first_day to 00:30 on 2026-09-13."""
    return _trial(
        tmp_path,
        live_lines,
        started="23:30:00",
        day=first_day,
        finished="00:30:00",
        finished_day="2026-09-13",
    )


def test_a_window_crossing_local_midnight_reads_both_sides_of_the_rotation(tmp_path, capsys):
    """The rotated sibling holds the pre-midnight half; reporting only the live log loses it."""
    trial_dir, server_log = _midnight_trial(tmp_path, [_chat("00:10:00", 300, 10.0, 30.0, 3000)])
    rotated = tmp_path / "server.log.2026-09-12"
    rotated.write_text(
        "\n".join(
            [
                _chat("23:40:00", 100, 10.0, 10.0, 1000, day="2026-09-12"),
                _chat("23:50:00", 200, 10.0, 20.0, 2000, day="2026-09-12"),
            ]
        )
        + "\n"
    )
    R.report(trial_dir, server_log)
    assert [r["prompt_tokens"] for r in _rows(trial_dir)] == ["1000", "2000", "3000"]
    err = capsys.readouterr().err
    assert "reading 2 log files" in err
    assert str(rotated) in err


def test_merged_events_are_ordered_by_timestamp_not_by_file(tmp_path):
    """build_turns accumulates statefully, so the file read first must not win."""
    trial_dir, server_log = _midnight_trial(tmp_path, [_throttle("00:05:00", day="2026-09-13")])
    (tmp_path / "server.log.2026-09-12").write_text(_throttle("23:40:00", day="2026-09-12") + "\n")
    started_at, finished_at = R.read_window(trial_dir)
    events = R.collect_window_events(server_log, started_at, finished_at)
    assert [ts for ts, _kind, _fields in events] == [
        _local("2026-09-12", "23:40:00"),
        _local("2026-09-13", "00:05:00"),
    ]


def test_a_same_day_window_reads_only_the_given_log(tmp_path, capsys):
    trial_dir, server_log = _trial(tmp_path, [_chat("09:02:00", 100, 10.0, 10.0, 1000)])
    R.report(trial_dir, server_log)
    assert capsys.readouterr().err == ""


def test_the_log_being_read_is_not_suggested_back_to_the_user(tmp_path):
    """Pointed at the pre-midnight rotated file, the gap is the live log, not itself."""
    (tmp_path / "server.log").write_text("")
    siblings = R.rotated_siblings(
        tmp_path / "server.log.2026-09-12",
        _local("2026-09-12", "23:30:00"),
        _local("2026-09-13", "00:30:00"),
    )
    assert siblings == [tmp_path / "server.log"]


def test_a_sibling_retention_has_already_deleted_is_skipped(tmp_path, capsys):
    """logging.retention_days removes old rotated logs; naming one sends the reader nowhere."""
    trial_dir, server_log = _midnight_trial(
        tmp_path, [_chat("00:10:00", 300, 10.0, 30.0, 3000)], first_day="2026-09-11"
    )
    kept = tmp_path / "server.log.2026-09-11"
    kept.write_text(_chat("23:40:00", 100, 10.0, 10.0, 1000, day="2026-09-11") + "\n")
    deleted = tmp_path / "server.log.2026-09-12"
    assert not deleted.exists()

    R.report(trial_dir, server_log)

    assert R.rotated_siblings(server_log, *R.read_window(trial_dir)) == [kept]
    assert [r["prompt_tokens"] for r in _rows(trial_dir)] == ["1000", "3000"]
    err = capsys.readouterr().err
    assert str(deleted) not in err
    assert str(kept) in err


def test_a_trial_that_died_before_agent_execution_gets_a_clear_error(tmp_path):
    trial_dir = tmp_path / "sometask__abc1234"
    trial_dir.mkdir()
    (trial_dir / "result.json").write_text(json.dumps({"id": "sometask__abc1234", "agent_execution": None}))
    with pytest.raises(SystemExit) as excinfo:
        R.read_window(trial_dir)
    assert "no agent_execution timing" in str(excinfo.value)


def test_a_run_directory_mistaken_for_a_trial_directory_gets_the_same_error(tmp_path):
    run_dir = tmp_path / "2026-09-13__00-27-39"
    run_dir.mkdir()
    (run_dir / "result.json").write_text(
        json.dumps({"id": "run", "started_at": "2026-09-13T00:27:39Z", "n_total_trials": 89})
    )
    server_log = tmp_path / "server.log"
    server_log.write_text("")
    with pytest.raises(SystemExit) as excinfo:
        R.main(["--trial-dir", str(run_dir), "--server-log", str(server_log)])
    assert "no agent_execution timing" in str(excinfo.value)


def test_cli_rejects_a_trial_dir_that_does_not_exist(tmp_path):
    server_log = tmp_path / "server.log"
    server_log.write_text("")
    with pytest.raises(SystemExit):
        R.main(["--trial-dir", str(tmp_path / "typo"), "--server-log", str(server_log)])
