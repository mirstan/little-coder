"""run_watch IO: tailing (partial lines, truncation, rotation, backlog), job
discovery, the poll loop end to end, the status line, the CLI, and running
from a lone copy of the script."""
import ast
import io
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_watch as W  # noqa: E402

TRIAL = "fix-git__abc123"


def rec(turn, **kw):
    base = {"v": 1, "turn": turn, "ts_start": 1000.0 + turn * 10, "ts_end": 1005.0 + turn * 10,
            "iso_end": "", "headers_s": 0.5, "ttft_s": 1.0, "gen_s": 4.0, "turn_s": 5.0,
            "stop_reason": "toolUse", "error_message": None, "usage_reported": True,
            "input": 500, "cache_read": 0, "cache_write": 0, "output": 200, "reasoning": 0,
            "prompt_tokens": 500, "context_tokens": 700, "cache_hit": 0.0, "text_chars": 0,
            "thinking_chars": 0, "n_tool_calls": 0, "max_tool_arg_bytes": 0, "tool_calls": [],
            "tool_errors": 0, "retention": None, "demoted_new": None, "stubs": None,
            "stubbed_new": None, "guards": [], "compactions": [], "retries_before": 0}
    base.update(kw)
    return base


def make_job(root: Path, name="2026-10-07__18-00-00", trials=(TRIAL,)) -> Path:
    job = root / name
    job.mkdir(parents=True)
    (job / "config.json").write_text(json.dumps({
        "job_name": name, "datasets": [{"task_names": ["terminal-bench/fix-git", "terminal-bench/regex-log"]}]}))
    for t in trials:
        (job / t / "agent").mkdir(parents=True)
        (job / t / "config.json").write_text(json.dumps({"trial_name": t}))
    return job


@pytest.fixture(autouse=True)
def _no_watch_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("RUN_WATCH_"):
            monkeypatch.delenv(key)


def test_a_partial_trailing_line_waits_for_its_newline(tmp_path):
    p = tmp_path / "turns.jsonl"
    p.write_bytes(b"a\nb")
    st = W.TailState(path=p)
    assert W.read_new_lines(st) == ["a"]
    with p.open("ab") as fh:
        fh.write(b"c\n")
    assert W.read_new_lines(st) == ["bc"]
    assert W.read_new_lines(st) == []


def test_a_missing_file_reads_empty_until_it_appears(tmp_path):
    p = tmp_path / "turns.jsonl"
    st = W.TailState(path=p)
    assert W.read_new_lines(st) == []
    p.write_text("x\n")
    assert W.read_new_lines(st) == ["x"]


def test_truncation_restarts_from_the_top(tmp_path):
    p = tmp_path / "serve.log"
    p.write_text("one\ntwo\n")
    st = W.TailState(path=p)
    assert W.read_new_lines(st) == ["one", "two"]
    p.write_text("x\n")
    assert W.read_new_lines(st) == ["x"]


def test_rotation_to_a_new_inode_reopens_the_new_file(tmp_path):
    p = tmp_path / "omlx.log"
    p.write_text("a\nb\n")
    st = W.TailState(path=p)
    assert W.read_new_lines(st) == ["a", "b"]
    fresh = tmp_path / "omlx.log.new"
    fresh.write_text("c\nd\ne\n")
    os.replace(fresh, p)
    assert W.read_new_lines(st) == ["c", "d", "e"]


def test_a_server_log_backlog_starts_near_the_end_on_a_line_boundary(tmp_path):
    p = tmp_path / "omlx.log"
    p.write_bytes(b"line-one\nline-two\nline-three\n")
    st = W.TailState(path=p, start_at_end_bytes=15)
    assert W.read_new_lines(st) == ["line-three"]


def test_a_backlog_that_starts_exactly_on_a_line_start_keeps_that_line(tmp_path):
    p = tmp_path / "omlx.log"
    p.write_bytes(b"line-one\nline-two\nline-three\n")
    st = W.TailState(path=p, start_at_end_bytes=len(b"line-two\nline-three\n"))
    assert W.read_new_lines(st) == ["line-two", "line-three"]


def test_job_and_trial_discovery_use_harbor_config_keys(tmp_path):
    old = make_job(tmp_path, "2026-10-06__10-00-00")
    new = make_job(tmp_path, "2026-10-07__18-00-00", trials=(TRIAL, "regex-log__xyz789"))
    (new / "not-a-trial").mkdir()
    os.utime(old / "config.json", (1000, 1000))
    os.utime(new / "config.json", (2000, 2000))
    assert W.resolve_job_dir(tmp_path) == new
    assert W.resolve_job_dir(old) == old
    assert W.resolve_job_dir(tmp_path / "missing") is None
    assert [p.name for p in W.discover_trials(new)] == [TRIAL, "regex-log__xyz789"]


def test_a_trial_counts_as_finished_only_with_finished_at(tmp_path):
    job = make_job(tmp_path)
    t = job / TRIAL
    assert W.read_trial_result(t) is None
    (t / "result.json").write_text(json.dumps({"started_at": "2026-10-07T18:00:00Z", "finished_at": None}))
    assert W.read_trial_result(t) is None
    (t / "result.json").write_text("{\"finished_at\": \"2026-10-07T19:")
    assert W.read_trial_result(t) is None
    (t / "result.json").write_text(json.dumps({"finished_at": "2026-10-07T19:00:00Z",
                                               "verifier_result": {"rewards": {"reward": 1.0}}}))
    res = W.read_trial_result(t)
    assert (res["reward"], res["passed"]) == (1.0, True)
    assert isinstance(res["finished_ts"], float)


def test_attribute_trial_picks_the_window_holding_the_timestamp():
    windows = [("a__1", 100.0, 200.0), ("b__2", 200.0, 300.0), ("c__3", 250.0, 400.0)]
    assert W.attribute_trial(150.0, windows) == "a__1"
    assert W.attribute_trial(260.0, windows) == "c__3"
    assert W.attribute_trial(500.0, windows) is None


def test_poll_once_alerts_on_new_turns_once_and_reports_finished_trials(tmp_path):
    job = make_job(tmp_path)
    turns = job / TRIAL / "agent" / "turns.jsonl"
    turns.write_text(json.dumps(rec(1)) + "\n" + json.dumps(rec(2, stop_reason="length", output=80_000)) + "\n")
    ws = W.WatchState(target=tmp_path, cfg=W.RuleConfig())
    now = time.time()
    alerts, lines = W.poll_once(ws, now)
    assert ws.job_dir == job
    assert [(a.rule, a.trial) for a in alerts] == [("length_stop", TRIAL), ("big_output", TRIAL)]
    assert any(line.startswith("[status") for line in lines)

    alerts2, lines2 = W.poll_once(ws, now + 1)
    assert alerts2 == []
    assert not any(line.startswith("[status") for line in lines2)

    with turns.open("a") as fh:
        fh.write(json.dumps(rec(3, ttft_s=200.0)))
    assert W.poll_once(ws, now + 2)[0] == []
    with turns.open("a") as fh:
        fh.write("\n")
    assert [a.rule for a in W.poll_once(ws, now + 3)[0]] == ["ttft_spike"]

    (job / TRIAL / "result.json").write_text(json.dumps({
        "finished_at": "2026-10-07T19:00:00Z", "verifier_result": {"rewards": {"reward": 1.0}}}))
    _, lines4 = W.poll_once(ws, now + 400)
    assert any("fix-git finished PASS" in line for line in lines4)
    assert any("1/2 done, pass=1 fail=0" in line for line in lines4)


def test_a_trial_with_no_new_turn_for_stall_min_raises_one_stall_alert(tmp_path):
    job = make_job(tmp_path)
    turns = job / TRIAL / "agent" / "turns.jsonl"
    turns.write_text(json.dumps(rec(1)) + "\n")
    (job / TRIAL / "agent" / "little_coder.log").write_text("")  # created empty at trial start
    t0 = time.time() - 3600
    for p in (job / TRIAL / "config.json", turns):
        os.utime(p, (t0, t0))
    ws = W.WatchState(target=job, cfg=W.RuleConfig())
    alerts, _ = W.poll_once(ws, t0 + 21 * 60)
    assert [a.rule for a in alerts] == ["stall"]
    assert "hung" in alerts[0].message
    assert W.poll_once(ws, t0 + 22 * 60)[0] == []


def test_a_trial_whose_agent_finished_is_verifying_not_stalled(tmp_path):
    job = make_job(tmp_path)
    agent = job / TRIAL / "agent"
    turns = agent / "turns.jsonl"
    turns.write_text(json.dumps(rec(1)) + "\n")
    (agent / "little_coder.log").write_text("=== stop_reason: agent_end ===\n")
    t0 = time.time() - 3600
    for p in (job / TRIAL / "config.json", turns):
        os.utime(p, (t0, t0))
    ws = W.WatchState(target=job, cfg=W.RuleConfig())
    alerts, lines = W.poll_once(ws, t0 + 30 * 60)
    assert alerts == []
    assert any("fix-git agent done, verifying" in line for line in lines)
    assert ws.trials[TRIAL].state.verifying is True
    _, lines2 = W.poll_once(ws, t0 + 31 * 60, force_status=True)
    assert not any("agent done, verifying" in line for line in lines2)
    assert any(line.endswith(" verifying") for line in lines2 if line.startswith("[status"))


def test_server_lines_from_before_the_job_seed_state_but_never_alert(tmp_path):
    job = make_job(tmp_path)
    start = datetime(2026, 10, 7, 18, 0, 0).timestamp()
    for p in (job / "config.json", job / TRIAL / "config.json"):
        os.utime(p, (start, start))
    log = tmp_path / "omlx.log"
    log.write_text(
        "2026-10-07 17:10:00,000 - omlx.server - ERROR - prefill_memory_exceeded: request rejected\n"
        "2026-10-07 18:05:00,000 - omlx.server - ERROR - prefill_memory_exceeded: request rejected\n")
    ws = W.WatchState(target=job, cfg=W.RuleConfig(), server=W.TailState(path=log), server_kind="omlx")
    alerts, _ = W.poll_once(ws, start + 600)
    assert [(a.rule, a.trial, a.ts) for a in alerts] == [
        ("server_error", TRIAL, datetime(2026, 10, 7, 18, 5, 0).timestamp())]


def test_without_a_target_it_uses_RUN_WATCH_JOBS_DIR_or_the_cwd_and_fails_clearly(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert W.main(["--once"], out=io.StringIO()) == 2
    assert "benchmarks/harbor_runs" in capsys.readouterr().err
    job = make_job(tmp_path / "benchmarks" / "harbor_runs")
    (job / TRIAL / "agent" / "turns.jsonl").write_text(json.dumps(rec(1, stop_reason="length")) + "\n")
    assert W.main(["--once"], out=io.StringIO()) == 1
    other = tmp_path / "elsewhere"
    make_job(other)
    monkeypatch.setenv("RUN_WATCH_JOBS_DIR", str(other))
    out = io.StringIO()
    assert W.main(["--once"], out=out) == 0
    assert f"watching job {other}" in out.getvalue()


def test_splash_lines_across_midnight_are_dated_and_attributed_to_the_running_trial(tmp_path):
    job = make_job(tmp_path)
    start = datetime(2026, 10, 7, 23, 50, 0).timestamp()
    for p in (job / "config.json", job / TRIAL / "config.json"):
        os.utime(p, (start, start))
    log = tmp_path / "serve-tb.log"
    log.write_text(
        "23:58:10 Done · input 50,000 · cached 45,000 · output 120 · TTFT 2.0s · 40.0 tok/s\n"
        "00:00:30 Done · input 52,000 · cached 9,000 · output 48,227 · tools 4·a72fb0c4 · TTFT 247.3s · 30.1 tok/s\n",
        encoding="utf-8")
    written = datetime(2026, 10, 8, 0, 0, 31).timestamp()
    os.utime(log, (written, written))
    ws = W.WatchState(target=job, cfg=W.RuleConfig(), server=W.TailState(path=log), server_kind="auto")
    alerts, _ = W.poll_once(ws, written + 5)
    at = datetime(2026, 10, 8, 0, 0, 30).timestamp()
    assert sorted((a.rule, a.trial, a.ts) for a in alerts) == [
        ("big_output", TRIAL, at), ("cache_collapse", TRIAL, at), ("ttft_spike", TRIAL, at)]
    assert ws.server_kind == "splash"


def test_an_omlx_mid_history_divergence_is_attributed_by_timestamp(tmp_path):
    job = make_job(tmp_path)
    start = datetime(2026, 10, 7, 18, 0, 0).timestamp()
    for p in (job / "config.json", job / TRIAL / "config.json"):
        os.utime(p, (start, start))
    log = tmp_path / "omlx.log"
    log.write_text(
        "2026-10-07 18:04:07,398 - omlx.scheduler - INFO - prefix cache: request 964228db-3670-4e00-b06b-efba8572ae77 "
        "re-prefills 40220 of 71530 tokens (reused 31310); closest stored sequence "
        "0fbf9c00-2858-4e1e-9f2f-3847d517846d shares the first 31310 of 69632 comparable tokens before diverging\n")
    ws = W.WatchState(target=job, cfg=W.RuleConfig(), server=W.TailState(path=log), server_kind="omlx")
    alerts, _ = W.poll_once(ws, start + 300)
    assert [(a.rule, a.trial) for a in alerts] == [("prefix_divergence", TRIAL)]


def test_the_status_line_reports_progress_the_current_trial_and_the_server(tmp_path):
    job = make_job(tmp_path, trials=(TRIAL, "regex-log__xyz789"))
    now = time.time()
    done = W.TrialState(name=TRIAL, start_ts=now - 3600, finished=True, reward=1.0)
    running = W.TrialState(name="regex-log__xyz789", start_ts=now - 720, n_turns=34,
                           last_prompt_tokens=54_321, peak_prompt_tokens=88_000, compactions=1, max_demoted=8)
    ws = W.WatchState(target=job, cfg=W.RuleConfig(), job_dir=job, server_last={"ttft_s": 3.14, "prompt": 50_000, "cached": 46_000})
    ws.trials = {t.name: W.TrialWatch(dir=job / t.name, state=t, tail=W.TailState(path=job / t.name / "agent" / "turns.jsonl"))
                 for t in (done, running)}
    line = W.format_status(ws, now)
    assert line.startswith("[status ")
    assert line.endswith("] 2026-10-07__18-00-00: 1/2 done, pass=1 fail=0; current: regex-log "
                         "(12 min, turn 34, ctx 54K peak 88K, 1 cmp, demoted 8) | server ttft 3.1s cache 92%")


def test_once_writes_alerts_jsonl_returns_1_on_crit_and_does_not_repeat(tmp_path):
    job = make_job(tmp_path)
    (job / TRIAL / "agent" / "turns.jsonl").write_text(json.dumps(rec(1, stop_reason="length", output=900)) + "\n")
    out = io.StringIO()
    assert W.main([str(job), "--once"], out=out) == 1
    rows = [json.loads(line) for line in (job / "alerts.jsonl").read_text().splitlines()]
    assert [(r["rule"], r["trial"], r["subject"]) for r in rows] == [("length_stop", TRIAL, "turn 1")]
    assert "length_stop" in out.getvalue() and "[status" in out.getvalue()
    out2 = io.StringIO()
    assert W.main([str(job), "--once"], out=out2) == 0
    assert len((job / "alerts.jsonl").read_text().splitlines()) == 1


def test_rule_flags_beat_env_and_env_beats_defaults(tmp_path, monkeypatch):
    job = make_job(tmp_path)
    (job / TRIAL / "agent" / "turns.jsonl").write_text(json.dumps(rec(1, output=900, ttft_s=1.0)) + "\n")
    monkeypatch.setenv("RUN_WATCH_TTFT_MAX_S", "0.5")
    monkeypatch.setenv("RUN_WATCH_MAX_OUTPUT_TOKENS", "100000")
    assert W.main([str(job), "--once", "--max-output-tokens", "500"], out=io.StringIO()) == 0
    rules = sorted(json.loads(line)["rule"] for line in (job / "alerts.jsonl").read_text().splitlines())
    assert rules == ["big_output", "ttft_spike"]


def test_a_lone_copy_of_the_script_runs_and_imports_only_the_stdlib(tmp_path):
    job = make_job(tmp_path / "runs")
    (job / TRIAL / "agent" / "turns.jsonl").write_text(json.dumps(rec(1, stop_reason="length")) + "\n")
    copy_dir = tmp_path / "copy"
    copy_dir.mkdir()
    copy = copy_dir / "run_watch.py"
    shutil.copy(W.__file__, copy)
    env = {k: v for k, v in os.environ.items() if not k.startswith("RUN_WATCH_")}
    proc = subprocess.run([sys.executable, str(copy), str(job), "--once"],
                          capture_output=True, text=True, timeout=60, env=env, cwd=str(copy_dir))
    assert "length_stop" in proc.stdout, proc.stderr
    assert proc.returncode == 1

    tree = ast.parse(Path(W.__file__).read_text(encoding="utf-8"))
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            top.add((node.module or "").split(".")[0])
    nested = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) and n not in tree.body]
    assert nested == []
    assert top <= set(sys.stdlib_module_names) | {"__future__"}


def test_server_lines_never_alert_before_a_job_exists(tmp_path):
    """Live regression: started ahead of harbor, the watcher alerted on the
    whole 1 MiB omlx backlog because no job start bounded it yet."""
    jobs = tmp_path / "harbor_runs"
    jobs.mkdir()
    log = tmp_path / "omlx.log"
    log.write_text("2026-10-07 17:10:00,000 - omlx.server - ERROR - prefill_memory_exceeded: request rejected\n")
    ws = W.WatchState(target=jobs, cfg=W.RuleConfig(), server=W.TailState(path=log), server_kind="omlx")
    alerts, lines = W.poll_once(ws, datetime(2026, 10, 7, 18, 0, 0).timestamp())
    assert alerts == []
    assert any("no Harbor job" in line for line in lines)
