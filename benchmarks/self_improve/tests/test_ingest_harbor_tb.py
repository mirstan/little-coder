from benchmarks.self_improve.ingest import harbor_tb_ingest


def test_load_parses_stop_reason_from_log_marker(tmp_path):
    log = tmp_path / "task-a.log"
    log.write_text("=== stop_reason: process_exit ===\nsome text\n")
    trajs = harbor_tb_ingest.load(tmp_path, benchmark="harbor")
    assert trajs[0].stop_reason == "process_exit"


def test_load_defaults_stop_reason_when_marker_absent(tmp_path):
    log = tmp_path / "task-b.log"
    log.write_text("no marker in this file at all\n")
    trajs = harbor_tb_ingest.load(tmp_path, benchmark="tb")
    assert trajs[0].stop_reason == "unknown"


def test_load_sets_benchmark_from_argument_not_content(tmp_path):
    """harbor and tb adapter logs look identical in format -- the caller
    must tell this module which benchmark literal to stamp, content alone
    doesn't disambiguate."""
    log = tmp_path / "task-a.log"
    log.write_text("=== stop_reason: agent_end ===\n")
    trajs = harbor_tb_ingest.load(tmp_path, benchmark="tb")
    assert trajs[0].benchmark == "tb"


def test_load_derives_success_from_stop_reason_when_no_other_signal(tmp_path):
    """Neither harbor nor tb adapter logs carry an explicit pass/fail field
    in this format -- success is approximated as (stop_reason == 'agent_end')
    until a richer signal is wired in. This approximation must be visible
    and tested, not silently assumed."""
    log = tmp_path / "task-a.log"
    log.write_text("=== stop_reason: agent_end ===\n")
    assert harbor_tb_ingest.load(tmp_path, benchmark="harbor")[0].success is True

    tmp_path2 = tmp_path / "other"
    tmp_path2.mkdir()
    log2 = tmp_path2 / "task-b.log"
    log2.write_text("=== stop_reason: deadline ===\n")
    assert harbor_tb_ingest.load(tmp_path2, benchmark="harbor")[0].success is False


def test_load_extracts_tool_calls_from_log_lines(tmp_path):
    log = tmp_path / "task-a.log"
    log.write_text(
        "=== stop_reason: agent_end ===\n"
        ">> bash({'command': 'ls'})\n"
        ">> read({'path': '/x'})\n"
    )
    traj = harbor_tb_ingest.load(tmp_path, benchmark="harbor")[0]
    assert "bash" in traj.summarized_transcript


def test_load_handles_empty_directory(tmp_path):
    assert harbor_tb_ingest.load(tmp_path, benchmark="harbor") == []


def test_load_handles_unreadable_file_without_crashing(tmp_path):
    """A permissions error or binary-garbage log file must produce a skip +
    warning, not an ingest-wide crash -- this is the module's core
    graceful-degradation contract per the plan."""
    log = tmp_path / "task-a.log"
    log.write_bytes(b"\xff\xfe\x00\x01garbage")
    trajs = harbor_tb_ingest.load(tmp_path, benchmark="harbor")
    assert trajs == [] or trajs[0].failure_signals  # either skipped or flagged, never raises


def test_load_components_used_is_empty(tmp_path):
    """No notification capture exists in this log format at all today --
    lock this in explicitly, same rationale as the aider_polyglot gap."""
    log = tmp_path / "task-a.log"
    log.write_text("=== stop_reason: agent_end ===\n")
    assert harbor_tb_ingest.load(tmp_path, benchmark="harbor")[0].components_used == []
