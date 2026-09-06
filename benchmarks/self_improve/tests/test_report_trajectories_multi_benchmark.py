"""Integration test: report_trajectories.py's actual pipeline (ingest_all +
dry_run_report) against TWO real benchmarks' data at once, not each in
isolation. Moved here from run_gepa.py, which no longer scores from frozen
historical data at all (see polyglot_adapter.py's module docstring) --
report_trajectories.py still serves VALIDATION_PLAN Layers 2-3 independent
of the live-execution GEPA loop.

Coverage gap this closes: Layer 2/3 were validated for aider_polyglot and tb
SEPARATELY (one --log-roots key at a time) during manual VALIDATION_PLAN.md
runs, but never together in one invocation -- the exact multi-benchmark
aggregation scenario the tool exists for. Uses the real captured tb fixture
(tests/fixtures/real_tb_run/) plus a small real-shaped aider_polyglot
fixture built the same way _dump_trajectory() actually writes it.
"""
import json
from pathlib import Path

from benchmarks.self_improve.components import load_components
from benchmarks.self_improve.report_trajectories import DEFAULT_WEIGHTS, dry_run_report, ingest_all as _ingest_all

REAL_TB_FIXTURE = Path(__file__).parent / "fixtures" / "real_tb_run"


def _make_aider_fixture(tmp_path):
    log_root = tmp_path / "logs"
    ex = log_root / "python" / "leap"
    ex.mkdir(parents=True)
    (ex / "trajectory_1.json").write_text(json.dumps({
        "attempt": "1", "agent_ended": True, "turn_count": 2,
        "compaction_events": 0, "assistant_text": "done",
        "tool_calls": [{"name": "write", "args": {}, "result_text": "ok", "is_error": False}],
    }))
    results_json = tmp_path / "results_full_polyglot.json"
    results_json.write_text(json.dumps({"exercises": {
        "python/leap": {"status": "pass_1", "stop_reason_1": "agent_end",
                         "stop_reason_2": None, "turn_count": 2},
    }}))
    return log_root, results_json


def test_ingest_all_combines_both_benchmarks(tmp_path):
    log_root, results_json = _make_aider_fixture(tmp_path)
    log_roots = {
        "aider": f"{log_root},{results_json}",
        "tb": str(REAL_TB_FIXTURE),
    }
    trajectories, empty_sources = _ingest_all(log_roots)
    benchmarks_seen = {t.benchmark for t in trajectories}
    assert benchmarks_seen == {"aider_polyglot", "tb"}
    assert len(trajectories) == 2
    assert empty_sources == []


def test_ingest_all_records_empty_source_when_a_requested_root_yields_nothing(tmp_path):
    """Real gap, confirmed by review: a requested source that ingests
    successfully but finds zero trajectories (e.g. a log_root that exists
    but is empty) must be distinguishable from one that was never requested,
    so _real_run() can refuse rather than silently training on a subset."""
    log_root, results_json = _make_aider_fixture(tmp_path)
    empty_gaia_root = tmp_path / "empty_gaia_logs"
    empty_gaia_root.mkdir()
    log_roots = {
        "aider": f"{log_root},{results_json}",
        "gaia": str(empty_gaia_root),
    }
    trajectories, empty_sources = _ingest_all(log_roots)
    assert len(trajectories) == 1  # aider's one trajectory
    assert empty_sources == ["gaia"]


def test_ingest_all_reports_clear_error_when_aider_log_roots_missing_comma(tmp_path):
    """Real gap, confirmed by review: `--log-roots aider=<dir>` with no
    ',<results.json>' suffix used to make Path("") -> Path(".") -- whose
    .exists() is True -- silently bypass aider_polyglot_ingest.load()'s own
    FileNotFoundError guard and surface as an opaque IsADirectoryError deep
    inside json.loads(). Must be caught before ever calling load()."""
    log_root, _results_json = _make_aider_fixture(tmp_path)
    log_roots = {"aider": str(log_root)}  # no comma, no results.json path
    trajectories, empty_sources = _ingest_all(log_roots)
    assert trajectories == []
    assert empty_sources == ["aider"]


def test_ingest_all_reports_clear_error_when_aider_log_root_segment_is_empty(tmp_path):
    """Real follow-up bug, confirmed by review: `--log-roots aider=,results.json`
    (empty segment BEFORE a present comma) still made Path("") -> Path(".")
    for the log_root -- the original fix only checked the comma's presence,
    not that both sides of it were non-empty."""
    _log_root, results_json = _make_aider_fixture(tmp_path)
    log_roots = {"aider": f",{results_json}"}
    trajectories, empty_sources = _ingest_all(log_roots)
    assert trajectories == []
    assert empty_sources == ["aider"]


def test_ingest_all_reports_clear_error_when_aider_results_json_segment_is_empty(tmp_path):
    """Same gap, other side: `--log-roots aider=logs,` (empty segment AFTER
    a present comma) for the results_json path."""
    log_root, _results_json = _make_aider_fixture(tmp_path)
    log_roots = {"aider": f"{log_root},"}
    trajectories, empty_sources = _ingest_all(log_roots)
    assert trajectories == []
    assert empty_sources == ["aider"]


def test_ingest_all_records_empty_source_when_ingest_raises(tmp_path):
    """Same guarantee on the exception path: a malformed results.json for
    aider_polyglot raises inside ingest today (confirmed: aider_polyglot_ingest
    reads it eagerly) -- this must still be reported as an empty source, not
    just logged and forgotten."""
    log_root, _results_json = _make_aider_fixture(tmp_path)
    malformed_results_json = tmp_path / "not_json.json"
    malformed_results_json.write_text("not valid json{{{")
    log_roots = {"aider": f"{log_root},{malformed_results_json}"}
    trajectories, empty_sources = _ingest_all(log_roots)
    assert trajectories == []
    assert empty_sources == ["aider"]


def test_dry_run_produces_a_plausible_weighted_aggregate_across_benchmarks(tmp_path, capsys):
    """The real aider_polyglot trajectory passes (score 1.0), the real tb
    trajectory fails (score 0.0, per real is_resolved=false ground truth) --
    the weighted aggregate must land strictly between them, not collapse to
    either extreme or crash."""
    log_root, results_json = _make_aider_fixture(tmp_path)
    log_roots = {
        "aider": f"{log_root},{results_json}",
        "tb": str(REAL_TB_FIXTURE),
    }
    real_repo_root = Path(__file__).parent.parent.parent.parent  # little-coder-self-improve/
    # repo_root: real gap, confirmed by review -- main() always passes
    # repo_root through to _ingest_all() (needed to resolve knowledge-inject
    # component usage), so omitting it here made this integration test
    # exercise a calling convention main() never actually uses.
    trajectories, _empty_sources = _ingest_all(log_roots, repo_root=real_repo_root)

    components = load_components(
        real_repo_root / "benchmarks" / "self_improve" / "config" / "components.yaml",
        repo_root=real_repo_root,
    )

    dry_run_report(trajectories, components, DEFAULT_WEIGHTS)
    out = capsys.readouterr().out

    assert "aider_polyglot: 1 trajectories, avg score 1.000" in out
    assert "tb: 1 trajectories, avg score 0.000" in out
    # aggregate must be strictly between the two per-benchmark scores
    agg_line = next(line for line in out.splitlines() if "Weighted aggregate score" in line)
    agg_value = float(agg_line.split(":")[1].strip())
    assert 0.0 < agg_value < 1.0
