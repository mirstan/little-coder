"""Integration test: --dry-run's actual pipeline (_dry_run + _ingest_all)
against TWO real benchmarks' data at once, not each in isolation.

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
from benchmarks.self_improve.run_gepa import DEFAULT_WEIGHTS, _dry_run, _ingest_all

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
    trajectories = _ingest_all(log_roots)
    benchmarks_seen = {t.benchmark for t in trajectories}
    assert benchmarks_seen == {"aider_polyglot", "tb"}
    assert len(trajectories) == 2


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
    trajectories = _ingest_all(log_roots)

    real_repo_root = Path(__file__).parent.parent.parent.parent  # little-coder-self-improve/
    components = load_components(
        real_repo_root / "benchmarks" / "self_improve" / "config" / "components.yaml",
        repo_root=real_repo_root,
    )

    _dry_run(trajectories, components, DEFAULT_WEIGHTS)
    out = capsys.readouterr().out

    assert "aider_polyglot: 1 trajectories, avg score 1.000" in out
    assert "tb: 1 trajectories, avg score 0.000" in out
    assert "Errors encountered: 0" in out
    # aggregate must be strictly between the two per-benchmark scores
    agg_line = next(line for line in out.splitlines() if "Weighted aggregate score" in line)
    agg_value = float(agg_line.split(":")[1].strip())
    assert 0.0 < agg_value < 1.0
