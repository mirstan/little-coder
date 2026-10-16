"""The ledger's IO half: never raises into a trial, writes one flushed line
per turn, and is wired into the Harbor adapter."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import turn_ledger as L  # noqa: E402

ADAPTER = Path(__file__).parent / "harbor_adapter" / "little_coder_agent.py"

MSG = {"role": "assistant", "content": [], "stopReason": "stop",
       "usage": {"input": 5, "output": 1, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 6}}


def test_each_turn_end_appends_one_flushed_jsonl_line(tmp_path):
    path = tmp_path / "turns.jsonl"
    logs = []
    times = iter([10.0, 11.0, 20.0, 21.0])
    sink = L.make_ledger_sink(path, log=logs.append, clock=lambda: next(times))
    L.sink_on_event(sink, {"type": "turn_start"})
    L.sink_on_event(sink, {"type": "turn_end", "message": MSG, "toolResults": []})
    # Read before close: run_watch tails this file while the trial runs.
    assert [json.loads(line)["turn"] for line in path.read_text().splitlines()] == [1]
    L.sink_on_event(sink, {"type": "turn_start"})
    L.sink_on_event(sink, {"type": "turn_end", "message": MSG, "toolResults": []})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["turn"] for r in rows] == [1, 2]
    assert rows[1]["ts_end"] == 21.0
    L.sink_close(sink)
    assert logs == []


def test_a_ledger_that_raises_never_reaches_the_trial_and_logs_once(tmp_path, monkeypatch):
    sink = L.LedgerSink(path=tmp_path / "turns.jsonl", log=None, clock=lambda: 1.0)
    logs = []
    sink.log = logs.append

    def boom(state, ev, now):
        raise KeyError("bad event")

    monkeypatch.setattr(L, "observe", boom)
    for _ in range(3):
        L.sink_on_event(sink, {"type": "turn_end"})
    assert len(logs) == 1 and "KeyError" in logs[0]
    assert L.sink_metadata(sink)["ledger_errors"] == 3


def test_an_unwritable_path_is_logged_once_and_the_summary_still_works(tmp_path):
    path = tmp_path / "is_a_dir"
    path.mkdir()
    logs = []
    sink = L.make_ledger_sink(path, log=logs.append, clock=lambda: 1.0)
    for _ in range(2):
        L.sink_on_event(sink, {"type": "turn_start"})
        L.sink_on_event(sink, {"type": "turn_end", "message": MSG, "toolResults": []})
    assert len(logs) == 1
    meta = L.sink_metadata(sink)
    assert (meta["n_turns"], meta["ledger_errors"]) == (2, 2)
    L.sink_close(sink)


def test_no_path_means_no_file_but_still_a_summary(tmp_path):
    sink = L.make_ledger_sink(None, log=lambda m: None, clock=lambda: 1.0)
    L.sink_on_event(sink, {"type": "turn_start"})
    L.sink_on_event(sink, {"type": "turn_end", "message": MSG, "toolResults": []})
    assert L.sink_metadata(sink)["n_turns"] == 1
    assert list(tmp_path.iterdir()) == []


def test_a_summary_failure_degrades_to_the_error_count(monkeypatch):
    sink = L.LedgerSink(path=None, log=lambda m: 1 / 0, clock=lambda: 1.0)
    monkeypatch.setattr(L, "summarize", lambda state: 1 / 0)
    assert L.sink_metadata(sink) == {"ledger_errors": 1}


def test_close_is_safe_before_any_write_and_twice():
    sink = L.LedgerSink(path=None, log=lambda m: None, clock=lambda: 1.0)
    L.sink_close(sink)
    L.sink_close(sink)
    assert sink.fh is None and sink.errors == 0


def test_the_harbor_adapter_feeds_every_event_to_the_ledger_and_records_its_summary():
    """Textual, like test_adapter_error_retry.py's adapter check: importing the
    adapter needs harbor, which CI does not install."""
    source = ADAPTER.read_text()
    assert "from turn_ledger import" in source
    assert "make_ledger_sink(" in source and "LEDGER_FILENAME" in source
    assert "sink_on_event(ledger, ev)" in source
    # First thing in on_event, ahead of the live-log early return.
    assert source.index("sink_on_event(ledger, ev)") < source.index("if live_log_fh is None:")
    assert '"turn_ledger": sink_metadata(ledger)' in source
    assert "sink_close(ledger)" in source
