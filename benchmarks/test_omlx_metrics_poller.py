"""Tests for omlx_metrics_poller.

Nothing here touches a live omlx server, a real `sysctl`/`vm_stat`, or either
clock: the subprocess runner is injected and both the wall-clock and the
monotonic reading are scripted, so the 5-minute persistence window is exercised
without sleeping through it -- and a wall-clock jump can be staged at will.
"""
from __future__ import annotations

import http.client
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import omlx_metrics_poller as P  # noqa: E402

SWAPUSAGE = "vm.swapusage: total = 3072.00M  used = 1626.00M  free = 1446.00M  (encrypted)\n"
VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                  2461872.
Pages active:                                 707800.
Pages inactive:                               698539.
Pages speculative:                              7891.
"""
VM_STAT_INTEL = """Mach Virtual Memory Statistics: (page size of 4096 bytes)
Pages free:                                    12345.
Pages active:                                 707800.
"""


class _ClockExhausted(BaseException):
    """Ends poll()'s otherwise-infinite loop once a test's scripted times run out.

    BaseException, not Exception, so poll's per-sample error guard doesn't swallow it.
    """


def _clock(*times):
    remaining = iter(times)

    def clock():
        try:
            return next(remaining)
        except StopIteration:
            raise _ClockExhausted

    return clock


def _runner(swap=SWAPUSAGE, vm=VM_STAT):
    def run(argv):
        if argv[0] == "sysctl":
            return swap
        if argv[0] == "vm_stat":
            return vm
        raise AssertionError(f"unexpected command {argv}")

    return run


def test_swap_used_is_read_from_the_used_field_not_total_or_free():
    assert P.parse_swap_used_mb(SWAPUSAGE) == 1626.0


def test_swap_unit_suffix_is_honoured():
    assert P.parse_swap_used_mb("vm.swapusage: total = 8.00G  used = 2.50G  free = 5.50G") == 2560.0


def test_swap_parse_rejects_unrecognised_output():
    with pytest.raises(ValueError):
        P.parse_swap_used_mb("vm.swapusage: nothing useful here")


def test_page_size_comes_from_vm_stats_own_header():
    assert P.parse_vm_stat(VM_STAT) == (2461872, 16384)
    assert P.parse_vm_stat(VM_STAT_INTEL) == (12345, 4096)


def test_vm_stat_parse_rejects_output_without_a_free_line():
    with pytest.raises(ValueError):
        P.parse_vm_stat("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages active: 1.\n")


def test_sample_records_host_fields_and_null_status_when_server_is_down(monkeypatch):
    monkeypatch.setattr(P, "fetch_status", lambda url, **kw: None)
    row = P.sample(1234.5, run=_runner())
    assert row == {
        "ts": 1234.5,
        "api_status": None,
        "swap_used_mb": 1626.0,
        "pages_free": 2461872,
        "page_size_bytes": 16384,
    }


def test_fetch_status_returns_none_rather_than_raising_when_unreachable(monkeypatch):
    def unreachable(url, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", unreachable)
    assert P.fetch_status("http://127.0.0.1:1/api/status", timeout=0.5) is None


def test_fetch_status_survives_a_half_dead_server_hanging_up_mid_response(monkeypatch):
    """HTTPException is neither OSError nor ValueError, so it would escape and lose the row."""
    def hangs_up(url, timeout=None):
        raise http.client.IncompleteRead(b"{\"active_re")

    monkeypatch.setattr(P.urllib.request, "urlopen", hangs_up)
    assert P.fetch_status("http://127.0.0.1:8000/api/status") is None


def test_a_status_failure_still_leaves_the_swap_figure_the_tripwire_needs(monkeypatch, tmp_path):
    def hangs_up(url, timeout=None):
        raise http.client.BadStatusLine("")

    monkeypatch.setattr(P.urllib.request, "urlopen", hangs_up)
    monkeypatch.setattr(P, "_sleep_until", lambda *a, **kw: None)

    with pytest.raises(_ClockExhausted):
        P.poll(tmp_path, interval=1.0, run=_runner(), clock=_clock(100.0), monotonic_clock=_clock(0.0))

    rows = [json.loads(line) for line in (tmp_path / P.METRICS_FILENAME).read_text().splitlines()]
    assert [(r["api_status"], r["swap_used_mb"]) for r in rows] == [(None, 1626.0)]


def test_tripwire_first_sample_only_sets_the_baseline():
    tw = P.SwapTripwire()
    assert tw.observe(0.0, 9000.0) is False
    assert tw.baseline_mb == 9000.0


def test_tripwire_ignores_elevation_shorter_than_the_hold_window():
    tw = P.SwapTripwire()
    tw.observe(0.0, 1000.0)
    assert tw.observe(30.0, 4000.0) is False
    assert tw.observe(300.0, 4000.0) is False


def test_tripwire_fires_once_the_hold_window_is_exceeded():
    tw = P.SwapTripwire()
    tw.observe(0.0, 1000.0)
    tw.observe(30.0, 4000.0)
    assert tw.observe(331.0, 4000.0) is True


def test_tripwire_fires_only_once():
    tw = P.SwapTripwire()
    tw.observe(0.0, 1000.0)
    tw.observe(30.0, 4000.0)
    assert tw.observe(331.0, 4000.0) is True
    assert tw.observe(400.0, 5000.0) is False


def test_a_single_dip_below_threshold_restarts_the_hold_window():
    tw = P.SwapTripwire()
    tw.observe(0.0, 1000.0)
    tw.observe(30.0, 4000.0)
    tw.observe(200.0, 1500.0)
    tw.observe(230.0, 4000.0)
    assert tw.observe(400.0, 4000.0) is False
    assert tw.observe(600.0, 4000.0) is True


def test_exactly_at_the_threshold_is_not_elevated():
    tw = P.SwapTripwire()
    tw.observe(0.0, 1000.0)
    tw.observe(30.0, 1000.0 + P.SWAP_TRIPWIRE_MB)
    assert tw.elevated_since is None


def test_sentinel_names_the_values_that_tripped_it(tmp_path):
    path = tmp_path / P.SENTINEL_FILENAME
    P.write_sentinel(path, 1757740000.0, 1000.0, 4100.0, 300.0)
    text = path.read_text()
    assert "1000.00" in text
    assert "4100.00" in text
    assert "3100.00" in text


def test_poll_writes_one_flushed_jsonl_line_per_sample(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "fetch_status", lambda url, **kw: {"active_requests": 1})
    monkeypatch.setattr(P, "_sleep_until", lambda *a, **kw: None)

    with pytest.raises(_ClockExhausted):
        P.poll(
            tmp_path,
            interval=1.0,
            run=_runner(),
            clock=_clock(100.0, 200.0, 300.0),
            monotonic_clock=_clock(0.0, 100.0, 200.0),
        )

    rows = [json.loads(line) for line in (tmp_path / P.METRICS_FILENAME).read_text().splitlines()]
    assert [r["ts"] for r in rows] == [100.0, 200.0, 300.0]
    assert all(r["api_status"] == {"active_requests": 1} for r in rows)
    assert all(r["swap_used_mb"] == 1626.0 for r in rows)


def test_poll_keeps_going_after_a_failing_sample(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "fetch_status", lambda url, **kw: None)
    monkeypatch.setattr(P, "_sleep_until", lambda *a, **kw: None)

    calls = {"n": 0}

    def flaky(argv):
        if argv[0] == "sysctl":
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("sysctl vanished")
        return _runner()(argv)

    with pytest.raises(_ClockExhausted):
        P.poll(tmp_path, interval=1.0, run=flaky, clock=_clock(100.0, 200.0), monotonic_clock=_clock(0.0, 100.0))

    rows = (tmp_path / P.METRICS_FILENAME).read_text().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["ts"] == 200.0


def test_poll_appends_rather_than_truncating_an_existing_file(monkeypatch, tmp_path):
    (tmp_path / P.METRICS_FILENAME).write_text('{"ts": 1.0}\n')
    monkeypatch.setattr(P, "fetch_status", lambda url, **kw: None)
    monkeypatch.setattr(P, "_sleep_until", lambda *a, **kw: None)

    with pytest.raises(_ClockExhausted):
        P.poll(tmp_path, interval=1.0, run=_runner(), clock=_clock(100.0), monotonic_clock=_clock(0.0))

    assert len((tmp_path / P.METRICS_FILENAME).read_text().splitlines()) == 2


ELEVATING_SWAPS = [
    "vm.swapusage: total = 8192.00M  used = 1000.00M  free = 7192.00M",
    "vm.swapusage: total = 8192.00M  used = 5000.00M  free = 3192.00M",
    "vm.swapusage: total = 8192.00M  used = 5200.00M  free = 2992.00M",
]


def _elevating_runner():
    swaps = iter(ELEVATING_SWAPS)

    def run(argv):
        return next(swaps) if argv[0] == "sysctl" else VM_STAT

    return run


def test_poll_writes_the_sentinel_when_swap_stays_elevated(monkeypatch, tmp_path, capsys):
    """The hold window is timed off the monotonic clock, which is the one that advances here."""
    monkeypatch.setattr(P, "fetch_status", lambda url, **kw: None)
    monkeypatch.setattr(P, "_sleep_until", lambda *a, **kw: None)

    with pytest.raises(_ClockExhausted):
        P.poll(
            tmp_path,
            interval=1.0,
            run=_elevating_runner(),
            clock=_clock(1757740000.0, 1757740001.0, 1757740002.0),
            monotonic_clock=_clock(0.0, 30.0, 400.0),
        )

    sentinel = tmp_path / P.SENTINEL_FILENAME
    assert sentinel.exists()
    text = sentinel.read_text()
    assert "5200.00" in text
    # fired_at is a calendar time an operator reads, so it comes from the wall clock.
    assert "epoch 1757740002" in text
    assert "SWAP TRIPWIRE" in capsys.readouterr().out


def test_a_wall_clock_jump_does_not_move_the_tripwires_hold_window(monkeypatch, tmp_path):
    """NTP or DST can shift wall-clock time by hours mid-run; only elapsed time may count."""
    monkeypatch.setattr(P, "fetch_status", lambda url, **kw: None)
    monkeypatch.setattr(P, "_sleep_until", lambda *a, **kw: None)

    with pytest.raises(_ClockExhausted):
        P.poll(
            tmp_path,
            interval=1.0,
            run=_elevating_runner(),
            # Six hours forward between the second and third sample: timed off
            # this clock the 5-minute hold window would look long since elapsed.
            clock=_clock(1757740000.0, 1757740001.0, 1757761601.0),
            monotonic_clock=_clock(0.0, 30.0, 100.0),
        )

    assert not (tmp_path / P.SENTINEL_FILENAME).exists()


def test_an_existing_sentinel_is_not_rewritten(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "fetch_status", lambda url, **kw: None)
    monkeypatch.setattr(P, "_sleep_until", lambda *a, **kw: None)
    sentinel = tmp_path / P.SENTINEL_FILENAME
    sentinel.write_text("fired earlier, hands off\n")

    swaps = iter(["used = 1000.00M", "used = 5000.00M", "used = 5200.00M"])

    def run(argv):
        return next(swaps) if argv[0] == "sysctl" else VM_STAT

    with pytest.raises(_ClockExhausted):
        P.poll(
            tmp_path,
            interval=1.0,
            run=run,
            clock=_clock(1757740000.0, 1757740030.0, 1757740400.0),
            monotonic_clock=_clock(0.0, 30.0, 400.0),
        )

    assert sentinel.read_text() == "fired earlier, hands off\n"


def test_cli_requires_a_run_dir():
    with pytest.raises(SystemExit):
        P.main([])


@pytest.mark.parametrize("interval", ["0", "-5", "nan", "inf"])
def test_cli_rejects_a_non_positive_interval(tmp_path, interval):
    """Without a wait between samples poll() spins, hammering the server and sysctl.

    nan/inf are included because `<= 0` alone lets them through -- every
    comparison against nan is False, so a lone `<= 0` guard would still hand
    poll() a value that later raises ValueError out of time.sleep(nan).
    """
    with pytest.raises(SystemExit) as excinfo:
        P.main(["--run-dir", str(tmp_path), "--interval", interval])
    assert excinfo.value.code != 0
