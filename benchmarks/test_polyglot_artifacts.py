"""Artifact hygiene: stale files must not survive a rerun, and each attempt's
snapshot must reflect that attempt.

Regression cover for a bug that produced a wrong review conclusion: LOG_ROOT is
deterministic and was never purged, so a one-attempt rerun left the PREVIOUS
run's trajectory_2/workdir_2 beside a fresh trajectory_1. Comparing that pair
looks like comparing two attempts of one run. It is not.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aider_polyglot as AP  # noqa: E402


def test_purge_removes_prior_run_artifacts(tmp_path):
    log_dir = tmp_path / "python" / "some-exercise"
    log_dir.mkdir(parents=True)
    (log_dir / "trajectory_1.json").write_text("{}")
    (log_dir / "trajectory_2.json").write_text("{}")
    (log_dir / "final_output.txt").write_text("old")
    (log_dir / "final_output_2.txt").write_text("old")
    (log_dir / "workdir_1").mkdir()
    (log_dir / "workdir_2").mkdir()
    (log_dir / "workdir_2" / "stale.py").write_text("stale")
    keep = log_dir / "notes.md"
    keep.write_text("not ours")

    AP._purge_log_dir(log_dir)

    assert not list(log_dir.glob("trajectory_*"))
    assert not list(log_dir.glob("workdir_*"))
    assert not list(log_dir.glob("final_output*"))
    assert keep.exists(), "purge must only remove harness-owned artifacts"


def test_purge_is_safe_on_empty_dir(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    AP._purge_log_dir(d)  # must not raise


def test_snapshots_of_two_attempts_differ(tmp_path):
    """The B3 regression: each attempt's workdir must capture its own state."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    work = tmp_path / "work"
    work.mkdir()

    class R:
        agent_ended = True
        turn_count = 1
        compaction_events = 0
        assistant_text = "a"
        tool_calls = []

    (work / "solution.py").write_text("attempt one")
    AP._dump_trajectory(log_dir, "1", R(), work)
    (work / "solution.py").write_text("attempt two -- different")
    AP._dump_trajectory(log_dir, "2", R(), work)

    one = (log_dir / "workdir_1" / "solution.py").read_text()
    two = (log_dir / "workdir_2" / "solution.py").read_text()
    assert one == "attempt one"
    assert two == "attempt two -- different"
    assert one != two


class _FakeRpc:
    """Stands in for PiRpc: each prompt mutates the worktree, so the ordering
    of snapshot vs prompt is observable."""

    def __init__(self, *a, **kw):
        self.cwd = Path(kw["cwd"])
        # Session ids are "poly-<lang>-<ex>-attempt<i>" and each attempt gets
        # its own PiRpc, so derive which attempt this instance represents
        # from the id rather than from a per-instance call counter.
        self.n = int(kw["session_id"].rsplit("attempt", 1)[-1])
        self._notifications = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def notifications(self):
        return []

    def prompt_and_collect(self, message, timeout=900):
        (self.cwd / "solution.py").write_text(f"written by attempt {self.n}")
        self._notifications.append(
            {"message": f"skill-inject: +1 [bash]  # attempt {self.n}", "notifyType": "info"}
        )

        class R:
            agent_ended = True
            turn_count = 1
            compaction_events = 0
            assistant_text = f"attempt {self.n}"
            tool_calls = []
        return R()

    def notifications(self):
        return list(self._notifications)


def test_attempt1_snapshot_predates_the_retry_prompt(tmp_path, monkeypatch):
    """The B3 regression, at the call site where it actually lived.

    Both _dump_trajectory calls used to sit after the `with PiRpc(...)` block,
    so attempt 1's snapshot captured the post-retry tree. Driving _run_exercise
    with a fake agent that rewrites the file on every prompt makes the ordering
    observable: if attempt 1 is snapshotted late, workdir_1 holds attempt 2's
    text.
    """
    src = tmp_path / "practice" / "ex"
    src.mkdir(parents=True)
    (src / "ex.py").write_text("stub")
    (src / "ex_test.py").write_text("test")

    def prepare(s, w):
        AP._copy_exercise(s, w)
        return [w / "ex.py"], [w / "ex_test.py"]

    monkeypatch.setitem(AP.LANG_DESCRIPTORS, "faker", {
        "practice_dir": tmp_path / "practice",
        "prepare": prepare,
        "run_tests": lambda work, timeout: (False, "boom"),   # always fail -> retry
        "syntax_hint": "",
        "timeout_s": 5,
    })
    monkeypatch.setattr(AP, "PiRpc", _FakeRpc)
    monkeypatch.setattr(AP, "LOG_ROOT", tmp_path / "logs")

    AP._run_exercise("faker", "ex", "fake/model", agent="pi", verbose=False, retry=True)

    log_dir = tmp_path / "logs" / "pi" / "faker" / "ex"
    one = (log_dir / "workdir_1" / "solution.py").read_text()
    two = (log_dir / "workdir_2" / "solution.py").read_text()
    assert one == "written by attempt 1", f"attempt 1 snapshot is stale: {one!r}"
    assert two == "written by attempt 2"
    assert (log_dir / "final_output_1.txt").exists()
    assert (log_dir / "final_output_2.txt").exists()


def test_log_dir_namespaced_by_agent(tmp_path, monkeypatch):
    """Two agents run against the same exercise name -- pi and codex must
    not clobber each other's raw diagnostic artifacts. Regression cover for
    an un-namespaced log_dir: introducing --agent codex without this would
    have silently overwritten whichever agent ran the same exercise name
    first, the same class of bug test_snapshots_of_two_attempts_differ
    guards for within one agent's own attempts."""
    src = tmp_path / "practice" / "ex"
    src.mkdir(parents=True)
    (src / "ex.py").write_text("stub")

    monkeypatch.setitem(AP.LANG_DESCRIPTORS, "faker", {
        "practice_dir": tmp_path / "practice",
        "prepare": lambda s, w: (AP._copy_exercise(s, w), ([w / "ex.py"], []))[1],
        "run_tests": lambda where, timeout: (True, "ok"),
        "syntax_hint": "",
        "timeout_s": 5,
    })
    monkeypatch.setattr(AP, "PiRpc", _FakeRpc)
    monkeypatch.setattr(
        AP, "_run_codex_turn",
        lambda model, work, prompt, session_id, log_dir, attempt_name: (
            AP.PromptResult(turn_count=1, agent_ended=True, stop_reason="agent_end",
                             assistant_text="codex did it"),
            "fake-session-id",
        ))
    monkeypatch.setattr(AP, "LOG_ROOT", tmp_path / "logs")

    AP._run_exercise("faker", "ex", "fake/model", agent="pi", verbose=False, retry=False)
    AP._run_exercise("faker", "ex", "fake/model", agent="codex", verbose=False, retry=False)

    pi_traj = tmp_path / "logs" / "pi" / "faker" / "ex" / "trajectory_1.txt"
    codex_traj = tmp_path / "logs" / "codex" / "faker" / "ex" / "trajectory_1.txt"
    assert pi_traj.exists() and codex_traj.exists()
    assert "codex did it" not in pi_traj.read_text()
    assert "codex did it" in codex_traj.read_text()


class _FakeRpcWithCompactions(_FakeRpc):
    """Same as _FakeRpc, but reports a nonzero, attempt-dependent
    compaction_events count -- lets a test observe compaction_total actually
    summing across every attempt, the same way turn_total already does."""

    def prompt_and_collect(self, message, timeout=900):
        (self.cwd / "solution.py").write_text(f"written by attempt {self.n}")

        class R:
            agent_ended = True
            turn_count = 1
            compaction_events = self.n  # attempt 1 -> 1, attempt 2 -> 2, ...
            assistant_text = f"attempt {self.n}"
            tool_calls = []
        return R()


def test_run_exercise_sums_compaction_events_across_every_attempt(tmp_path, monkeypatch):
    """compaction_total must accumulate across attempts the way turn_total
    already does; without it a candidate's context-bloat symptom never
    reaches the results record, let alone the live GEPA loop reading it."""
    src = tmp_path / "practice" / "ex"
    src.mkdir(parents=True)
    (src / "ex.py").write_text("stub")
    (src / "ex_test.py").write_text("test")

    def prepare(s, w):
        AP._copy_exercise(s, w)
        return [w / "ex.py"], [w / "ex_test.py"]

    monkeypatch.setitem(AP.LANG_DESCRIPTORS, "faker", {
        "practice_dir": tmp_path / "practice",
        "prepare": prepare,
        "run_tests": lambda work, timeout: (False, "boom"),  # always fail -> retry
        "syntax_hint": "",
        "timeout_s": 5,
    })
    monkeypatch.setattr(AP, "PiRpc", _FakeRpcWithCompactions)
    monkeypatch.setattr(AP, "LOG_ROOT", tmp_path / "logs")

    record = AP._run_exercise("faker", "ex", "fake/model", agent="pi", verbose=False, retry=True)
    assert record["compaction_total"] == 1 + 2  # attempt 1's 1 + attempt 2's 2


class _FakeRpcWithLessons(_FakeRpc):
    """Same as _FakeRpc, but attempt 2+ includes a LESSON: line in its
    prose (mirroring a real model following the retry prompt's new
    instruction) -- attempt 1 never gets asked, so it has none, matching
    real behavior."""

    def prompt_and_collect(self, message, timeout=900):
        (self.cwd / "solution.py").write_text(f"written by attempt {self.n}")

        lesson_line = f"\nLESSON: needed guidance {self.n}\n" if self.n > 1 else ""

        class R:
            agent_ended = True
            turn_count = 1
            compaction_events = 0
            assistant_text = f"Some prose about attempt {self.n}.{lesson_line}More prose."
            tool_calls = []
        return R()


def test_run_exercise_captures_lesson_from_a_retried_attempts_response(tmp_path, monkeypatch):
    """Real gap this closes: aider_polyglot.py's retry loop fed the next
    attempt raw test output but never solicited or captured an explicit
    self-reflection -- Reflexion's actual validated technique (verbal
    self-reflection between retry attempts) was unavailable here."""
    src = tmp_path / "practice" / "ex"
    src.mkdir(parents=True)
    (src / "ex.py").write_text("stub")
    (src / "ex_test.py").write_text("test")

    def prepare(s, w):
        AP._copy_exercise(s, w)
        return [w / "ex.py"], [w / "ex_test.py"]

    monkeypatch.setitem(AP.LANG_DESCRIPTORS, "faker", {
        "practice_dir": tmp_path / "practice",
        "prepare": prepare,
        "run_tests": lambda work, timeout: (False, "boom"),  # always fail -> retry
        "syntax_hint": "",
        "timeout_s": 5,
    })
    monkeypatch.setattr(AP, "PiRpc", _FakeRpcWithLessons)
    monkeypatch.setattr(AP, "LOG_ROOT", tmp_path / "logs")

    record = AP._run_exercise("faker", "ex", "fake/model", agent="pi", verbose=False, retry=True)
    # Attempt 1 was never asked (no retry prompt exists for the first
    # attempt), so only attempt 2's lesson is captured.
    assert record["lessons"] == ["needed guidance 2"]


class _FakeRpcWithLessonOnAttemptOneOnly(_FakeRpc):
    """Simulates a model emitting an unrelated but coincidentally
    LESSON:-matching line on attempt 1 (e.g. a `# LESSON: ...` code
    comment) even though attempt 1's prompt never asks for one."""

    def prompt_and_collect(self, message, timeout=900):
        (self.cwd / "solution.py").write_text(f"written by attempt {self.n}")

        lesson_line = "\nLESSON: unrelated coincidental match\n" if self.n == 1 else ""

        class R:
            agent_ended = True
            turn_count = 1
            compaction_events = 0
            assistant_text = f"Some prose about attempt {self.n}.{lesson_line}More prose."
            tool_calls = []
        return R()


def test_run_exercise_never_captures_a_lesson_from_attempt_one_even_if_a_line_matches(tmp_path, monkeypatch):
    """Extraction is gated on i > 1 explicitly, not just on attempt 1's
    prompt never ASKING for a LESSON: line: assistant_text carries ALL of
    the model's text on every attempt, so a coincidentally matching line
    (a code comment, say) would otherwise be picked up."""
    src = tmp_path / "practice" / "ex"
    src.mkdir(parents=True)
    (src / "ex.py").write_text("stub")
    (src / "ex_test.py").write_text("test")

    def prepare(s, w):
        AP._copy_exercise(s, w)
        return [w / "ex.py"], [w / "ex_test.py"]

    monkeypatch.setitem(AP.LANG_DESCRIPTORS, "faker", {
        "practice_dir": tmp_path / "practice",
        "prepare": prepare,
        "run_tests": lambda work, timeout: (False, "boom"),  # always fail -> retry
        "syntax_hint": "",
        "timeout_s": 5,
    })
    monkeypatch.setattr(AP, "PiRpc", _FakeRpcWithLessonOnAttemptOneOnly)
    monkeypatch.setattr(AP, "LOG_ROOT", tmp_path / "logs")

    record = AP._run_exercise("faker", "ex", "fake/model", agent="pi", verbose=False, retry=True)
    assert record["lessons"] == []


def test_run_exercise_lessons_is_empty_when_no_lesson_line_present(tmp_path, monkeypatch):
    src = tmp_path / "practice" / "ex"
    src.mkdir(parents=True)
    (src / "ex.py").write_text("stub")
    (src / "ex_test.py").write_text("test")

    def prepare(s, w):
        AP._copy_exercise(s, w)
        return [w / "ex.py"], [w / "ex_test.py"]

    monkeypatch.setitem(AP.LANG_DESCRIPTORS, "faker", {
        "practice_dir": tmp_path / "practice",
        "prepare": prepare,
        "run_tests": lambda work, timeout: (False, "boom"),
        "syntax_hint": "",
        "timeout_s": 5,
    })
    monkeypatch.setattr(AP, "PiRpc", _FakeRpc)  # never emits a LESSON: line
    monkeypatch.setattr(AP, "LOG_ROOT", tmp_path / "logs")

    record = AP._run_exercise("faker", "ex", "fake/model", agent="pi", verbose=False, retry=True)
    assert record["lessons"] == []


def test_run_exercise_captures_one_lesson_per_retried_attempt(tmp_path, monkeypatch):
    src = tmp_path / "practice" / "ex"
    src.mkdir(parents=True)
    (src / "ex.py").write_text("stub")
    (src / "ex_test.py").write_text("test")

    def prepare(s, w):
        AP._copy_exercise(s, w)
        return [w / "ex.py"], [w / "ex_test.py"]

    monkeypatch.setitem(AP.LANG_DESCRIPTORS, "faker", {
        "practice_dir": tmp_path / "practice",
        "prepare": prepare,
        "run_tests": lambda work, timeout: (False, "boom"),
        "syntax_hint": "",
        "timeout_s": 5,
    })
    monkeypatch.setattr(AP, "PiRpc", _FakeRpcWithLessons)
    monkeypatch.setattr(AP, "LOG_ROOT", tmp_path / "logs")

    record = AP._run_exercise("faker", "ex", "fake/model", agent="pi", verbose=False, retry=True,
                               max_attempts=3)
    assert record["lessons"] == ["needed guidance 2", "needed guidance 3"]


class _FakeRpcWithDecoratedLesson(_FakeRpc):
    """Attempt 2's LESSON: line is wrapped in common markdown decoration a
    model might reasonably use -- a bullet, heading, or leading bold
    wrapper, none of which the bare regex matches."""

    def prompt_and_collect(self, message, timeout=900):
        (self.cwd / "solution.py").write_text(f"written by attempt {self.n}")
        lesson_line = f"\n- **LESSON:** needed guidance {self.n}\n" if self.n > 1 else ""

        class R:
            agent_ended = True
            turn_count = 1
            compaction_events = 0
            assistant_text = f"Some prose about attempt {self.n}.{lesson_line}More prose."
            tool_calls = []
        return R()


def test_run_exercise_captures_a_lesson_wrapped_in_markdown_decoration(tmp_path, monkeypatch):
    src = tmp_path / "practice" / "ex"
    src.mkdir(parents=True)
    (src / "ex.py").write_text("stub")
    (src / "ex_test.py").write_text("test")

    def prepare(s, w):
        AP._copy_exercise(s, w)
        return [w / "ex.py"], [w / "ex_test.py"]

    monkeypatch.setitem(AP.LANG_DESCRIPTORS, "faker", {
        "practice_dir": tmp_path / "practice",
        "prepare": prepare,
        "run_tests": lambda work, timeout: (False, "boom"),
        "syntax_hint": "",
        "timeout_s": 5,
    })
    monkeypatch.setattr(AP, "PiRpc", _FakeRpcWithDecoratedLesson)
    monkeypatch.setattr(AP, "LOG_ROOT", tmp_path / "logs")

    record = AP._run_exercise("faker", "ex", "fake/model", agent="pi", verbose=False, retry=True)
    assert record["lessons"] == ["needed guidance 2"]


class _FakeRpcWithLessonContentStartingBold(_FakeRpc):
    """Attempt 2's LESSON content itself starts with its own bold markup,
    which _LESSON_RE must leave intact -- it may only strip a label
    wrapper's own closing marker."""

    def prompt_and_collect(self, message, timeout=900):
        (self.cwd / "solution.py").write_text(f"written by attempt {self.n}")
        lesson_line = f"\nLESSON: **Refactor X** now\n" if self.n > 1 else ""

        class R:
            agent_ended = True
            turn_count = 1
            compaction_events = 0
            assistant_text = f"Some prose about attempt {self.n}.{lesson_line}More prose."
            tool_calls = []
        return R()


def test_run_exercise_preserves_the_lessons_own_opening_bold_markup(tmp_path, monkeypatch):
    src = tmp_path / "practice" / "ex"
    src.mkdir(parents=True)
    (src / "ex.py").write_text("stub")
    (src / "ex_test.py").write_text("test")

    def prepare(s, w):
        AP._copy_exercise(s, w)
        return [w / "ex.py"], [w / "ex_test.py"]

    monkeypatch.setitem(AP.LANG_DESCRIPTORS, "faker", {
        "practice_dir": tmp_path / "practice",
        "prepare": prepare,
        "run_tests": lambda work, timeout: (False, "boom"),
        "syntax_hint": "",
        "timeout_s": 5,
    })
    monkeypatch.setattr(AP, "PiRpc", _FakeRpcWithLessonContentStartingBold)
    monkeypatch.setattr(AP, "LOG_ROOT", tmp_path / "logs")

    record = AP._run_exercise("faker", "ex", "fake/model", agent="pi", verbose=False, retry=True)
    assert record["lessons"] == ["**Refactor X** now"]


class _FakeRpcWithHugeLesson(_FakeRpc):
    """Same as _FakeRpc, but attempt 2's LESSON: line is far longer than any
    real one-sentence answer should be -- every other free-text field on
    this path is capped (out[-4000:], TRAJECTORY_TEXT_CHARS); this one used
    to be the exception."""

    def prompt_and_collect(self, message, timeout=900):
        (self.cwd / "solution.py").write_text(f"written by attempt {self.n}")
        huge = "x" * 10_000
        lesson_line = f"\nLESSON: {huge}\n" if self.n > 1 else ""

        class R:
            agent_ended = True
            turn_count = 1
            compaction_events = 0
            assistant_text = f"Some prose about attempt {self.n}.{lesson_line}More prose."
            tool_calls = []
        return R()


def test_run_exercise_caps_an_unreasonably_long_lesson(tmp_path, monkeypatch):
    src = tmp_path / "practice" / "ex"
    src.mkdir(parents=True)
    (src / "ex.py").write_text("stub")
    (src / "ex_test.py").write_text("test")

    def prepare(s, w):
        AP._copy_exercise(s, w)
        return [w / "ex.py"], [w / "ex_test.py"]

    monkeypatch.setitem(AP.LANG_DESCRIPTORS, "faker", {
        "practice_dir": tmp_path / "practice",
        "prepare": prepare,
        "run_tests": lambda work, timeout: (False, "boom"),
        "syntax_hint": "",
        "timeout_s": 5,
    })
    monkeypatch.setattr(AP, "PiRpc", _FakeRpcWithHugeLesson)
    monkeypatch.setattr(AP, "LOG_ROOT", tmp_path / "logs")

    record = AP._run_exercise("faker", "ex", "fake/model", agent="pi", verbose=False, retry=True)
    assert len(record["lessons"]) == 1
    assert len(record["lessons"][0]) == AP.LESSON_MAX_CHARS


def test_run_id_is_stable_within_a_process():
    assert AP.RUN_ID and AP.RUN_ID == AP.RUN_ID


def test_dump_trajectory_notifications_default_to_empty_list(tmp_path):
    """Backward compat: existing callers (this file's own R() fixture class)
    don't pass notifications -- must not raise, payload gets []."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    class R:
        agent_ended = True
        turn_count = 1
        compaction_events = 0
        assistant_text = "a"
        tool_calls = []

    AP._dump_trajectory(log_dir, "1", R())
    payload = json.loads((log_dir / "trajectory_1.json").read_text())
    assert payload["notifications"] == []
    # Same backward-compat guarantee for non_text_deltas: R() above has no
    # such attribute at all (an older PromptResult, or any caller that
    # hasn't been touched by the rpc_client.py diagnostic-capture change),
    # must not raise, payload gets [].
    assert payload["non_text_deltas"] == []


def test_dump_trajectory_persists_non_text_deltas_when_present(tmp_path):
    """PromptResult.non_text_deltas (rpc_client.py) -- diagnostic capture of
    any assistantMessageEvent whose type isn't "text_delta" (a candidate
    reasoning/thinking-content stream) -- must survive into the persisted
    trajectory so it can actually be inspected after a real run."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    class R:
        agent_ended = True
        turn_count = 1
        compaction_events = 0
        assistant_text = "a"
        tool_calls = []
        non_text_deltas = [{"type": "thinking_delta", "delta": "reasoning..."}]

    AP._dump_trajectory(log_dir, "1", R())
    payload = json.loads((log_dir / "trajectory_1.json").read_text())
    assert payload["non_text_deltas"] == [{"type": "thinking_delta", "delta": "reasoning..."}]


def test_cap_non_text_deltas_returns_input_unchanged_when_under_budget():
    deltas = [{"type": "thinking_delta", "delta": f"chunk {i}"} for i in range(5)]
    assert AP._cap_non_text_deltas(deltas, char_budget=10_000) == deltas


def test_cap_non_text_deltas_keeps_head_and_tail_drops_middle():
    # Each entry serializes to ~55 raw JSON chars (verified:
    # json.dumps({"type": "thinking_delta", "delta": "chunk number 000"})
    # == 55); budget 400 splits to 200 head/200 tail, i.e. ~3 entries
    # survive on each end out of 100.
    deltas = [{"type": "thinking_delta", "delta": f"chunk number {i:03d}"} for i in range(100)]
    result = AP._cap_non_text_deltas(deltas, char_budget=400)
    assert result[0] == deltas[0]
    assert result[-1] == deltas[-1]
    marker = next(d for d in result if d.get("type") == "_omitted")
    assert marker["omitted_count"] > 0
    # Head, marker, tail -- middle entries genuinely gone, not just hidden.
    assert marker["omitted_count"] == 100 - (len(result) - 1)
    kept_indices = [deltas.index(d) for d in result if d is not marker]
    assert kept_indices == sorted(kept_indices)  # order preserved, no shuffling


def test_cap_non_text_deltas_does_not_crash_on_a_single_oversized_entry():
    """Asserts actual boundedness and dropping, not just
    isinstance(result, list) -- that stays true even if the huge entry is
    echoed back unbounded, so it would pass through a regression that
    stopped capping oversized entries."""
    huge = {"type": "toolcall_delta", "delta": "x" * 1_000_000}
    result = AP._cap_non_text_deltas([huge, {"type": "thinking_delta", "delta": "short"}], char_budget=1_000)
    assert len(json.dumps(result, default=str)) < 1_000  # nowhere near the huge entry's own size
    assert huge not in result  # actually dropped, not just left in unbounded
    assert any(d.get("type") == "_omitted" for d in result)


def test_cap_non_text_deltas_skips_an_oversized_tail_entry_instead_of_stopping():
    """A backward walk that stopped at the FIRST tail entry too big to fit
    (a huge toolcall_delta, say) would discard every smaller,
    budget-fitting entry behind it, including genuinely recent reasoning.
    Padded so
    NEITHER small entry is claimed by the head slice -- the only way either
    survives is via the tail walk actually skipping past huge_middle."""
    huge_at_start = {"type": "toolcall_delta", "delta": "x" * 500_010}  # too big for head_budget alone
    small_before = {"type": "thinking_delta", "delta": "recent reasoning before the huge entry"}
    huge_middle = {"type": "toolcall_delta", "delta": "y" * 500_020}
    small_after = {"type": "thinking_delta", "delta": "final reasoning at the true end"}
    result = AP._cap_non_text_deltas(
        [huge_at_start, small_before, huge_middle, small_after], char_budget=1_000,
    )
    assert small_before in result  # would have been dropped by the old stop-at-first-miss logic
    assert small_after in result
    assert huge_at_start not in result
    assert huge_middle not in result
    # Order preserved despite the internal skip.
    assert result.index(small_before) < result.index(small_after)


def test_dump_trajectory_no_longer_drops_the_tail_of_a_long_but_small_reasoning_stream(tmp_path):
    """Regression for the exact bug fixed here: the OLD policy was a flat
    first-200-entries cutoff BY COUNT, so a 250-chunk reasoning stream lost
    its last 50 chunks outright -- including whatever reasoning happened
    right before the model's final decision, exactly what
    live_eval.py's _reasoning_excerpt_from_trajectory needs. The new
    char-budget policy keeps everything here, since the whole stream is
    tiny in bytes even at 250 entries."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    class R:
        agent_ended = True
        turn_count = 1
        compaction_events = 0
        assistant_text = "a"
        tool_calls = []
        non_text_deltas = [{"type": "thinking_delta", "delta": f"chunk {i}"} for i in range(250)]

    AP._dump_trajectory(log_dir, "1", R())
    payload = json.loads((log_dir / "trajectory_1.json").read_text())
    # Index 249 -- entirely dropped by the old [:200] cutoff -- survives now.
    assert {"type": "thinking_delta", "delta": "chunk 249"} in payload["non_text_deltas"]


def test_dump_trajectory_persists_notifications_when_given(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    class R:
        agent_ended = True
        turn_count = 1
        compaction_events = 0
        assistant_text = "a"
        tool_calls = []

    notes = [{"message": "skill-inject: +1 [bash]", "notifyType": "info"}]
    AP._dump_trajectory(log_dir, "1", R(), notifications=notes)
    payload = json.loads((log_dir / "trajectory_1.json").read_text())
    assert payload["notifications"] == notes


def test_run_exercise_notifications_are_isolated_per_attempt(tmp_path, monkeypatch):
    """Each attempt now opens a FRESH PiRpc session (see _run_exercise's own
    comment), so rpc.notifications() is already scoped to just that attempt
    -- no delta-slicing across a shared session is needed (or possible)
    anymore. This guards that attempt 2's dump doesn't somehow pick up
    attempt 1's notifications despite the two running in separate fake
    instances."""
    src = tmp_path / "practice" / "ex"
    src.mkdir(parents=True)
    (src / "ex.py").write_text("stub")
    (src / "ex_test.py").write_text("test")

    def prepare(s, w):
        AP._copy_exercise(s, w)
        return [w / "ex.py"], [w / "ex_test.py"]

    monkeypatch.setitem(AP.LANG_DESCRIPTORS, "faker2", {
        "practice_dir": tmp_path / "practice",
        "prepare": prepare,
        "run_tests": lambda work, timeout: (False, "boom"),   # always fail -> retry
        "syntax_hint": "",
        "timeout_s": 5,
    })
    monkeypatch.setattr(AP, "PiRpc", _FakeRpc)
    monkeypatch.setattr(AP, "LOG_ROOT", tmp_path / "logs")

    AP._run_exercise("faker2", "ex", "fake/model", verbose=False, retry=True)

    log_dir = tmp_path / "logs" / "pi" / "faker2" / "ex"
    payload_1 = json.loads((log_dir / "trajectory_1.json").read_text())
    payload_2 = json.loads((log_dir / "trajectory_2.json").read_text())

    assert len(payload_1["notifications"]) == 1
    assert "attempt 1" in payload_1["notifications"][0]["message"]
    assert len(payload_2["notifications"]) == 1
    assert "attempt 2" in payload_2["notifications"][0]["message"]


class _FakeRpcWithNotifications(_FakeRpc):
    """Same as _FakeRpc, but simulates the thinking-budget extension firing --
    the ctx.ui.notify event that was invisible in every trajectory before
    this was wired up -- it took a live manual re-run with rpc.notifications()
    to discover this on a real `bowling` failure."""

    def notifications(self):
        return [
            {"message": "little-coder scaffold loaded", "notifyType": "info"},
            {"message": "harness intervention: the model has thought long "
                        "enough -- forcing it to start implementing.", "notifyType": "info"},
        ]


def test_notifications_are_persisted_in_the_trajectory(tmp_path, monkeypatch):
    """A thinking-budget intervention (or any ctx.ui.notify event) must land
    in trajectory_<n>.json/.txt -- previously _dump_trajectory never received
    or wrote them at all, so an attempt that read files and stopped looked
    identical whether the model chose to stop or the harness force-aborted
    its thinking."""
    src = tmp_path / "practice" / "ex"
    src.mkdir(parents=True)
    (src / "ex.py").write_text("stub")
    (src / "ex_test.py").write_text("test")

    def prepare(s, w):
        AP._copy_exercise(s, w)
        return [w / "ex.py"], [w / "ex_test.py"]

    monkeypatch.setitem(AP.LANG_DESCRIPTORS, "faker", {
        "practice_dir": tmp_path / "practice",
        "prepare": prepare,
        "run_tests": lambda work, timeout: (True, "ok"),
        "syntax_hint": "",
        "timeout_s": 5,
    })
    monkeypatch.setattr(AP, "PiRpc", _FakeRpcWithNotifications)
    monkeypatch.setattr(AP, "LOG_ROOT", tmp_path / "logs")

    AP._run_exercise("faker", "ex", "fake/model", agent="pi", verbose=False, retry=False)

    log_dir = tmp_path / "logs" / "pi" / "faker" / "ex"
    payload = json.loads((log_dir / "trajectory_1.json").read_text())
    assert any("thought long enough" in n["message"] for n in payload["notifications"])
    assert "harness intervention" in (log_dir / "trajectory_1.txt").read_text()
