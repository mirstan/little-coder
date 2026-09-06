# TDD Spec — `benchmarks/self_improve/`

Scope: build order steps 1–3 from the plan (`ingest/`, `metric.py`, `components.py` +
`apply_results.py`'s roundtrip logic) plus the `rpc_client.py` system-prompt change,
all fully test-driven. Steps 4+ (real GEPA runs) are explicitly out of scope for this
spec — they are manual smoke tests, not unit-testable.

Convention: every module below is specified as **test file first**, listing exact
test names, fixture shapes, and assertions, followed by the **minimal implementation**
that satisfies them. Write the test, watch it fail, then write just enough code to
pass it. No test in this spec should require network access, an LLM call, or the
`dspy` package to be installed for the test itself to execute (import-time guards are
noted where relevant).

---

## 0. Shared fixtures and confirmed real-world shapes

These are taken directly from the existing codebase (not invented) and every fixture
below must match them exactly:

- Skill file frontmatter (`skills/tools/bash.md`):
  ```
  ---
  name: bash-guidance
  type: tool-guidance
  target_tool: bash
  priority: 10
  token_cost: 120
  user-invocable: false
  ---
  ## `bash` Tool
  ...body...
  ```
  Delimiter is a bare `---` line, YAML block, bare `---` line, then body. No trailing
  frontmatter fields are ever quoted/multi-line in current files — a naive
  `str.split("---", 2)` on the first two occurrences is sufficient, but the impl must
  reject a file that doesn't start with `---` (i.e. AGENTS.md and PRINCIPLES.md, which
  have no frontmatter, must be handled as frontmatter-less).

- `skill-inject` notification message format (`.pi/extensions/skill-inject/index.ts:385`):
  `"skill-inject: +2 [bash,read]"` — literal prefix `skill-inject: +`, digit count,
  space, `[`, comma-joined tool names, `]`. When `selected.length === 0` the parts
  array can still contain `+research-directive` alone with no bracket group, e.g.
  `"skill-inject: +research-directive"` — the count/bracket segment is only present
  when `selected.length > 0`.

- `knowledge-inject` notification message format (`.pi/extensions/knowledge-inject/index.ts:156`):
  `"knowledge-inject: +3 [binary_search,two_pointer]"` — same shape, topics not tools,
  always has the count+bracket (no conditional segment).

- gaia per-task `notifications.txt` line format (`benchmarks/gaia.py:225-226`):
  `f"[{n.get('notifyType', 'info')}] {n.get('message', '')}\n"` → literal lines like
  `"[info] skill-inject: +2 [bash,read]"`.

- gaia per-task `tool_calls.jsonl`: one JSON object per line, fields include at least
  `name`, `args`, `result_text` (truncated to 2000 chars), `result_text_len`, `is_error`.

- gaia `result.json`: `{"model_answer": ..., "gold": ..., "correct": bool, "elapsed_s": float}`
  (`gold` only present for val-split runs, per `benchmarks/gaia.py` comments).

- gaia directory layout: `<out_dir>/<task_id>/{prompt.txt, transcript.txt,
  tool_calls.jsonl, notifications.txt, stderr.log, result.json}`, plus
  `<out_dir>/manifest.json` and `<out_dir>/results.json` (all tasks) and
  `<out_dir>/submission.jsonl`.

- aider_polyglot trajectory JSON (`benchmarks/aider_polyglot.py::_dump_trajectory`,
  confirmed against `test_polyglot_artifacts.py`'s fake `R` class): written to
  `LOG_ROOT/<lang>/<exercise>/trajectory_<n>.json` with fields `attempt`,
  `agent_ended`, `turn_count`, `compaction_events`, `assistant_text`, `tool_calls`
  (list of `{name, args, result_text, is_error}`). aider_polyglot does **not**
  persist `notifications` into the trajectory JSON today (confirmed: `_dump_trajectory`'s
  payload dict in `aider_polyglot.py` has no `notifications` key) — component-usage
  extraction for aider_polyglot must instead scan `assistant_text` is not an option
  either; see §1.2 for how this gap is handled.

- aider_polyglot `results_full_polyglot.json` (aggregate, `_run_exercise`'s return
  value written per exercise under `results["exercises"][f"{lang}/{name}"]`):
  `{"run_id", "status" (pass_1/pass_2/fail/fail_timeout/empty_response/error),
  "stop_reason_1", "stop_reason_2", "elapsed_s", "turn_count", "scoring_params"}`.
  **SCHEMA EVOLVED (merged from `dev`, generalizing retries from a hardcoded
  2 attempts to `--max-attempts` N attempts)**: `stop_reason_1`/`stop_reason_2`
  became a `stop_reasons` list (one entry per attempt that ran), and `status`
  can now be `pass_N` for any N, not just `pass_1`/`pass_2`. A hardcoded
  2-entry score lookup silently scored any `pass_3`+ as a failure -- confirmed
  real bug, caught while resolving the merge, fixed via `_pass_n_score()`
  (decaying score floored at 0.4, exactly preserving the original
  `pass_1`=1.0/`pass_2`=0.7 values). `aider_polyglot_ingest.py` handles BOTH
  schemas: `stop_reasons` when present, falling back to the old fixed fields
  for already-captured real data that predates this change.

- tb adapter logs — **UPDATE (superseded the original guess below): confirmed
  against a real captured `tb run` (hello-world task, pi routed through
  fake_pi.py via `LITTLE_CODER_PI_BIN_OVERRIDE`, no LLM cost). Real structure**,
  see `tests/fixtures/real_tb_run/`:
  ```
  <log_root>/results.json                                    run-level aggregate (ignored)
  <log_root>/<task_id>/<task_id>.N-of-M.<ts>/results.json     per-trial: is_resolved (GROUND TRUTH)
  <log_root>/<task_id>/<task_id>.N-of-M.<ts>/agent-logs/*.log little-coder's own log:
    === stop_reason: X ===
    === assistant text ===
    >> tool(args)
    << result
  ```
  **Confirmed real bug this superseded**: a flat `*.log` glob directly under
  `log_root` matches terminal-bench's own top-level `run.log` (task_id="run",
  meaningless data) instead of the real nested agent log — `harbor_tb_ingest.py`
  now walks `<task_id>/<trial>/agent-logs/*.log` explicitly.
  **Confirmed real ground-truth mismatch**: `stop_reason == agent_end` is NOT a
  reliable success proxy — the real fixture has `stop_reason=agent_end` (pi
  finished its turn) but `is_resolved=false` (the task actually failed; a fake
  agent finishing a turn cleanly says nothing about whether it solved the task).
  `success` must come from the trial's own `results.json::is_resolved`.
- harbor adapter logs — **CONFIRMED against a real captured `harbor run`**
  (hello-world task, same `fake_pi.py`/`LITTLE_CODER_PI_BIN_OVERRIDE`
  technique, no LLM cost). **Genuinely different real structure from tb** —
  not a naming variant, a different harness entirely — see
  `tests/fixtures/real_harbor_run/`:
  ```
  <log_root>/result.json                          run-level aggregate (ignored)
  <log_root>/<trial_name>/result.json              (SINGULAR) per-trial, richer:
    agent_result.metadata: {stop_reason, n_tool_calls, n_turns, n_compactions, n_notifications}
    verifier_result.rewards.reward: float           GROUND TRUTH (0.0/1.0, not a boolean)
    task_name: "org/name"  (e.g. "hello-world/hello-world")
  <log_root>/<trial_name>/agent/*.log              (singular "agent", not "agent-logs")
  ```
  Confirmed real differences from tb: single-level trial dirs (not nested
  `<task_id>/<trial>/`), singular `result.json` filename (tb's is plural
  `results.json`), reward-float ground truth instead of an `is_resolved`
  boolean, and — genuinely useful — `agent_result.metadata` already carries
  structured `stop_reason`/`n_turns`/etc., so harbor ingestion reads that
  directly rather than regex-parsing the log file the way tb's ingestion
  must (tb's `results.json` has no equivalent structured field).
  `harbor_tb_ingest.py::load()` dispatches on the `benchmark` argument to two
  separate internal loaders (`_load_harbor`/`_load_tb`) rather than one
  shared parser, since the real shapes don't actually share a code path.

**Gap CLOSED** (was originally flagged here as a known gap; fixed in a later pass):
aider_polyglot's `_dump_trajectory` now takes a `notifications=` kwarg,
populated by the caller (`_run_exercise`) with `rpc.notifications()`,
delta-sliced between attempt 1 and attempt 2 since `PiRpc.notifications()`
accumulates for the whole rpc session, not per-prompt (a real correctness
detail — without the delta slice, attempt 2's trajectory would double-count
attempt 1's notifications). `aider_polyglot_ingest.py` extracts
`components_used` from this field the same way `gaia_ingest.py` already did,
via the shared `merge_component_usage()` helper. Older `trajectory_*.json`
files written before this change have no `"notifications"` key at all and
must still degrade gracefully to `components_used=[]` — kept as a test case
(`test_load_components_used_degrades_gracefully_for_older_trajectories_without_notifications`)
rather than removed, since real historical data in this shape will keep
showing up. See `benchmarks/test_polyglot_artifacts.py` for the upstream
`_dump_trajectory`/`_run_exercise` tests (delta-slicing correctness in
particular) and `aider_polyglot_ingest.py`'s own tests for the ingestion side.

---

## 1. `schema.py`

### 1.1 Test file: `tests/test_schema.py`

```python
def test_component_usage_minimal_construction():
    """ComponentUsage requires only pred_name; other fields have sane defaults."""
    u = ComponentUsage(pred_name="skills_tools_bash")
    assert u.invocation_count == 0
    assert u.was_error_context is False

def test_component_usage_rejects_negative_count():
    """invocation_count must be >= 0 -- a negative count is a parsing bug, not data."""
    with pytest.raises(ValidationError):
        ComponentUsage(pred_name="x", invocation_count=-1)

def test_normalized_trajectory_minimal_construction():
    """Only benchmark/task_id/success/stop_reason/turn_count are required;
    everything else defaults to empty/None so a degraded ingest source can
    still produce a valid object."""
    t = NormalizedTrajectory(
        benchmark="harbor", task_id="t1", success=False,
        stop_reason="deadline", turn_count=3,
    )
    assert t.partial_score is None
    assert t.components_used == []
    assert t.failure_signals == []
    assert t.summarized_transcript == ""
    assert t.raw_paths == {}

def test_normalized_trajectory_rejects_unknown_benchmark():
    with pytest.raises(ValidationError):
        NormalizedTrajectory(
            benchmark="not_a_real_benchmark", task_id="t1",
            success=True, stop_reason="agent_end", turn_count=1,
        )

def test_normalized_trajectory_partial_score_bounds():
    """partial_score, when present, must be in [0.0, 1.0] -- it feeds directly
    into the metric's base_score and an out-of-range value would silently
    corrupt GEPA's Pareto comparisons."""
    with pytest.raises(ValidationError):
        NormalizedTrajectory(
            benchmark="aider_polyglot", task_id="t1", success=True,
            stop_reason="agent_end", turn_count=1, partial_score=1.5,
        )

def test_normalized_trajectory_is_json_roundtrippable():
    """Ingestion modules persist a normalized cache to disk between runs
    (see run_gepa.py's --log-roots caching) -- must survive a dump/load cycle
    with identical field values, including nested ComponentUsage list."""
    t = NormalizedTrajectory(
        benchmark="gaia", task_id="t1", success=True, stop_reason="agent_end",
        turn_count=5, components_used=[ComponentUsage(pred_name="x", invocation_count=2)],
    )
    restored = NormalizedTrajectory.model_validate_json(t.model_dump_json())
    assert restored == t
```

### 1.2 Implementation: `schema.py`

```python
from typing import Literal
from pydantic import BaseModel, Field, field_validator

class ComponentUsage(BaseModel):
    pred_name: str
    invocation_count: int = Field(default=0, ge=0)
    was_error_context: bool = False

class NormalizedTrajectory(BaseModel):
    benchmark: Literal["aider_polyglot", "gaia", "harbor", "tb"]
    task_id: str
    success: bool
    stop_reason: str
    turn_count: int
    partial_score: float | None = Field(default=None, ge=0.0, le=1.0)
    components_used: list[ComponentUsage] = Field(default_factory=list)
    failure_signals: list[str] = Field(default_factory=list)
    summarized_transcript: str = ""
    raw_paths: dict[str, str] = Field(default_factory=dict)
```

(`Literal` + pydantic's own validation gives the "unknown benchmark" rejection and
the "unknown extra field" behavior for free — no custom validators needed beyond the
`ge`/`le` bounds already declared inline.)

---

## 2. `ingest/common.py`

### 2.1 Test file: `tests/test_ingest_common.py`

```python
def test_parse_skill_inject_notification_with_tools():
    usages = parse_notification_line("[info] skill-inject: +2 [bash,read]")
    assert usages == [
        ComponentUsage(pred_name="skills_tools_bash", invocation_count=1),
        ComponentUsage(pred_name="skills_tools_read", invocation_count=1),
    ]

def test_parse_skill_inject_notification_research_directive_only():
    """selected.length === 0 case: no bracket group present at all."""
    usages = parse_notification_line("[info] skill-inject: +research-directive")
    assert usages == []

def test_parse_knowledge_inject_notification():
    usages = parse_notification_line("[info] knowledge-inject: +2 [binary_search,two_pointer]")
    assert usages == [
        ComponentUsage(pred_name="skills_knowledge_binary_search", invocation_count=1),
        ComponentUsage(pred_name="skills_knowledge_two_pointer", invocation_count=1),
    ]

def test_parse_notification_line_ignores_unrelated_lines():
    """quality-monitor / thinking-budget notifications exist but aren't
    component-usage signals for THIS purpose -- must not raise, must return []."""
    assert parse_notification_line("[warn] quality-monitor: empty response detected") == []
    assert parse_notification_line("garbage line with no brackets at all") == []

def test_merge_component_usage_sums_counts_across_lines():
    """A skill injected in 3 different turns of one trajectory must
    aggregate into ONE ComponentUsage with invocation_count=3, not three
    separate entries."""
    lines = [
        "[info] skill-inject: +1 [bash]",
        "[info] skill-inject: +1 [bash]",
        "[info] skill-inject: +1 [read]",
    ]
    merged = merge_component_usage(lines)
    by_name = {u.pred_name: u.invocation_count for u in merged}
    assert by_name == {"skills_tools_bash": 2, "skills_tools_read": 1}

def test_mark_error_context_flags_usage_immediately_after_tool_error():
    """was_error_context=True when a skill-inject notification line
    immediately follows a tool_calls entry with is_error=True, within the
    same trajectory's ordered event stream."""
    tool_calls = [{"name": "bash", "is_error": True}, {"name": "bash", "is_error": False}]
    notif_lines = ["[info] skill-inject: +1 [bash]"]
    # notif attributed to the turn right after the FIRST (erroring) call
    merged = merge_component_usage(notif_lines, follows_error=True)
    assert merged[0].was_error_context is True

def test_summarize_for_reflection_caps_length():
    long_text = "x" * 50_000
    out = summarize_for_reflection(assistant_text=long_text, tool_calls=[], cap=8_000)
    assert len(out) <= 8_000

def test_summarize_for_reflection_prioritizes_error_tool_calls():
    """When truncating, error tool calls must survive in the summary even if
    they occurred early in a long trajectory -- this is the material
    reflection_lm actually needs."""
    tool_calls = [
        {"name": "bash", "args": {"command": "flaky"}, "result_text": "boom", "is_error": True},
        *[{"name": "read", "args": {}, "result_text": "ok", "is_error": False} for _ in range(500)],
    ]
    out = summarize_for_reflection(assistant_text="", tool_calls=tool_calls, cap=2_000)
    assert "boom" in out
    assert "flaky" in out

def test_summarize_for_reflection_handles_empty_input():
    assert summarize_for_reflection(assistant_text="", tool_calls=[], cap=8_000) == ""
```

### 2.2 Implementation notes for `common.py`

- `parse_notification_line(line: str) -> list[ComponentUsage]`: regex
  `r"^\[(?P<level>\w+)\]\s+(?P<source>skill-inject|knowledge-inject):\s+(?:\+\d+\s+\[(?P<names>[^\]]*)\])?"`.
  Map `source == "skill-inject"` names to `skills_tools_{name}`, `knowledge-inject`
  names to `skills_knowledge_{name}` (matches `config/components.yaml` pred_name
  convention from the plan). No match → `[]`, never raise.
- `merge_component_usage(lines, follows_error=False)`: groups by `pred_name`,
  sums counts, propagates `was_error_context` per-name (True if ANY contributing
  line was error-adjacent).
- `summarize_for_reflection`: build error-call excerpts first (name, args, truncated
  result_text) up to the cap, then backfill remaining budget with the tail of
  `assistant_text`; never raise on empty inputs.

---

## 3. `ingest/gaia_ingest.py`

### 3.1 Test file: `tests/test_ingest_gaia.py`, fixtures under `tests/fixtures/gaia_run/`

Fixture layout to construct in a `tmp_path`-based fixture function (not checked into
the repo as static files, to keep the fixture obviously in sync with the schema
above — build it programmatically in a `conftest.py` helper `make_gaia_run(tmp_path)`):

```
<tmp_path>/task-001/
  result.json          {"model_answer": "42", "gold": "42", "correct": true, "elapsed_s": 12.3}
  tool_calls.jsonl      one line: {"name": "bash", "args": {}, "result_text": "ok", "is_error": false}
  notifications.txt     "[info] skill-inject: +1 [bash]\n"
  transcript.txt        "final answer: 42"
  prompt.txt            "..."
<tmp_path>/task-002/
  result.json          {"model_answer": "", "gold": "7", "correct": false, "elapsed_s": 900.0}
  tool_calls.jsonl      (empty file)
  notifications.txt     (empty file)
  transcript.txt        ""
  stderr.log            "Traceback ...\nRuntimeError: pi exited\n"
```

```python
def test_load_returns_one_trajectory_per_task_dir(gaia_run):
    trajs = gaia_ingest.load(gaia_run)
    assert {t.task_id for t in trajs} == {"task-001", "task-002"}

def test_load_maps_correct_field_to_success(gaia_run):
    trajs = {t.task_id: t for t in gaia_ingest.load(gaia_run)}
    assert trajs["task-001"].success is True
    assert trajs["task-002"].success is False

def test_load_sets_benchmark_literal(gaia_run):
    for t in gaia_ingest.load(gaia_run):
        assert t.benchmark == "gaia"

def test_load_extracts_components_used_from_notifications(gaia_run):
    trajs = {t.task_id: t for t in gaia_ingest.load(gaia_run)}
    names = {u.pred_name for u in trajs["task-001"].components_used}
    assert "skills_tools_bash" in names
    assert trajs["task-002"].components_used == []

def test_load_sets_failure_signal_on_stderr_present(gaia_run):
    trajs = {t.task_id: t for t in gaia_ingest.load(gaia_run)}
    assert "process_exit" in trajs["task-002"].failure_signals or \
           "harness_error" in trajs["task-002"].failure_signals

def test_load_skips_task_dir_missing_result_json(tmp_path):
    """A task dir with no result.json (interrupted run) must be skipped,
    not crash the whole ingest -- log a warning instead."""
    (tmp_path / "task-incomplete").mkdir()
    (tmp_path / "task-incomplete" / "transcript.txt").write_text("partial")
    trajs = gaia_ingest.load(tmp_path)
    assert trajs == []

def test_load_ignores_manifest_and_results_json_at_root(gaia_run):
    """manifest.json / results.json / submission.jsonl live at the run root,
    not inside a task dir -- must not be misread as task directories."""
    (gaia_run / "manifest.json").write_text("{}")
    (gaia_run / "results.json").write_text("{}")
    (gaia_run / "submission.jsonl").write_text("")
    trajs = gaia_ingest.load(gaia_run)
    assert len(trajs) == 2  # unchanged from task-001/task-002

def test_load_raw_paths_point_back_to_originals(gaia_run):
    trajs = {t.task_id: t for t in gaia_ingest.load(gaia_run)}
    assert Path(trajs["task-001"].raw_paths["transcript"]).exists()
```

### 3.2 Implementation notes for `gaia_ingest.py`

`def load(log_root: Path) -> list[NormalizedTrajectory]`: iterate immediate
subdirectories of `log_root`, skip any without a `result.json`, read
`result.json` for `correct`→`success`, read `notifications.txt` line-by-line
through `parse_notification_line` + `merge_component_usage`, read
`tool_calls.jsonl` for `summarize_for_reflection`, set `stop_reason` from... —
**note**: gaia's `result.json` per the confirmed shape does NOT currently include
`stop_reason` (only `manifest.json`'s aggregate does, per prior research:
"Aggregate manifest also records n_notifications and stop_reason"). Read
`stop_reason` from the per-task manifest entry if present, else default to
`"agent_end" if success else "unknown"` — write a test for this fallback
explicitly (`test_load_defaults_stop_reason_when_absent`) since it's a real gap
like the aider_polyglot notifications one in §0.

---

## 4. `ingest/aider_polyglot_ingest.py`

### 4.1 Test file: `tests/test_ingest_aider_polyglot.py`

Fixture builder `make_aider_run(tmp_path)`:

```
<tmp_path>/python/two-fer/
  trajectory_1.json   {"attempt": "1", "agent_ended": true, "turn_count": 4,
                        "compaction_events": 0, "assistant_text": "...",
                        "tool_calls": [{"name": "write", "args": {}, "result_text": "ok", "is_error": false}]}
  final_output_1.txt   "1 passed"
<tmp_path>/python/leap/
  trajectory_1.json   {"attempt": "1", "agent_ended": false, "turn_count": 1, ...}
  trajectory_2.json   {"attempt": "2", "agent_ended": true, "turn_count": 3, ...}
  final_output_2.txt  "FAILED test_leap"
results_full_polyglot.json:
  {"exercises": {
     "python/two-fer": {"status": "pass_1", "stop_reason_1": "agent_end", "stop_reason_2": null, "turn_count": 4},
     "python/leap":    {"status": "fail",   "stop_reason_1": "deadline", "stop_reason_2": "agent_end", "turn_count": 4}
  }}
```

```python
def test_load_one_trajectory_per_exercise_using_latest_attempt(aider_run):
    """One NormalizedTrajectory per exercise (not per attempt file) -- the
    exercise's FINAL recorded status is what matters for scoring; attempt_1
    of a two-attempt exercise must not each become a separate example."""
    trajs = aider_polyglot_ingest.load(aider_run["log_root"], aider_run["results_json"])
    assert {t.task_id for t in trajs} == {"python/two-fer", "python/leap"}

def test_load_maps_status_to_success_and_partial_score(aider_run):
    trajs = {t.task_id: t for t in aider_polyglot_ingest.load(aider_run["log_root"], aider_run["results_json"])}
    assert trajs["python/two-fer"].success is True
    assert trajs["python/two-fer"].partial_score == 1.0
    assert trajs["python/leap"].success is False
    assert trajs["python/leap"].partial_score == 0.0

def test_load_pass_2_gets_partial_credit_score(tmp_path):
    """A status of pass_2 (fixed on retry) scores 0.7, per the metric design
    in the plan -- distinguishes a clean first-try pass from a
    needed-a-nudge pass without conflating either with a hard failure."""
    ...
    assert traj.partial_score == 0.7

def test_load_uses_stop_reason_2_when_present_else_stop_reason_1(aider_run):
    trajs = {t.task_id: t for t in aider_polyglot_ingest.load(aider_run["log_root"], aider_run["results_json"])}
    assert trajs["python/leap"].stop_reason == "agent_end"      # stop_reason_2
    assert trajs["python/two-fer"].stop_reason == "agent_end"   # only stop_reason_1 exists

def test_load_components_used_is_empty_pending_notification_capture(aider_run):
    """Confirmed gap (see TDD_SPEC.md §0): aider_polyglot trajectories do not
    currently carry notifications, so components_used MUST be empty, not
    fabricated or inferred from tool_calls. This test locks in the gap so a
    future fix to aider_polyglot.py's _dump_trajectory is a deliberate,
    visible change to this test's expectation -- not a silent behavior
    drift."""
    for t in aider_polyglot_ingest.load(aider_run["log_root"], aider_run["results_json"]):
        assert t.components_used == []

def test_load_skips_exercise_missing_from_results_json(tmp_path):
    """An exercise dir exists on disk (e.g. from a stale/aborted run) but has
    no entry in results_full_polyglot.json -- must be skipped, not crash."""
    ...
    assert trajs == []

def test_load_missing_results_json_raises_clear_error(tmp_path):
    """Distinguish 'no results yet' (caller error, should be loud) from
    'exercise not yet scored' (expected, silent skip)."""
    with pytest.raises(FileNotFoundError):
        aider_polyglot_ingest.load(tmp_path, tmp_path / "does_not_exist.json")
```

### 4.2 Implementation notes

`def load(log_root: Path, results_json_path: Path) -> list[NormalizedTrajectory]`.
Read `results_json_path` first (raise `FileNotFoundError` if absent — this is a
caller error, distinct from an individual exercise being unscored). For each
`(lang, exercise)` key in `results["exercises"]`, build one `NormalizedTrajectory`
from the aggregate record directly (status → success/partial_score per the mapping
`pass_1→(True,1.0)`, `pass_2→(True,0.7)`, everything else→`(False,0.0)`); use
`log_root/<lang>/<exercise>/` only to populate `summarized_transcript` and
`raw_paths` from whichever `trajectory_<n>.json` is highest-numbered (latest
attempt), tolerating its absence (missing detail files degrade gracefully, missing
results.json entries mean skip the exercise entirely per
`test_load_skips_exercise_missing_from_results_json`).

---

## 5. `ingest/harbor_tb_ingest.py`

### 5.1 Test file: `tests/test_ingest_harbor_tb.py`

This is explicitly the "riskiest, least-structured source" per the plan — the spec
leans harder on graceful-degradation tests than on happy-path tests here.

```python
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
    log2 = tmp_path / "task-b.log"
    log2.write_text("=== stop_reason: deadline ===\n")
    assert harbor_tb_ingest.load(tmp_path, benchmark="harbor")[1].success is False

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

def test_load_handles_unreadable_file_without_crashing(tmp_path, monkeypatch):
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
```

### 5.2 Implementation notes

`def load(log_root: Path, benchmark: Literal["harbor", "tb"]) -> list[NormalizedTrajectory]`.
One `*.log` file per task (`task_id` = filename stem). Wrap the per-file parse in
`try/except Exception` at the loop level — on failure, either skip (append nothing)
or append with `failure_signals=["ingest_error"]`, whichever the test above locks
in; log a warning either way. `task_id` set from filename stem, `turn_count`
approximated as the count of `>> ` lines.

---

## 6. `metric.py`

### 6.1 Test file: `tests/test_metric.py`

No DSPy import required for these tests — `metric()` is pure Python taking plain
objects, guard the module's own import with
`dspy = pytest.importorskip("dspy")` only in the ONE test that constructs a real
`dspy.Example`/`ScoreWithFeedback`; everything else uses a tiny local stand-in.

```python
class FakeExample:
    def __init__(self, trajectory): self.trajectory = trajectory

def test_metric_episode_level_returns_base_score_for_pass(monkeypatch):
    traj = NormalizedTrajectory(benchmark="aider_polyglot", task_id="t", success=True,
                                 stop_reason="agent_end", turn_count=3, partial_score=1.0)
    result = metric(FakeExample(traj), pred=None, trace=None, pred_name=None, pred_trace=None)
    assert result.score == 1.0

def test_metric_episode_level_returns_base_score_for_fail(monkeypatch):
    traj = NormalizedTrajectory(benchmark="gaia", task_id="t", success=False,
                                 stop_reason="deadline", turn_count=3, partial_score=0.0)
    result = metric(FakeExample(traj), pred=None, trace=None, pred_name=None, pred_trace=None)
    assert result.score == 0.0

def test_metric_pass_2_scores_partial_credit():
    traj = NormalizedTrajectory(benchmark="aider_polyglot", task_id="t", success=True,
                                 stop_reason="agent_end", turn_count=5, partial_score=0.7)
    result = metric(FakeExample(traj), pred=None, trace=None, pred_name=None, pred_trace=None)
    assert result.score == 0.7

def test_metric_component_not_used_returns_score_no_feedback():
    """When pred_name is a specific component but it wasn't active in this
    trajectory, there's no attributable signal -- score still reflects the
    episode outcome, but feedback must be None (not fabricated text)."""
    traj = NormalizedTrajectory(benchmark="gaia", task_id="t", success=False,
                                 stop_reason="deadline", turn_count=2, partial_score=0.0,
                                 components_used=[])
    result = metric(FakeExample(traj), pred=None, trace=None,
                     pred_name="skills_tools_bash", pred_trace=None)
    assert result.score == 0.0
    assert result.feedback is None

def test_metric_component_used_in_failure_returns_grounded_feedback():
    traj = NormalizedTrajectory(
        benchmark="gaia", task_id="task-42", success=False, stop_reason="deadline",
        turn_count=10, partial_score=0.0, failure_signals=["repeated_tool_call"],
        components_used=[ComponentUsage(pred_name="skills_tools_bash", invocation_count=3,
                                         was_error_context=True)],
    )
    result = metric(FakeExample(traj), pred=None, trace=None,
                     pred_name="skills_tools_bash", pred_trace=None)
    assert result.score == 0.0
    assert result.feedback is not None
    assert "task-42" in result.feedback
    assert "deadline" in result.feedback
    assert "repeated_tool_call" in result.feedback
    assert "3" in result.feedback  # invocation count cited

def test_metric_component_used_in_success_returns_positive_feedback():
    """A component active in a PASSING trajectory should also get feedback
    -- GEPA needs positive signal too, not only failure critique, to avoid
    only ever pushing components toward defensiveness."""
    traj = NormalizedTrajectory(
        benchmark="aider_polyglot", task_id="ex-1", success=True, stop_reason="agent_end",
        turn_count=4, partial_score=1.0,
        components_used=[ComponentUsage(pred_name="skills_tools_bash", invocation_count=1)],
    )
    result = metric(FakeExample(traj), pred=None, trace=None,
                     pred_name="skills_tools_bash", pred_trace=None)
    assert result.score == 1.0
    assert result.feedback is not None

def test_score_for_benchmark_dispatches_by_benchmark_literal():
    """Unit-test the internal dispatcher directly -- each benchmark's
    partial_score->score mapping must be independently verifiable, not only
    exercised transitively through metric()."""
    assert _score_for_benchmark(NormalizedTrajectory(
        benchmark="gaia", task_id="t", success=True, stop_reason="agent_end",
        turn_count=1)) == 1.0

def test_aggregate_score_applies_benchmark_weights():
    """benchmark_weights.yaml lets aggregate valset scoring combine the three
    benchmarks without the largest sample count dominating -- test the
    weighting function directly against a hand-built per-benchmark score dict."""
    weights = {"aider_polyglot": 0.5, "gaia": 0.3, "harbor": 0.1, "tb": 0.1}
    per_benchmark_avg = {"aider_polyglot": 0.8, "gaia": 0.4, "harbor": 1.0, "tb": 0.0}
    agg = weighted_aggregate(per_benchmark_avg, weights)
    assert agg == pytest.approx(0.8*0.5 + 0.4*0.3 + 1.0*0.1 + 0.0*0.1)

def test_weighted_aggregate_missing_benchmark_treated_as_zero_weight():
    """If a benchmark has no examples in a given val split, it must not
    contribute NaN or crash the aggregate -- treat as absent, renormalize
    remaining weights."""
    agg = weighted_aggregate({"gaia": 1.0}, {"aider_polyglot": 0.5, "gaia": 0.5})
    assert agg == 1.0  # renormalized: gaia is the only present benchmark
```

### 6.2 Implementation notes

`ScoreWithFeedback` is a `dspy.teleprompt.gepa` type in real usage; define a local
`@dataclass ScoreWithFeedback: score: float; feedback: str | None = None` in
`metric.py` guarded by `try: from dspy.teleprompt.gepa import ScoreWithFeedback
except ImportError: <local dataclass>` so the test suite runs without `dspy`
installed, and swap to the real import once step 4 (actual GEPA wiring) begins.

`_score_for_benchmark(traj) -> float`: `traj.partial_score if traj.partial_score is
not None else (1.0 if traj.success else 0.0)`.

`_component_feedback(traj, usage, pred_name) -> str`: an f-string template citing
`task_id`, `benchmark`, `stop_reason`, `failure_signals`, `usage.invocation_count`,
`usage.was_error_context`, and a one-line excerpt from `summarized_transcript` —
exact wording is an implementation detail, but the test suite pins which *fields*
must appear (task_id, stop_reason, failure_signals content, invocation count) so a
refactor can't silently drop grounding material.

`weighted_aggregate(per_benchmark_avg: dict[str, float], weights: dict[str, float])
-> float`: renormalize weights over only the keys present in `per_benchmark_avg`.

---

## 7. `components.py` + `apply_results.py` roundtrip

### 7.1 Test file: `tests/test_components_roundtrip.py`

```python
FRONTMATTER_FIXTURE = """---
name: bash-guidance
type: tool-guidance
target_tool: bash
priority: 10
token_cost: 120
user-invocable: false
---
## `bash` Tool
Execute a shell command and return stdout+stderr.
"""

def test_split_frontmatter_separates_yaml_and_body():
    frontmatter, body = split_frontmatter(FRONTMATTER_FIXTURE)
    assert "target_tool: bash" in frontmatter
    assert body.strip().startswith("## `bash` Tool")

def test_split_frontmatter_handles_file_with_no_frontmatter():
    """AGENTS.md and PRINCIPLES.md have no YAML frontmatter block at all --
    must return (None, full_text), not raise or misparse the first '#' line
    as a delimiter."""
    frontmatter, body = split_frontmatter("# little-coder\n\nSome text.\n")
    assert frontmatter is None
    assert body == "# little-coder\n\nSome text.\n"

def test_reattach_frontmatter_reconstructs_original_byte_for_byte(tmp_path):
    """The critical regression guard: split then reattach with an UNCHANGED
    body must reproduce the exact original bytes, including trailing
    newline conventions."""
    reconstructed = reattach_frontmatter(*split_frontmatter(FRONTMATTER_FIXTURE))
    assert reconstructed == FRONTMATTER_FIXTURE

def test_reattach_frontmatter_replaces_only_body_when_changed():
    frontmatter, _old_body = split_frontmatter(FRONTMATTER_FIXTURE)
    new_body = "## `bash` Tool (revised)\nNew guidance text.\n"
    reconstructed = reattach_frontmatter(frontmatter, new_body)
    assert "target_tool: bash" in reconstructed   # frontmatter untouched
    assert "revised" in reconstructed
    assert "Execute a shell command" not in reconstructed  # old body gone

def test_reattach_frontmatter_with_none_frontmatter_returns_body_as_is():
    assert reattach_frontmatter(None, "plain text\n") == "plain text\n"

def test_load_components_from_real_repo_files_via_components_yaml(tmp_path):
    """End-to-end against REAL files (not fixtures) -- copy AGENTS.md and one
    real skill file into a temp repo layout, drive components.yaml, and
    confirm load_components() produces bodies with frontmatter stripped."""
    ...
    comps = load_components(components_yaml_path, repo_root=tmp_path)
    assert "target_tool: bash" not in comps["skills_tools_bash"]  # frontmatter stripped
    assert "## `bash` Tool" in comps["skills_tools_bash"]

def test_write_components_back_preserves_untouched_files(tmp_path):
    """apply_results.py must skip writing any file whose optimized body is
    IDENTICAL to its current body -- prevents spurious git diffs / PR noise
    for components GEPA left alone."""
    ...
    changed = write_components_back(
        components_yaml_path, repo_root=tmp_path,
        optimized={"skills_tools_bash": "## `bash` Tool\nExecute a shell command and return stdout+stderr.\n"},
    )
    assert changed == []  # identical to current body -- no file touched

def test_write_components_back_reports_changed_files(tmp_path):
    ...
    changed = write_components_back(
        components_yaml_path, repo_root=tmp_path,
        optimized={"skills_tools_bash": "## `bash` Tool (v2)\nRevised.\n"},
    )
    assert changed == [tmp_path / "skills" / "tools" / "bash.md"]
    written = (tmp_path / "skills" / "tools" / "bash.md").read_text()
    assert "target_tool: bash" in written  # frontmatter still present
    assert "v2" in written
```

### 7.2 Implementation notes

`split_frontmatter(text: str) -> tuple[str | None, str]`: if `text` doesn't start
with `---\n`, return `(None, text)`. Else find the second `---` line, return
(the block between them inclusive of delimiters, everything after). `reattach_frontmatter`
is the exact inverse — string concatenation, no reformatting, no YAML re-serialization
(never round-trip through a YAML parser/dumper, since that would reformat quoting/
ordering and defeat the byte-for-byte guarantee the tests above pin down).

`components.yaml` schema per the plan: `{pred_name: relative/path/from/repo/root.md}`.
`load_components`/`write_components_back` both take this same mapping so there is
exactly one place a new skill file is registered.

---

## 8. `rpc_client.py::_build_system_prompt()` change

### 8.1 Test file: `benchmarks/test_rpc_system_prompt.py` (lives alongside the existing
`benchmarks/test_*.py` files, NOT under `self_improve/`, since it modifies
`rpc_client.py` itself)

```python
def test_build_system_prompt_returns_agents_md_path_when_no_principles(tmp_path, monkeypatch):
    """Backward-compat guarantee: with no PRINCIPLES.md present, behavior is
    byte-identical to today -- the generated-file mechanism must not
    activate at all."""
    (tmp_path / "AGENTS.md").write_text("# little-coder\n\nBody.\n")
    monkeypatch.setattr(rpc_client, "REPO_ROOT", tmp_path)
    path = rpc_client._build_system_prompt()
    assert path == tmp_path / "AGENTS.md"

def test_build_system_prompt_concatenates_when_principles_present(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("# little-coder\n\nBody.\n")
    (tmp_path / "PRINCIPLES.md").write_text("Be concise.\n")
    monkeypatch.setattr(rpc_client, "REPO_ROOT", tmp_path)
    path = rpc_client._build_system_prompt()
    content = path.read_text()
    assert "# little-coder" in content
    assert "# Principles" in content
    assert "Be concise." in content
    # generated file lives under .pi/, not overwriting AGENTS.md itself
    assert path != tmp_path / "AGENTS.md"

def test_build_system_prompt_generated_file_is_gitignored_path(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("x")
    (tmp_path / "PRINCIPLES.md").write_text("y")
    monkeypatch.setattr(rpc_client, "REPO_ROOT", tmp_path)
    path = rpc_client._build_system_prompt()
    assert ".pi" in path.parts
    assert path.name.startswith(".system-prompt.generated")

def test_build_system_prompt_refreshes_on_each_call(tmp_path, monkeypatch):
    """Edits to PRINCIPLES.md between two PiRpc constructions must be picked
    up -- the generated file is rewritten every call, not cached."""
    (tmp_path / "AGENTS.md").write_text("A")
    (tmp_path / "PRINCIPLES.md").write_text("v1")
    monkeypatch.setattr(rpc_client, "REPO_ROOT", tmp_path)
    p1 = rpc_client._build_system_prompt()
    assert "v1" in p1.read_text()
    (tmp_path / "PRINCIPLES.md").write_text("v2")
    p2 = rpc_client._build_system_prompt()
    assert "v2" in p2.read_text()
    assert "v1" not in p2.read_text()

def test_piRpc_init_uses_build_system_prompt(tmp_path, monkeypatch):
    """Integration point: PiRpc.__init__ must call _build_system_prompt()
    rather than hardcoding AGENTS.md directly, or this whole change is dead
    code. Assert via the constructed subprocess cmd list, not by actually
    spawning pi."""
    calls = []
    monkeypatch.setattr(rpc_client, "_build_system_prompt", lambda: tmp_path / "generated.md")
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: calls.append(cmd) or FakePopen())
    ...
    assert str(tmp_path / "generated.md") in calls[0]
```

### 8.2 Implementation notes

`_build_system_prompt() -> Path`: `agents_md = REPO_ROOT / "AGENTS.md"`,
`principles_md = REPO_ROOT / "PRINCIPLES.md"`. If not `principles_md.exists()`:
return `agents_md` (unchanged behavior, no generated file created — this satisfies
the backward-compat test without even touching disk when PRINCIPLES.md is absent).
Else: write `agents_md.read_text() + "\n\n# Principles\n\n" + principles_md.read_text()`
to `REPO_ROOT / ".pi" / ".system-prompt.generated.md"` (creating `.pi/` if needed),
return that path. Change the existing call site (`rpc_client.py:129-131`) from
`agents_md = REPO_ROOT / "AGENTS.md"` / inline use, to
`system_prompt_path = _build_system_prompt()`, keeping the surrounding
`if agents_md.exists()` guard logic equivalent (guard becomes
`if system_prompt_path.exists()`, which is always true once `_build_system_prompt`
has run, so simplify to an unconditional `cmd.extend(...)` — write a test confirming
the `--no-context-files`/`--system-prompt` flags are still added even when only
`AGENTS.md` exists, matching current behavior).

Add `.pi/.system-prompt.generated.md` to `.gitignore` as part of this change (one
line, not test-driven — a `.gitignore` entry has no meaningful unit test, verify
manually with `git status` showing it untracked after a run).

---

## 9. Execution order for actually doing this

Work through sections 1 → 8 in order, each as its own red-green-refactor cycle:
write the test file's contents, run `pytest` and confirm every test fails for the
right reason (import error / `NameError`, not an assertion mismatch that reveals a
typo in the test itself), then write the minimal implementation, rerun until green,
then move to the next section. Do not write section N+1's implementation before
section N is fully green — `metric.py` (§6) intentionally has no dependency on the
ingestion modules (§3–5) so it can be built and verified in isolation, but
`components.py`/`apply_results.py` (§7) should come after schema/metric are solid
since real GEPA wiring (out of this spec's scope) needs all three together.

After §1–8 are green, the exit criteria for this TDD pass are: `python -m pytest
benchmarks/self_improve/tests/ benchmarks/test_rpc_system_prompt.py -q` passes with
zero failures and zero skips (the `pytest.importorskip("dspy")` guard should not
trigger any skips once `pip install -e 'benchmarks/self_improve[dev]'` has been
run from the REPO ROOT in the dev environment -- this command is repo-root-relative
to match the pytest invocation above it; document that `dspy-ai` is a required dev
dependency for running these tests, not merely for real GEPA runs, since one
`ScoreWithFeedback` import path is exercised).
