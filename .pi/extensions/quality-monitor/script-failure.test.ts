import { describe, it, expect, beforeEach } from "vitest";
import { looksLikeScriptRun, ScriptFailureTracker } from "./script-failure.ts";
import { buildScriptFailureMessage } from "./quality.ts";
import setupQualityMonitor from "./index.ts";

const FOOTER_OK = "[exit=0 cwd=/app timed_out=false backend=subprocess]";
const FOOTER_FAIL = (code: number) => `[exit=${code} cwd=/app timed_out=false backend=subprocess]`;

function scriptResult(path: string, code: number, tail = "some output"): { input: unknown; text: string; isError: boolean } {
  return {
    input: { command: `perl ${path}` },
    text: `${tail}\n${code === 0 ? FOOTER_OK : FOOTER_FAIL(code)}`,
    isError: false,
  };
}

// Calls record() and, if it returned a detection, immediately marks it
// notified -- simulating the real caller (index.ts), which only ever calls
// markNotified() once a detection has actually been delivered. Tests that
// specifically exercise the "delivery never happens" path call
// tracker.record() directly instead, without this helper.
function recordAndDeliver(tracker: ScriptFailureTracker, obs: { input: unknown; text: string; isError: boolean }) {
  const detection = tracker.record(obs);
  if (detection) tracker.markNotified();
  return detection;
}

describe("looksLikeScriptRun", () => {
  it("matches an interpreter invoking a common script extension", () => {
    expect(looksLikeScriptRun({ command: "perl final.pl" })).toBe(true);
    expect(looksLikeScriptRun({ command: "python3 fit_peaks.py" })).toBe(true);
    expect(looksLikeScriptRun({ command: "bash run.sh" })).toBe(true);
  });
  it("does not match a command with no recognizable interpreter+extension pair", () => {
    expect(looksLikeScriptRun({ command: "ls -la" })).toBe(false);
    expect(looksLikeScriptRun({ command: "cat final.pl" })).toBe(false); // no interpreter word
    expect(looksLikeScriptRun({ command: "perl -v" })).toBe(false); // no script extension
  });
  it("does not match a non-string or missing command", () => {
    expect(looksLikeScriptRun({})).toBe(false);
    expect(looksLikeScriptRun(null)).toBe(false);
    expect(looksLikeScriptRun({ command: 42 })).toBe(false);
  });

  // Truth: "The interpreter+script-extension match only counts when that
  // token is in the chain's ATTRIBUTED segment" -- that is the final
  // `;`/`|`/newline-delimited segment UNLESS it's a pure
  // output filter piped from an upstream segment, in which case attribution
  // walks back to that upstream segment (see the dedicated "piped real-run
  // attribution" describe block above for that case). A `;`/newline
  // boundary is never walked back across -- the two sides are independent
  // commands -- which is what the first test below still exercises.
  describe("attributed-segment requirement", () => {
    it("does not match when the interpreter+script token is in a `;`-separated, unrelated segment", () => {
      expect(looksLikeScriptRun({ command: "python3 solve.py; grep expected missing.txt" })).toBe(false);
    });
    it("matches when the interpreter+script token IS the chain's final segment", () => {
      expect(looksLikeScriptRun({ command: "grep expected missing.txt; python3 solve.py" })).toBe(true);
      expect(looksLikeScriptRun({ command: "cd /app && perl final.pl" })).toBe(true);
      expect(looksLikeScriptRun({ command: "make clean || python3 fit_peaks.py" })).toBe(true);
    });
    it("treats a quoted separator inside the script's own argument as part of one segment", () => {
      // The `;` here is inside single quotes -- part of perl's own -e
      // argument, not a chain separator -- so this is one segment whose
      // interpreter+extension match still counts.
      expect(looksLikeScriptRun({ command: `perl -e 'print "a;b"' final.pl` })).toBe(true);
    });
    it("does not split on && / || -- a script that short-circuits a trailing step still matches", () => {
      // `&&` short-circuits: if solve.py fails, `echo done` never runs and
      // the exit code IS solve.py's own. Splitting on `&&` here (treating
      // `echo done` as "the final segment") would wrongly exclude a script
      // failure that is exactly what the chain is reporting.
      expect(looksLikeScriptRun({ command: "python3 solve.py && echo done" })).toBe(true);
      expect(looksLikeScriptRun({ command: "perl final.pl || echo fallback" })).toBe(true);
    });
  });

  // Truth: a real trial-shaped piped script run is detected and attributed
  // to the script segment, not the trailing filter. These are the exact
  // command shapes replayed from the real overfull-hbox__Mq8QoTa trial
  // (benchmarks/harbor_runs/2026-09-24__21-02-23 in the sibling
  // little-coder-dev checkout).
  describe("piped real-run attribution (iter2-f00 Member 1)", () => {
    it("attributes a perl run piped to a filtering grep to the perl segment", () => {
      expect(
        looksLikeScriptRun({ command: 'perl solve.pl 40 2>&1 | grep -vE "Use of uninitialized"' }),
      ).toBe(true);
    });
    it("attributes a python3 run piped to head to the python3 segment", () => {
      expect(looksLikeScriptRun({ command: "python3 solve.py 2>&1 | head -40" })).toBe(true);
    });
    it("walks back across several chained pure-filter segments", () => {
      expect(
        looksLikeScriptRun({ command: 'perl solve.pl 2>&1 | grep -vE "x" | head -40' }),
      ).toBe(true);
    });
    it("still recognizes the same real command wrapped in `timeout N`", () => {
      // The real trial wraps nearly every long-running script call this way
      // (`timeout 60 perl test6.pl`, `timeout 150 perl lever.pl`, ...) --
      // without stripping the wrapper, the first-word invocation check in
      // `looksLikeInvocation` would reject these the same way it correctly
      // rejects `ls python3 missing.py`.
      expect(
        looksLikeScriptRun({ command: 'timeout 60 perl test6.pl 2>&1 | grep -vE "Uninitialized"' }),
      ).toBe(true);
    });
    it("does not reattribute across a `;`/newline boundary -- only a `|` carries a filter's input from the script", () => {
      // Same command text as the existing final-segment test above, stated
      // here under its Member-1 rationale: `;` makes the two commands
      // independent, so the chain's real exit (grep's) is judged on its own
      // and correctly does not match.
      expect(
        looksLikeScriptRun({ command: "python3 solve.py; grep expected missing.txt" }),
      ).toBe(false);
    });
  });

  // Truth: a non-script command whose text merely CONTAINS
  // interpreter+extension-shaped substrings in unrelated args does not
  // match -- the interpreter token must be the actual executable (or a
  // wrapper's argument that becomes one), not just present somewhere in the
  // string.
  describe("command-invocation-shape requirement (iter2-f00 Member 1, Codex)", () => {
    it("does not match when the interpreter word is another command's argument, not its own executable", () => {
      expect(looksLikeScriptRun({ command: "ls python3 missing.py" })).toBe(false);
    });
    it("does not match a literal interpreter+extension substring inside a grep pattern", () => {
      expect(
        looksLikeScriptRun({ command: 'grep "python3 solve.py" file.txt' }),
      ).toBe(false);
    });
  });
});

describe("ScriptFailureTracker", () => {
  let tracker: ScriptFailureTracker;
  beforeEach(() => {
    tracker = new ScriptFailureTracker();
  });

  it("fires at count 3 with distinct failing scripts, then again at count 6", () => {
    const paths = ["solve.pl", "lever.pl", "test6.pl", "net6.pl", "final.pl", "retry.pl"];
    const detections = paths.map((p, i) => recordAndDeliver(tracker, scriptResult(p, 2, `bug ${i}`)));
    expect(detections[0]).toBeNull();
    expect(detections[1]).toBeNull();
    expect(detections[2]).toEqual({ count: 3, escalated: false });
    expect(detections[3]).toBeNull();
    expect(detections[4]).toBeNull();
    expect(detections[5]).toEqual({ count: 6, escalated: true });
  });

  // Re-checked per fix-discipline: this fixture's expected count changes
  // again with the dedup key. Each of the 8 events below carries a
  // DIFFERENT failure body (`distinct failure ${i}`), i.e. each is a
  // genuinely distinct failure even where the command text repeats
  // (`solve.pl`/`net6.pl`/`final.pl` each repeat their own immediately
  // preceding command verbatim, in TEXT only). Iteration 1's command-text-
  // only dedup wrongly suppressed those 3 repeats as "the same attempt seen
  // twice", losing 3 of the raw 8 -- exactly what dedup on (command text,
  // failure signature) fixes: a command rerun that fails DIFFERENTLY is no
  // longer conflated with a truly identical immediate repeat. All 8 are now
  // counted, crossing both thresholds.
  it("mirrors the real trial's event shape: all 8 raw events count once dedup keys on (command, failure signature) rather than command text alone", () => {
    const paths = [
      "solve.pl", "solve.pl", "lever.pl", "test6.pl",
      "net6.pl", "net6.pl", "final.pl", "final.pl",
    ];
    const detections = paths.map((p, i) => recordAndDeliver(tracker, scriptResult(p, 139, `distinct failure ${i}`)));
    expect(detections.filter((d) => d !== null)).toEqual([
      { count: 3, escalated: false },
      { count: 6, escalated: true },
    ]);
  });

  // Truth: a rewrite-then-rerun of the same command text, where the SECOND
  // run fails DIFFERENTLY, is not suppressed as a verbatim repeat -- this
  // is the canonical write->run->fail, REWRITE->run->fail loop the tracker
  // exists to catch, and the command text is typically identical across
  // retries (only the file changed).
  it("counts an immediate rerun of the identical command text when it fails differently (rewrite happened in between)", () => {
    recordAndDeliver(tracker, scriptResult("lever.pl", 2, "some other bug")); // count 1
    expect(
      recordAndDeliver(tracker, scriptResult("solve.pl", 2, "undefined variable $x at line 12")),
    ).toBeNull(); // count 2: first solve.pl attempt, below threshold 1
    // Same command text as the immediately preceding counted event, but the
    // file was rewritten between the two runs and this run fails
    // DIFFERENTLY -- must count as a new distinct failure, not be suppressed
    // as a verbatim repeat the way command-text-only dedup used to.
    expect(
      recordAndDeliver(tracker, scriptResult("solve.pl", 2, "index out of range at line 40")),
    ).toEqual({ count: 3, escalated: false });
  });

  it("does not double-notify threshold 1 on the turns between it and threshold 2", () => {
    const results = [0, 1, 2, 3, 4].map((i) => recordAndDeliver(tracker, scriptResult(`s${i}.pl`, 2)));
    // count 1,2 -> null; count 3 -> fires threshold 1; count 4,5 -> null again.
    expect(results).toEqual([null, null, { count: 3, escalated: false }, null, null]);
  });

  it("unrelated shell failures (no script pattern) never count", () => {
    const grepNoMatch = { input: { command: "grep foo bar.txt" }, text: `\n${FOOTER_FAIL(1)}`, isError: false };
    const lsMissing = { input: { command: "ls /does/not/exist" }, text: `no such file\n${FOOTER_FAIL(2)}`, isError: false };
    const bgJob = { input: { command: "some_daemon &" }, text: `\n${FOOTER_FAIL(1)}`, isError: false };
    expect(tracker.record(grepNoMatch)).toBeNull();
    expect(tracker.record(lsMissing)).toBeNull();
    expect(tracker.record(bgJob)).toBeNull();
    // Confirm the internal count truly never moved: 3 more matching failures
    // should still only just now reach the first threshold.
    recordAndDeliver(tracker, scriptResult("a.pl", 2));
    recordAndDeliver(tracker, scriptResult("b.pl", 2));
    expect(recordAndDeliver(tracker, scriptResult("c.pl", 2))).toEqual({ count: 3, escalated: false });
  });

  it("a passing script run neither increments nor resets the monotonic count", () => {
    recordAndDeliver(tracker, scriptResult("a.pl", 2));
    recordAndDeliver(tracker, scriptResult("b.pl", 2));
    expect(recordAndDeliver(tracker, scriptResult("ok.pl", 0))).toBeNull(); // passing run: no-op
    expect(recordAndDeliver(tracker, scriptResult("c.pl", 2))).toEqual({ count: 3, escalated: false });
  });

  // Truth: "A result with exit === -1 ... does not increment the count."
  describe("exit=-1 exclusion", () => {
    it("a result with exit=-1 (timeout/proxy/sentinel) does not increment the count", () => {
      expect(recordAndDeliver(tracker, scriptResult("solve.pl", -1))).toBeNull();
      // Two more genuine failures should still be only the 2nd and 3rd
      // counted events, not the 3rd and 4th.
      expect(recordAndDeliver(tracker, scriptResult("lever.pl", 2))).toBeNull();
      expect(recordAndDeliver(tracker, scriptResult("test6.pl", 2))).toBeNull();
      expect(recordAndDeliver(tracker, scriptResult("net6.pl", 2))).toEqual({ count: 3, escalated: false });
    });

    it("exit=-1 does not count even when isError is also true", () => {
      const timedOut = { input: { command: "perl solve.pl" }, text: `hung\n${FOOTER_FAIL(-1)}`, isError: true };
      expect(tracker.record(timedOut)).toBeNull();
    });
  });

  // Truth: a re-run with the same command text AND the same failure
  // signature is the same attempt seen twice, and does not increment the
  // count.
  describe("verbatim-repeat exclusion", () => {
    it("does not count an exact repeat of the immediately preceding counted command", () => {
      expect(recordAndDeliver(tracker, scriptResult("solve.pl", 2))).toBeNull(); // count 1
      expect(recordAndDeliver(tracker, scriptResult("solve.pl", 2))).toBeNull(); // verbatim repeat: excluded
      expect(recordAndDeliver(tracker, scriptResult("lever.pl", 2))).toBeNull(); // count 2
      expect(recordAndDeliver(tracker, scriptResult("test6.pl", 2))).toEqual({ count: 3, escalated: false }); // count 3
    });

    it("re-counts the same command once a different one has intervened", () => {
      recordAndDeliver(tracker, scriptResult("solve.pl", 2)); // count 1
      recordAndDeliver(tracker, scriptResult("lever.pl", 2)); // count 2
      // solve.pl again, but NOT the immediately preceding command anymore --
      // this is a new distinct failure and must count.
      expect(recordAndDeliver(tracker, scriptResult("solve.pl", 2))).toEqual({ count: 3, escalated: false });
    });
  });

  // Truth: the final-segment requirement, at the tracker level -- a chain
  // whose own final segment is unrelated to the script must not attribute
  // that segment's failing exit to the script.
  it("does not count a chain whose final segment is unrelated to the script", () => {
    const r = {
      input: { command: "python3 solve.py; grep expected missing.txt" },
      text: `no match\n${FOOTER_FAIL(1)}`,
      isError: false,
    };
    expect(recordAndDeliver(tracker, r)).toBeNull();
  });

  // Truth: the due()/markNotified() split -- a crossing whose message is
  // never delivered (markNotified() never called, e.g. because the turn it
  // happened on was classified non-ok) must not be lost.
  describe("due()/markNotified() split", () => {
    it("keeps reporting a crossed threshold on every subsequent record() call until markNotified() is called", () => {
      expect(tracker.record(scriptResult("solve.pl", 2))).toBeNull(); // count 1
      expect(tracker.record(scriptResult("lever.pl", 2))).toBeNull(); // count 2
      // count 3: threshold 1 crossed, but nothing calls markNotified() --
      // simulating a turn that turned out non-ok, so the message was never
      // actually sent.
      expect(tracker.record(scriptResult("test6.pl", 2))).toEqual({ count: 3, escalated: false });
      // Still un-delivered: the NEXT record() call must report it again
      // (now at the current count), not go silent.
      expect(tracker.record(scriptResult("net6.pl", 2))).toEqual({ count: 4, escalated: false });
      // Now actually deliver it.
      tracker.markNotified();
      // Threshold 1 no longer due; threshold 2 not yet reached.
      expect(tracker.record(scriptResult("final.pl", 2))).toBeNull();
    });
  });
});

describe("buildScriptFailureMessage", () => {
  it("names the count and points at isolating the failing construct (first threshold)", () => {
    const msg = buildScriptFailureMessage(3, false);
    expect(msg).toContain("3 different script-write attempts");
    expect(msg).toMatch(/isolate and test/i);
    expect(msg).not.toMatch(/turn budget/i);
  });
  it("escalates with a turn-budget-cost framing (second threshold)", () => {
    const msg = buildScriptFailureMessage(6, true);
    expect(msg).toContain("6 different script-write attempts");
    expect(msg).toMatch(/turn budget/i);
  });
});

// ── integration: wired into quality-monitor's turn_end, independent of the
// other two detectors ─────────────────────────────────────────────────────
function harness() {
  const handlers: Record<string, ((e: any, c: any) => any)[]> = {};
  const followUps: { msg: string; opts: any }[] = [];
  const pi = {
    handlers,
    on(name: string, h: (e: any, c: any) => any) {
      (handlers[name] ??= []).push(h);
    },
    sendUserMessage(msg: string, opts: any) {
      followUps.push({ msg, opts });
    },
  };
  const notifies: string[] = [];
  const ctx = { ui: { notify: (m: string) => notifies.push(m) } };
  setupQualityMonitor(pi as any);
  return { pi, ctx, followUps, notifies };
}
async function fire(h: any, name: string, event: any) {
  for (const fn of h.pi.handlers[name] ?? []) await fn(event, h.ctx);
}
let idSeq = 0;
async function fireScriptFailureTurn(h: any, path: string, code: number) {
  const id = `tc${++idSeq}`;
  for (const fn of h.pi.handlers["tool_call"] ?? []) {
    await fn({ type: "tool_call", toolCallId: id, toolName: "ShellSession", input: { command: `perl ${path}` } }, h.ctx);
  }
  for (const fn of h.pi.handlers["tool_result"] ?? []) {
    await fn(
      {
        type: "tool_result",
        toolCallId: id,
        toolName: "ShellSession",
        input: { command: `perl ${path}` },
        content: [{ type: "text", text: `distinct output for ${path}\n${FOOTER_FAIL(code)}` }],
        isError: false,
      },
      h.ctx,
    );
  }
  return fire(h, "turn_end", {
    message: {
      stopReason: "stop",
      content: [{ type: "toolCall", name: "ShellSession", arguments: { command: `perl ${path}` }, id }],
    },
  });
}
// A tool_result with no matching toolCall block in the closing message --
// used to cross a threshold on a turn that ends up classified non-ok
// (empty_response: no text, no tool calls in the assistant message) so the
// resulting steer is never actually sent.
async function fireScriptResultOnNonOkTurn(h: any, path: string, code: number) {
  const id = `tc${++idSeq}`;
  for (const fn of h.pi.handlers["tool_result"] ?? []) {
    await fn(
      {
        type: "tool_result",
        toolCallId: id,
        toolName: "ShellSession",
        input: { command: `perl ${path}` },
        content: [{ type: "text", text: `distinct output for ${path}\n${FOOTER_FAIL(code)}` }],
        isError: false,
      },
      h.ctx,
    );
  }
  return fire(h, "turn_end", { message: { stopReason: "stop", content: [] } });
}
// A plain ok-verdict turn with no tool calls/results of its own at all --
// used to prove `due()` is checked once per ok-verdict turn_end regardless
// of whether THIS turn produced a new qualifying script result, not only
// from inside scriptFailureTracker.record().
async function fireOkTextOnlyTurn(h: any, text: string) {
  return fire(h, "turn_end", {
    message: { stopReason: "stop", content: [{ type: "text", text }] },
  });
}

describe("quality-monitor turn_end integration: script-failure tracker", () => {
  it("sends its own steer message at count 3, independent of near_duplicate_loop", async () => {
    const h = harness();
    await fire(h, "session_start", {});
    const paths = ["solve.pl", "lever.pl", "test6.pl"];
    for (const p of paths) await fireScriptFailureTurn(h, p, 2);
    const scriptMsgs = h.followUps.filter((f) => f.msg.includes("different script-write attempts"));
    expect(scriptMsgs).toHaveLength(1);
    expect(scriptMsgs[0].opts).toEqual({ deliverAs: "steer" });
    expect(h.notifies.join("\n")).toMatch(/3 script write-run-fail attempts/i);
  });

  it("resets on a genuinely new prompt (input event), like the other two trackers", async () => {
    const h = harness();
    await fire(h, "session_start", {});
    for (const p of ["a.pl", "b.pl", "c.pl"]) await fireScriptFailureTurn(h, p, 2);
    expect(h.followUps.filter((f) => f.msg.includes("different script-write attempts"))).toHaveLength(1);
    await fire(h, "input", { source: "user" });
    for (const p of ["d.pl", "e.pl"]) await fireScriptFailureTurn(h, p, 2);
    // Only 2 failures since reset -- must not re-fire at a stale count.
    expect(h.followUps.filter((f) => f.msg.includes("different script-write attempts"))).toHaveLength(1);
  });

  // Truth: a threshold crossed on a turn later classified non-ok is not
  // lost -- it re-offers on a later ok-verdict turn: a threshold crosses on
  // a turn whose verdict comes back non-ok (here, empty_response), so the
  // `if (verdict.ok)` branch that would have sent the steer and called
  // markNotified() never runs.
  it("does not silently drop a threshold crossed on a non-ok-verdict turn -- it re-offers on the next ok turn", async () => {
    const h = harness();
    await fire(h, "session_start", {});
    await fireScriptFailureTurn(h, "solve.pl", 2); // count 1, ok verdict
    await fireScriptFailureTurn(h, "lever.pl", 2); // count 2, ok verdict
    // count 3: threshold 1 crossed, but this turn's assistant message has no
    // text and no tool calls -- assessResponse's empty_response, a non-ok
    // verdict -- so steerLoopDetection/markNotified() are never reached.
    await fireScriptResultOnNonOkTurn(h, "test6.pl", 2);
    expect(h.followUps.filter((f) => f.msg.includes("different script-write attempts"))).toHaveLength(0);
    // A later ok-verdict turn with a new distinct failure must still
    // deliver the (now-due-at-a-higher-count) message, not stay silent.
    await fireScriptFailureTurn(h, "net6.pl", 2); // count 4, ok verdict
    const scriptMsgs = h.followUps.filter((f) => f.msg.includes("different script-write attempts"));
    expect(scriptMsgs).toHaveLength(1);
    expect(scriptMsgs[0].msg).toContain("4 different script-write attempts");
  });

  // Truth: the gap the test above does NOT cover --
  // that one still delivers via a new qualifying script result arriving on
  // the later ok turn. Here the later ok-verdict turn has NO ShellSession
  // call/result of its own at all, so `record()` is never invoked for it;
  // only a `due()` check independent of that turn's own results can surface
  // the still-pending threshold.
  it("re-offers a threshold crossed on a non-ok turn on a later ok-verdict turn that has NO new script-shaped result of its own", async () => {
    const h = harness();
    await fire(h, "session_start", {});
    await fireScriptFailureTurn(h, "solve.pl", 2); // count 1, ok verdict
    await fireScriptFailureTurn(h, "lever.pl", 2); // count 2, ok verdict
    // count 3: threshold 1 crossed, but this turn's verdict is non-ok, so
    // the steer is never sent and markNotified() never runs.
    await fireScriptResultOnNonOkTurn(h, "test6.pl", 2);
    expect(h.followUps.filter((f) => f.msg.includes("different script-write attempts"))).toHaveLength(0);
    // A later ok-verdict turn with no tool call/result at all -- just text --
    // must still deliver the still-pending count-3 message.
    await fireOkTextOnlyTurn(h, "Let me reconsider the approach.");
    const scriptMsgs = h.followUps.filter((f) => f.msg.includes("different script-write attempts"));
    expect(scriptMsgs).toHaveLength(1);
    expect(scriptMsgs[0].msg).toContain("3 different script-write attempts");
  });
});
