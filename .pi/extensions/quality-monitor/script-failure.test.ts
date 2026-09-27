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
  // token is the command chain's own final segment."
  describe("final-segment requirement", () => {
    it("does not match when the interpreter+script token is not the chain's final segment", () => {
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

  // Re-checked per fix-discipline: verbatim-repeat exclusion changes this
  // trial's expected count. Of the raw 8 events, `solve.pl`/`net6.pl`/
  // `final.pl` each repeat their own immediately preceding command verbatim,
  // so only 5 are now genuinely distinct -- crossing threshold 1 but never
  // reaching threshold 2 (6). Still nonzero, per the invariant.
  it("mirrors the real trial's event shape after verbatim-repeat exclusion: 5 distinct events from the raw 8, threshold 1 only", () => {
    const paths = [
      "solve.pl", "solve.pl", "lever.pl", "test6.pl",
      "net6.pl", "net6.pl", "final.pl", "final.pl",
    ];
    const detections = paths.map((p, i) => recordAndDeliver(tracker, scriptResult(p, 139, `distinct failure ${i}`)));
    expect(detections.filter((d) => d !== null)).toEqual([{ count: 3, escalated: false }]);
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

  // Truth: "A verbatim-repeated command (same `input` as the immediately
  // preceding counted event) does not increment the count as a new distinct
  // failure."
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
  // lost -- it re-offers on a later ok-verdict turn. Reproduces Member B's
  // own scenario: a threshold crosses on a turn whose verdict comes back
  // non-ok (here, empty_response), so the `if (verdict.ok)` branch that
  // would have sent the steer and called markNotified() never runs.
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
});
