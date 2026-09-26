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
});

describe("ScriptFailureTracker", () => {
  let tracker: ScriptFailureTracker;
  beforeEach(() => {
    tracker = new ScriptFailureTracker();
  });

  it("fires at count 3 with distinct failing scripts, then again at count 6", () => {
    const paths = ["solve.pl", "lever.pl", "test6.pl", "net6.pl", "final.pl", "retry.pl"];
    const detections = paths.map((p, i) => tracker.record(scriptResult(p, 2, `bug ${i}`)));
    expect(detections[0]).toBeNull();
    expect(detections[1]).toBeNull();
    expect(detections[2]).toEqual({ count: 3, escalated: false });
    expect(detections[3]).toBeNull();
    expect(detections[4]).toBeNull();
    expect(detections[5]).toEqual({ count: 6, escalated: true });
  });

  it("mirrors the real trial's 8-event shape: threshold 1 by the 3rd event, threshold 2 by the 6th, none after", () => {
    const paths = [
      "solve.pl", "solve.pl", "lever.pl", "test6.pl",
      "net6.pl", "net6.pl", "final.pl", "final.pl",
    ];
    const detections = paths.map((p, i) => tracker.record(scriptResult(p, 139, `distinct failure ${i}`)));
    expect(detections.filter((d) => d !== null)).toEqual([
      { count: 3, escalated: false },
      { count: 6, escalated: true },
    ]);
  });

  it("does not double-notify threshold 1 on the turns between it and threshold 2", () => {
    const results = [0, 1, 2, 3, 4].map((i) => tracker.record(scriptResult(`s${i}.pl`, 2)));
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
    tracker.record(scriptResult("a.pl", 2));
    tracker.record(scriptResult("b.pl", 2));
    expect(tracker.record(scriptResult("c.pl", 2))).toEqual({ count: 3, escalated: false });
  });

  it("a passing script run neither increments nor resets the monotonic count", () => {
    tracker.record(scriptResult("a.pl", 2));
    tracker.record(scriptResult("b.pl", 2));
    expect(tracker.record(scriptResult("ok.pl", 0))).toBeNull(); // passing run: no-op
    expect(tracker.record(scriptResult("c.pl", 2))).toEqual({ count: 3, escalated: false });
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
});
