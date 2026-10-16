import { describe, it, expect } from "vitest";
import setupExtension, { hasBareBackgroundAmpersand, hasPkillOrKillall } from "./index.ts";

interface Handler {
  (event: any, ctx: any): Promise<unknown> | unknown;
}

function makeHarness() {
  const notifies: string[] = [];
  const sent: { text: string; options: any }[] = [];
  const handlers: Record<string, Handler[]> = {};
  const state = { sendThrows: false };
  const pi = {
    handlers,
    state,
    on(name: string, h: Handler) {
      (handlers[name] ??= []).push(h);
    },
    sendUserMessage(text: string, options: any) {
      if (state.sendThrows) throw new Error("SDK does not support sendUserMessage");
      sent.push({ text, options });
    },
  };
  const ctx = {
    ui: {
      notify(m: string) {
        notifies.push(m);
      },
    },
  };
  return { pi, ctx, notifies, sent, state };
}

async function fire(pi: any, name: string, event: any, ctx: any) {
  const results: unknown[] = [];
  for (const h of pi.handlers[name] ?? []) results.push(await h(event, ctx));
  return results;
}

function shellCall(command: string, toolName = "bash") {
  return { toolName, input: { command } };
}

// Every notify tagged "harness intervention: " (via the shared helper) as
// opposed to this extension's own plain "shell-contract-nudge: " diagnostics.
function interventionNotifies(notifies: string[]): string[] {
  return notifies.filter((m) => m.startsWith("harness intervention:"));
}

describe("hasBareBackgroundAmpersand (pure)", () => {
  it("detects a bare trailing &", () => {
    expect(hasBareBackgroundAmpersand("long_job &")).toBe(true);
  });
  it("does not false-positive on &&", () => {
    expect(hasBareBackgroundAmpersand("make && ./run")).toBe(false);
  });
  it("detects a bare & on one segment of a ;-separated chain", () => {
    expect(hasBareBackgroundAmpersand("echo start; long_job &")).toBe(true);
  });
  it("ignores an & quoted as literal text", () => {
    expect(hasBareBackgroundAmpersand('echo "a & b"')).toBe(false);
  });

  it("ignores fd-dup-in <&", () => {
    expect(hasBareBackgroundAmpersand("read x <&3")).toBe(false);
    expect(hasBareBackgroundAmpersand("exec 0<&-")).toBe(false);
  });
  it("ignores case-statement fallthrough ;& and ;;&", () => {
    expect(hasBareBackgroundAmpersand("case $x in a) foo ;;& b) bar ;; esac")).toBe(false);
    expect(hasBareBackgroundAmpersand("foo ;& bar")).toBe(false);
  });
  it("suppresses a job1 & job2 & wait idiom entirely", () => {
    expect(hasBareBackgroundAmpersand("job1 & job2 & wait")).toBe(false);
  });
  it("still flags job1 & job2 & with no trailing wait", () => {
    expect(hasBareBackgroundAmpersand("job1 & job2 &")).toBe(true);
  });
  it("flags an & that precedes a wait but is not itself followed by one", () => {
    expect(hasBareBackgroundAmpersand("job1 & wait; job2 &")).toBe(true);
  });
  it("suppresses when both stdout and stderr are redirected away first", () => {
    expect(hasBareBackgroundAmpersand("nohup cmd > job.log 2>&1 &")).toBe(false);
    expect(hasBareBackgroundAmpersand("cmd &> log &")).toBe(false);
    expect(hasBareBackgroundAmpersand("cmd >> log 2> err.log &")).toBe(false);
  });
  it("still flags a redirect that only covers stdout", () => {
    expect(hasBareBackgroundAmpersand("cmd > log &")).toBe(true);
  });
  it("still flags stderr duplicated onto the OLD stdout pipe before a later redirect", () => {
    // 2>&1 here duplicates stderr onto the pipe stdout had *before* the `>
    // log` redirect that follows it — the tool's original pipe is still in
    // play, so this must NOT be suppressed.
    expect(hasBareBackgroundAmpersand("cmd 2>&1 > log &")).toBe(true);
  });
});

describe("hasPkillOrKillall (pure)", () => {
  it("detects pkill -f", () => {
    expect(hasPkillOrKillall("pkill -f myserver")).toBe(true);
  });
  it("detects killall", () => {
    expect(hasPkillOrKillall("killall myserver")).toBe(true);
  });
  it("does not flag a plain pkill without -f", () => {
    expect(hasPkillOrKillall("pkill myserver")).toBe(false);
  });
  it("detects pkill --full", () => {
    expect(hasPkillOrKillall("pkill --full myserver")).toBe(true);
  });
  it("sees through a leading sudo", () => {
    expect(hasPkillOrKillall("sudo pkill -f myserver")).toBe(true);
  });

  it("sees through env VAR=val, with or without the literal env", () => {
    expect(hasPkillOrKillall("env FOO=bar pkill -f x")).toBe(true);
    expect(hasPkillOrKillall("FOO=bar pkill -f x")).toBe(true);
  });
  it("sees through a path-shaped assignment value (PATH=, LD_PRELOAD=, ...)", () => {
    // The assignment shape must be tested on the raw word BEFORE the
    // leading-path stripper runs, or "PATH=/usr/sbin" becomes "sbin" and
    // matches neither the assignment test nor any wrapper name.
    expect(hasPkillOrKillall("PATH=/usr/sbin pkill -f foo")).toBe(true);
    expect(hasPkillOrKillall("LD_PRELOAD=/x/y.so pkill -f foo")).toBe(true);
    expect(hasPkillOrKillall("env PATH=/usr/sbin pkill -f foo")).toBe(true);
    expect(hasPkillOrKillall("sudo PATH=/usr/bin pkill -f foo")).toBe(true);
    expect(hasPkillOrKillall("TMPDIR=/tmp killall x")).toBe(true);
  });
  it("sees through timeout and its flags/duration operand", () => {
    expect(hasPkillOrKillall("timeout 30 pkill -f x")).toBe(true);
    expect(hasPkillOrKillall("timeout --signal=KILL 30 pkill -f x")).toBe(true);
  });
  it("does not flag command -v/-V as an invocation", () => {
    expect(hasPkillOrKillall("command -v killall")).toBe(false);
  });
  it("does flag command actually running killall", () => {
    expect(hasPkillOrKillall("command killall x")).toBe(true);
  });
  it("flags a later segment even when an earlier one is a pure existence check", () => {
    expect(hasPkillOrKillall("command -v killall; killall foo")).toBe(true);
  });
  it("sees through nice with no explicit flags", () => {
    expect(hasPkillOrKillall("nice pkill -f x")).toBe(true);
  });
  it("sees a pkill call in the segment after a bare-& backgrounded job", () => {
    expect(hasPkillOrKillall("long_job & pkill -f watchdog")).toBe(true);
  });
  it("still handles the existing pkill cluster-flag case", () => {
    expect(hasPkillOrKillall("pkill -xf name")).toBe(true);
  });
  it("stays linear-time on an adversarial all-f flag cluster (ReDoS regression)", () => {
    const bad = "pkill -" + "f".repeat(50_000) + "! target";
    const t0 = Date.now();
    hasPkillOrKillall(bad);
    expect(Date.now() - t0).toBeLessThan(1000);
  });
});

// The bare-`&` scan carries the same class of hazard the pkill flag pattern
// did: it runs on every shell tool call against model-authored text, so any
// super-linear growth is a per-call stall. Each case below was quadratic
// before the fix (timings are the measured "before" on this machine).
describe("hasBareBackgroundAmpersand complexity regressions", () => {
  it("stays linear on a long redirect-shaped prefix (ReDoS regression)", () => {
    // BOTH_STREAMS_REDIRECTED_RE is `$`-anchored but not `^`-anchored, so it
    // retried at every `>` and each retry scanned the rest of the non-space
    // run: 5.4s on this input before the window cap.
    const bad = ">A".repeat(40_000) + " &";
    const t0 = Date.now();
    hasBareBackgroundAmpersand(bad);
    expect(Date.now() - t0).toBeLessThan(1000);
  });

  it("stays linear on the benign `job & job & … & wait` idiom", () => {
    // The suppression path this idiom takes is exactly the one that could not
    // short-circuit, so the safest possible command was the slowest: 1.7s at
    // a tenth of this size before the binary search over `followedByWait`.
    const bad = Array.from({ length: 40_000 }, () => "a &").join(" ") + " wait";
    const t0 = Date.now();
    expect(hasBareBackgroundAmpersand(bad)).toBe(false);
    expect(Date.now() - t0).toBeLessThan(1000);
  });

  it("stays linear when every `&` is redirect-suppressed", () => {
    const bad = Array.from({ length: 20_000 }, () => "c >f 2>&1 &").join(" ");
    const t0 = Date.now();
    expect(hasBareBackgroundAmpersand(bad)).toBe(false);
    expect(Date.now() - t0).toBeLessThan(1000);
  });

  it("stays linear when `wait` appears only as a substring, never as a segment", () => {
    const bad = Array.from({ length: 20_000 }, () => "echo waiting &").join(" ");
    const t0 = Date.now();
    expect(hasBareBackgroundAmpersand(bad)).toBe(true);
    expect(Date.now() - t0).toBeLessThan(1000);
  });

  it("still suppresses/reports correctly around the redirect window boundary", () => {
    // Inside the 256-char window the suppression holds...
    expect(hasBareBackgroundAmpersand("cmd > " + "p".repeat(200) + " 2>&1 &")).toBe(false);
    // ...and past it the clause is truncated, which can only cost a spurious
    // advisory (never a missed one), the same direction as the documented
    // quoted-filename gap.
    expect(hasBareBackgroundAmpersand("cmd > " + "p".repeat(400) + " 2>&1 &")).toBe(true);
  });
});

describe("shell-contract-nudge extension", () => {
  it("nudges once on a bare trailing &, with a diagnostic notify every time", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "session_start", {}, h.ctx);

    await fire(h.pi, "tool_call", shellCall("long_job &"), h.ctx);
    await fire(h.pi, "tool_call", shellCall("another_job &"), h.ctx);

    // Diagnostic notify fires on every matching call...
    const plain = h.notifies.filter((m) => m.startsWith("shell-contract-nudge:"));
    expect(plain.length).toBe(2);
    expect(plain[0]).toMatch(/bare "&"/);
    expect(plain[1]).toMatch(/bare "&"/);

    // ...but the steering nudge is sent only once per session.
    expect(h.sent.length).toBe(1);
    expect(h.sent[0].options).toEqual({ deliverAs: "steer" });
    expect(h.sent[0].text).toMatch(/ShellStart/);
  });

  it("does not treat && as a bare background &", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "session_start", {}, h.ctx);

    await fire(h.pi, "tool_call", shellCall("make && ./run"), h.ctx);

    expect(h.notifies).toEqual([]);
    expect(h.sent).toEqual([]);
  });

  it("nudges on pkill -f, once per session, distinctly from the & nudge", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "session_start", {}, h.ctx);

    await fire(h.pi, "tool_call", shellCall("pkill -f myserver"), h.ctx);
    await fire(h.pi, "tool_call", shellCall("pkill -f otherserver"), h.ctx);

    const plain = h.notifies.filter((m) => m.startsWith("shell-contract-nudge:"));
    expect(plain.length).toBe(2);
    expect(plain.every((m) => /pkill -f\/killall/.test(m))).toBe(true);

    expect(h.sent.length).toBe(1);
    expect(h.sent[0].text).toMatch(/pkill -f/);
    expect(h.sent[0].text).toMatch(/wrapper process/);
  });

  it("sends both nudges independently when both patterns appear in the same call", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "session_start", {}, h.ctx);

    // Two segments, split by the bare `&` itself: one backgrounds with a
    // bare `&`, the other runs pkill -f. (`long_job &; pkill -f watchdog`
    // would be invalid bash — `&;` is a syntax error — so this uses the
    // valid `& ` form, which now also exercises the sub-segmentation fix.)
    await fire(h.pi, "tool_call", shellCall("long_job & pkill -f watchdog"), h.ctx);

    expect(h.sent.length).toBe(2);
    const texts = h.sent.map((s) => s.text);
    expect(texts.some((t) => /ShellStart/.test(t))).toBe(true);
    expect(texts.some((t) => /wrapper process/.test(t))).toBe(true);

    // A second offending call in the same session re-notifies but does not
    // resend either nudge.
    await fire(h.pi, "tool_call", shellCall("long_job & pkill -f watchdog"), h.ctx);
    expect(h.sent.length).toBe(2);
  });

  it("resets the once-per-session latch on a new session_start", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);

    await fire(h.pi, "session_start", {}, h.ctx);
    await fire(h.pi, "tool_call", shellCall("long_job &"), h.ctx);
    expect(h.sent.length).toBe(1);

    await fire(h.pi, "session_start", {}, h.ctx);
    await fire(h.pi, "tool_call", shellCall("long_job &"), h.ctx);
    expect(h.sent.length).toBe(2);
  });

  it("is a no-op for a non-shell tool call", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "session_start", {}, h.ctx);

    await fire(
      h.pi,
      "tool_call",
      { toolName: "write", input: { path: "/tmp/x", content: "long_job &" } },
      h.ctx,
    );
    await fire(
      h.pi,
      "tool_call",
      { toolName: "Edit", input: { path: "/tmp/x", edits: [] } },
      h.ctx,
    );

    expect(h.notifies).toEqual([]);
    expect(h.sent).toEqual([]);
  });

  it("recognizes ShellSession, but exempts ShellStart from the bare-& branch", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "session_start", {}, h.ctx);

    await fire(h.pi, "tool_call", shellCall("long_job &", "ShellSession"), h.ctx);
    expect(h.notifies.filter((m) => m.startsWith("shell-contract-nudge:")).length).toBe(1);
    expect(h.sent.length).toBe(1);

    // bg-shell's own job registry doesn't have the pipe-teardown hazard, so
    // a bare `&` on ShellStart itself must NOT nudge or notify.
    await fire(h.pi, "tool_call", shellCall("long_job &", "ShellStart"), h.ctx);
    expect(h.notifies.filter((m) => m.startsWith("shell-contract-nudge:")).length).toBe(1);
    expect(h.sent.length).toBe(1);

    // The pkill/killall branch still applies to ShellStart — the self-pkill
    // hazard is unrelated to which registry mechanism started the job.
    await fire(h.pi, "tool_call", shellCall("pkill -f x", "ShellStart"), h.ctx);
    expect(h.sent.length).toBe(2); // pkill nudge is a distinct pattern, still fires once
  });

  it("does not burn the once-per-session latch when sendUserMessage throws", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "session_start", {}, h.ctx);

    h.state.sendThrows = true;
    await fire(h.pi, "tool_call", shellCall("long_job &"), h.ctx);
    expect(h.sent.length).toBe(0);
    // The diagnostic notify still fired even though the steer send failed.
    const plain = h.notifies.filter((m) => m.startsWith("shell-contract-nudge:"));
    expect(plain.length).toBe(1);
    // No harnessIntervention either — it is paired 1:1 with an actually
    // delivered steer, never with a failed send.
    expect(interventionNotifies(h.notifies).length).toBe(0);

    h.state.sendThrows = false;
    await fire(h.pi, "tool_call", shellCall("long_job &"), h.ctx);
    expect(h.sent.length).toBe(1);
  });

  it("never blocks the tool call — every handler returns undefined", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "session_start", {}, h.ctx);

    const results = await fire(h.pi, "tool_call", shellCall("long_job & pkill -f x"), h.ctx);
    for (const r of results) expect(r).toBeUndefined();
  });

  it("fires exactly one harnessIntervention-tagged notify per pattern per session, even across 2+ detections", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "session_start", {}, h.ctx);

    await fire(h.pi, "tool_call", shellCall("long_job &"), h.ctx);
    await fire(h.pi, "tool_call", shellCall("another_job &"), h.ctx);
    await fire(h.pi, "tool_call", shellCall("pkill -f myserver"), h.ctx);
    await fire(h.pi, "tool_call", shellCall("pkill -f otherserver"), h.ctx);

    // Two detections of each pattern (4 total), but only one
    // harnessIntervention per pattern (2 total) — one per successfully
    // delivered steer.
    expect(interventionNotifies(h.notifies).length).toBe(2);

    // The plain per-detection diagnostic still matches the number of
    // detections (4).
    const plain = h.notifies.filter((m) => m.startsWith("shell-contract-nudge:"));
    expect(plain.length).toBe(4);
  });

  it("gates ShellStart mention on tool availability via LITTLE_CODER_ALLOWED_TOOLS", async () => {
    const prev = process.env.LITTLE_CODER_ALLOWED_TOOLS;
    try {
      process.env.LITTLE_CODER_ALLOWED_TOOLS = "bash,Edit";
      const h = makeHarness();
      setupExtension(h.pi as any);
      await fire(h.pi, "session_start", {}, h.ctx);

      await fire(h.pi, "tool_call", shellCall("long_job &"), h.ctx);

      expect(h.sent.length).toBe(1);
      expect(h.sent[0].text).not.toMatch(/ShellStart/);
      expect(h.sent[0].text).toMatch(/2>&1/);
    } finally {
      if (prev === undefined) delete process.env.LITTLE_CODER_ALLOWED_TOOLS;
      else process.env.LITTLE_CODER_ALLOWED_TOOLS = prev;
    }
  });

  it("mentions ShellStart when unset (no allow-list gating)", async () => {
    const prev = process.env.LITTLE_CODER_ALLOWED_TOOLS;
    try {
      delete process.env.LITTLE_CODER_ALLOWED_TOOLS;
      const h = makeHarness();
      setupExtension(h.pi as any);
      await fire(h.pi, "session_start", {}, h.ctx);

      await fire(h.pi, "tool_call", shellCall("long_job &"), h.ctx);

      expect(h.sent.length).toBe(1);
      expect(h.sent[0].text).toMatch(/ShellStart/);
    } finally {
      if (prev === undefined) delete process.env.LITTLE_CODER_ALLOWED_TOOLS;
      else process.env.LITTLE_CODER_ALLOWED_TOOLS = prev;
    }
  });
});
