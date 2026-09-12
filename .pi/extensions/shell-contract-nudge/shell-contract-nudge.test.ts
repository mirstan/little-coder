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
});

describe("shell-contract-nudge extension", () => {
  it("nudges once on a bare trailing &, with a diagnostic notify every time", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "session_start", {}, h.ctx);

    await fire(h.pi, "tool_call", shellCall("long_job &"), h.ctx);
    await fire(h.pi, "tool_call", shellCall("another_job &"), h.ctx);

    // Diagnostic notify fires on every matching call...
    expect(h.notifies.length).toBe(2);
    expect(h.notifies[0]).toMatch(/bare "&"/);
    expect(h.notifies[1]).toMatch(/bare "&"/);

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

    expect(h.notifies.length).toBe(2);
    expect(h.notifies.every((m) => /pkill -f\/killall/.test(m))).toBe(true);

    expect(h.sent.length).toBe(1);
    expect(h.sent[0].text).toMatch(/pkill -f/);
    expect(h.sent[0].text).toMatch(/wrapper process/);
  });

  it("sends both nudges independently when both patterns appear in the same call", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "session_start", {}, h.ctx);

    // Two segments of one `;`-separated chain: one backgrounds with a bare
    // `&`, the other runs pkill -f.
    await fire(h.pi, "tool_call", shellCall("long_job &; pkill -f watchdog"), h.ctx);

    expect(h.sent.length).toBe(2);
    const texts = h.sent.map((s) => s.text);
    expect(texts.some((t) => /ShellStart/.test(t))).toBe(true);
    expect(texts.some((t) => /wrapper process/.test(t))).toBe(true);

    // A second offending call in the same session re-notifies but does not
    // resend either nudge.
    await fire(h.pi, "tool_call", shellCall("long_job &; pkill -f watchdog"), h.ctx);
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

  it("recognizes ShellSession and ShellStart, not just bash/Bash", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "session_start", {}, h.ctx);

    await fire(h.pi, "tool_call", shellCall("long_job &", "ShellSession"), h.ctx);
    expect(h.notifies.length).toBe(1);
    expect(h.sent.length).toBe(1);

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
    expect(h.notifies.length).toBe(1);

    h.state.sendThrows = false;
    await fire(h.pi, "tool_call", shellCall("long_job &"), h.ctx);
    expect(h.sent.length).toBe(1);
  });

  it("never blocks the tool call — every handler returns undefined", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "session_start", {}, h.ctx);

    const results = await fire(h.pi, "tool_call", shellCall("long_job &; pkill -f x"), h.ctx);
    for (const r of results) expect(r).toBeUndefined();
  });
});
