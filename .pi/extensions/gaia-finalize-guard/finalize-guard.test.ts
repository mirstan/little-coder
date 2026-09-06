import { describe, it, expect, beforeEach, afterEach } from "vitest";
import setupExtension from "./index.ts";

interface Handler {
  (event: any, ctx: any): Promise<unknown> | unknown;
}

function makeHarness() {
  const calls: string[] = [];
  const sent: { text: string; options: any }[] = [];
  const notifies: string[] = [];
  const handlers: Record<string, Handler[]> = {};
  const pi = {
    handlers,
    on(name: string, h: Handler) {
      (handlers[name] ??= []).push(h);
    },
    sendUserMessage(text: string, options: any) {
      sent.push({ text, options });
      calls.push("send");
    },
  };
  const ctx = {
    ui: {
      notify(m: string) {
        notifies.push(m);
        calls.push("notify");
      },
    },
  };
  return { pi, ctx, calls, sent, notifies };
}

async function fire(pi: any, name: string, event: any, ctx: any) {
  for (const h of pi.handlers[name] ?? []) await h(event, ctx);
}

function assistantTurn(opts: { text?: string; toolCalls?: number; stopReason?: string }) {
  const content: any[] = [];
  if (opts.text !== undefined) content.push({ type: "text", text: opts.text });
  for (let i = 0; i < (opts.toolCalls ?? 0); i++) {
    content.push({ type: "toolCall", name: "websearch", arguments: {} });
  }
  return { message: { content, stopReason: opts.stopReason } };
}

describe("gaia-finalize-guard", () => {
  beforeEach(() => {
    process.env.LITTLE_CODER_BENCHMARK = "gaia";
  });
  afterEach(() => {
    delete process.env.LITTLE_CODER_BENCHMARK;
  });

  it("does nothing outside GAIA", async () => {
    process.env.LITTLE_CODER_BENCHMARK = "terminal_bench";
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "session_start", {}, h.ctx);
    await fire(h.pi, "turn_end", assistantTurn({ text: "Let me check that." }), h.ctx);
    expect(h.calls).toEqual([]);
  });

  it("does nothing when the turn made a tool call", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "session_start", {}, h.ctx);
    await fire(
      h.pi,
      "turn_end",
      assistantTurn({ text: "Let me check that.", toolCalls: 1 }),
      h.ctx,
    );
    expect(h.calls).toEqual([]);
  });

  it("does nothing when the reply ends with an Answer: line", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "session_start", {}, h.ctx);
    await fire(h.pi, "turn_end", assistantTurn({ text: "Reasoning...\nAnswer: 42" }), h.ctx);
    expect(h.calls).toEqual([]);
  });

  it("accepts a Final Answer: line too (matches gaia_scorer.py's regex)", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "session_start", {}, h.ctx);
    await fire(h.pi, "turn_end", assistantTurn({ text: "Final Answer: backtick" }), h.ctx);
    expect(h.calls).toEqual([]);
  });

  it("skips aborted and errored turns", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "session_start", {}, h.ctx);
    await fire(
      h.pi,
      "turn_end",
      assistantTurn({ text: "Let me check that.", stopReason: "aborted" }),
      h.ctx,
    );
    await fire(
      h.pi,
      "turn_end",
      assistantTurn({ text: "Let me check that.", stopReason: "error" }),
      h.ctx,
    );
    expect(h.calls).toEqual([]);
  });

  it("nudges via steer + notifies when the turn trails off with no tool call and no Answer: line", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "session_start", {}, h.ctx);
    await fire(
      h.pi,
      "turn_end",
      assistantTurn({ text: "Let me fetch it to get the zip codes." }),
      h.ctx,
    );
    expect(h.calls).toEqual(["notify", "send"]);
    expect(h.notifies[0]).toMatch(/harness intervention:.*without a tool call or an Answer/i);
    expect(h.sent[0].options).toEqual({ deliverAs: "steer" });
    expect(h.sent[0].text).toMatch(/didn't call a tool or end with/i);
  });

  it("fires at most once per run", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "session_start", {}, h.ctx);
    await fire(h.pi, "turn_end", assistantTurn({ text: "Let me look into this." }), h.ctx);
    await fire(h.pi, "turn_end", assistantTurn({ text: "Let me look further." }), h.ctx);
    expect(h.sent).toHaveLength(1);
  });

  it("resets per session, so a new task can nudge again", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "session_start", {}, h.ctx);
    await fire(h.pi, "turn_end", assistantTurn({ text: "Let me look into this." }), h.ctx);
    expect(h.sent).toHaveLength(1);

    await fire(h.pi, "session_start", {}, h.ctx);
    await fire(h.pi, "turn_end", assistantTurn({ text: "Let me look into that too." }), h.ctx);
    expect(h.sent).toHaveLength(2);
  });
});
