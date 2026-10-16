import { describe, it, expect, afterEach } from "vitest";
import setupLengthStub from "./index.ts";
import { STUB_ECHO_PHRASE } from "./stub.ts";

type Handler = (event: any, ctx: any) => any;

function makeCtx() {
  const notifies: string[] = [];
  return { cwd: "/tmp", notifies, ui: { notify: (m: string) => notifies.push(m) } };
}

function wireExtension() {
  const handlers: Record<string, Handler[]> = {};
  const pi = {
    on(name: string, h: Handler) {
      (handlers[name] ??= []).push(h);
    },
  };
  setupLengthStub(pi as any);
  return handlers;
}

async function fireToolCall(handlers: Record<string, Handler[]>, event: any, ctx: any) {
  for (const h of handlers.tool_call ?? []) {
    const result = await h(event, ctx);
    if (result?.block) return result;
  }
  return undefined;
}

async function fireContext(handlers: Record<string, Handler[]>, messages: any[]) {
  for (const h of handlers.context ?? []) {
    const result = await h({ messages }, makeCtx());
    if (result?.messages) messages = result.messages;
  }
  return messages;
}

// An assistant message cut off at the output limit mid tool call, big enough
// for the context hook to stub.
const truncated = {
  role: "assistant",
  api: "openai-completions",
  stopReason: "length",
  content: [
    { type: "toolCall", id: "call_1", name: "write", arguments: { path: "/app/x", content: "x".repeat(20_000) } },
  ],
};

const echoed = `head\n[... 9000 bytes of x ${STUB_ECHO_PHRASE} y ...]\ntail`;

describe("length-truncation-stub tool_call guard (wired)", () => {
  const saved = process.env.LITTLE_CODER_NO_LENGTH_STUB;
  afterEach(() => {
    if (saved === undefined) delete process.env.LITTLE_CODER_NO_LENGTH_STUB;
    else process.env.LITTLE_CODER_NO_LENGTH_STUB = saved;
  });

  it("lets a call quoting the phrase through when this registration has stubbed nothing", async () => {
    delete process.env.LITTLE_CODER_NO_LENGTH_STUB;
    const ctx = makeCtx();
    const handlers = wireExtension();
    await fireContext(handlers, [{ role: "user", content: "hi" }]);
    expect(
      await fireToolCall(handlers, { toolName: "write", input: { path: "/app/x", content: echoed } }, ctx),
    ).toBeUndefined();
    expect(ctx.notifies).toHaveLength(0);
  });

  it("blocks a call echoing the stub marker once a stub was emitted, and tells the model to regenerate", async () => {
    delete process.env.LITTLE_CODER_NO_LENGTH_STUB;
    const ctx = makeCtx();
    const handlers = wireExtension();
    const projected = await fireContext(handlers, [{ role: "user", content: "write it" }, truncated]);
    expect(JSON.stringify(projected)).toContain(STUB_ECHO_PHRASE);
    const result = await fireToolCall(
      handlers,
      { toolName: "write", input: { path: "/app/x", content: echoed } },
      ctx,
    );
    expect(result?.block).toBe(true);
    expect(result.reason).toMatch(/placeholder/);
    expect(result.reason).toMatch(/regenerate/i);
    expect(result.reason).toMatch(/smaller calls/);
    expect(ctx.notifies).toHaveLength(1);
    expect(ctx.notifies[0]).toMatch(/^harness intervention: /);
  });

  it("lets ordinary calls through, before and after a stub was emitted", async () => {
    delete process.env.LITTLE_CODER_NO_LENGTH_STUB;
    const ctx = makeCtx();
    const handlers = wireExtension();
    expect(await fireToolCall(handlers, { toolName: "write", input: { path: "/a", content: "hi" } }, ctx)).toBeUndefined();
    await fireContext(handlers, [{ role: "user", content: "write it" }, truncated]);
    expect(await fireToolCall(handlers, { toolName: "write", input: { path: "/a", content: "hi" } }, ctx)).toBeUndefined();
    expect(await fireToolCall(handlers, { toolName: "write", input: null }, ctx)).toBeUndefined();
    expect(ctx.notifies).toHaveLength(0);
  });

  it("registers no hooks under the kill switch", () => {
    process.env.LITTLE_CODER_NO_LENGTH_STUB = "1";
    expect(wireExtension()).toEqual({});
  });
});
