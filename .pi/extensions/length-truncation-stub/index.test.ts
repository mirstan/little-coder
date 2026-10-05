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

const echoed = `head\n[... 9000 bytes of x ${STUB_ECHO_PHRASE} y ...]\ntail`;

describe("length-truncation-stub tool_call guard (wired)", () => {
  const saved = process.env.LITTLE_CODER_NO_LENGTH_STUB;
  afterEach(() => {
    if (saved === undefined) delete process.env.LITTLE_CODER_NO_LENGTH_STUB;
    else process.env.LITTLE_CODER_NO_LENGTH_STUB = saved;
  });

  it("blocks a call whose input echoes the stub marker, and tells the model to regenerate", async () => {
    delete process.env.LITTLE_CODER_NO_LENGTH_STUB;
    const ctx = makeCtx();
    const result = await fireToolCall(
      wireExtension(),
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

  it("lets ordinary calls through", async () => {
    delete process.env.LITTLE_CODER_NO_LENGTH_STUB;
    const ctx = makeCtx();
    const handlers = wireExtension();
    expect(await fireToolCall(handlers, { toolName: "write", input: { path: "/a", content: "hi" } }, ctx)).toBeUndefined();
    expect(await fireToolCall(handlers, { toolName: "write", input: null }, ctx)).toBeUndefined();
    expect(ctx.notifies).toHaveLength(0);
  });

  it("registers no hooks under the kill switch", () => {
    process.env.LITTLE_CODER_NO_LENGTH_STUB = "1";
    expect(wireExtension()).toEqual({});
  });
});
