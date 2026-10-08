import { describe, it, expect, beforeEach, afterEach } from "vitest";
import setupLengthStub from "./index.ts";
import { STUB_ECHO_PHRASE } from "./stub.ts";

type Handler = (event: any, ctx: any) => any;

function wire() {
  const handlers: Record<string, Handler[]> = {};
  const entries: Array<{ customType: string; data: any }> = [];
  const pi = {
    on(name: string, h: Handler) {
      (handlers[name] ??= []).push(h);
    },
    appendEntry(customType: string, data: any) {
      entries.push({ customType, data });
    },
  };
  setupLengthStub(pi as any);
  return { handlers, entries };
}

function makeCtx() {
  return { cwd: "/tmp", ui: { notify: (_m: string) => {} } };
}

async function fireContext(handlers: Record<string, Handler[]>, messages: any[]) {
  for (const h of handlers.context ?? []) {
    const result = await h({ messages }, makeCtx());
    if (result?.messages) messages = result.messages;
  }
  return messages;
}

const truncated = {
  role: "assistant",
  api: "openai-completions",
  stopReason: "length",
  content: [
    { type: "toolCall", id: "call_1", name: "write", arguments: { path: "/app/x", content: "x".repeat(20_000) } },
  ],
};

describe("length-truncation-stub telemetry", () => {
  let saved: Record<string, string | undefined>;
  beforeEach(() => {
    saved = {
      LITTLE_CODER_NO_LENGTH_STUB: process.env.LITTLE_CODER_NO_LENGTH_STUB,
      LITTLE_CODER_TELEMETRY: process.env.LITTLE_CODER_TELEMETRY,
    };
    delete process.env.LITTLE_CODER_NO_LENGTH_STUB;
    process.env.LITTLE_CODER_TELEMETRY = "1";
  });
  afterEach(() => {
    for (const [k, v] of Object.entries(saved)) {
      if (v === undefined) delete process.env[k];
      else process.env[k] = v;
    }
  });

  it("emits a length_stub snapshot only for requests that stub, and none under the kill switch", async () => {
    const { handlers, entries } = wire();
    await fireContext(handlers, [{ role: "user", content: "hi" }]);
    expect(entries).toEqual([]);

    const projected = await fireContext(handlers, [{ role: "user", content: "write it" }, truncated]);
    expect(entries.map((e) => [e.customType, e.data.kind, e.data.stubbed])).toEqual([["lc-telemetry", "length_stub", 1]]);
    expect(entries[0].data.bytesBefore).toBeGreaterThan(20_000);
    expect(entries[0].data.bytesAfter).toBeLessThan(2_000);
    expect(JSON.stringify(projected)).toContain(STUB_ECHO_PHRASE);

    delete process.env.LITTLE_CODER_TELEMETRY;
    const still = await fireContext(handlers, [{ role: "user", content: "write it" }, truncated]);
    expect(entries).toHaveLength(1);
    expect(JSON.stringify(still)).toContain(STUB_ECHO_PHRASE);
  });

  it("emits echo_block when it blocks a call that copies the stub", async () => {
    const { handlers, entries } = wire();
    await fireContext(handlers, [{ role: "user", content: "write it" }, truncated]);
    let blocked: any;
    for (const h of handlers.tool_call ?? []) {
      const r = await h(
        { toolName: "write", input: { path: "/app/x", content: `head\n[... 9000 bytes ${STUB_ECHO_PHRASE} ...]\ntail` } },
        makeCtx(),
      );
      if (r?.block) blocked = r;
    }
    expect(blocked?.block).toBe(true);
    expect(entries.filter((e) => e.data.kind === "echo_block").map((e) => e.data)).toEqual([
      { v: 1, kind: "echo_block", source: "length_stub", tool: "write" },
    ]);
  });
});
