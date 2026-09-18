import { describe, it, expect, beforeEach } from "vitest";
import setupQualityMonitor from "./index.ts";

function harness() {
  const handlers: Record<string, ((e: any, c: any) => any)[]> = {};
  const followUps: { msg: string; opts: any }[] = [];
  const pi = {
    handlers,
    on(name: string, h: (e: any, c: any) => any) { (handlers[name] ??= []).push(h); },
    sendUserMessage(msg: string, opts: any) { followUps.push({ msg, opts }); },
  };
  const notifies: string[] = [];
  const ctx = { ui: { notify: (m: string) => notifies.push(m) } };
  setupQualityMonitor(pi as any);
  return { pi, ctx, followUps, notifies };
}
async function fire(h: any, name: string, event: any) {
  for (const fn of h.pi.handlers[name] ?? []) await fn(event, h.ctx);
}
async function fireToolCall(h: any, toolName: string, input: unknown) {
  for (const fn of h.pi.handlers["tool_call"] ?? []) {
    const r = await fn({ type: "tool_call", toolCallId: "t1", toolName, input }, h.ctx);
    if (r?.block) return r;
  }
  return undefined;
}
function turn(calls: { name: string; input: unknown }[], text = "", stopReason = "stop") {
  return { message: { stopReason, content: [
    ...(text ? [{ type: "text", text }] : []),
    ...calls.map((c) => ({ type: "toolCall", name: c.name, arguments: c.input })),
  ] } };
}
async function fireTurn(h: any, calls: { name: string; input: unknown }[], text = "") {
  for (const c of calls) await fireToolCall(h, c.name, c.input);
  return fire(h, "turn_end", turn(calls, text));
}

const gcc = { name: "Bash", input: { command: "gcc encode.c" } };
const ls = { name: "Bash", input: { command: "ls -la" } };

describe("CANDIDATE 1: two same-name calls in the offending turn", () => {
  let h: ReturnType<typeof harness>;
  beforeEach(async () => { h = harness(); await fire(h, "session_start", {}); });

  it("arms the FIRST Bash, not the actual offender", async () => {
    await fireTurn(h, [gcc]);            // seed
    await fireTurn(h, [gcc]);            // fail 1
    await fireTurn(h, [gcc]);            // fail 2
    // fail 3 crosses the cap; turn contains ls -la FIRST, then the repeat gcc
    await fireTurn(h, [ls, gcc]);
    const blockedLs = await fireToolCall(h, ls.name, ls.input);
    const blockedGcc = await fireToolCall(h, gcc.name, gcc.input);
    console.log("ls blocked:", JSON.stringify(blockedLs));
    console.log("gcc blocked:", JSON.stringify(blockedGcc));
    expect({ ls: !!blockedLs?.block, gcc: !!blockedGcc?.block }).toEqual({ ls: false, gcc: true });
  });
});

describe("CANDIDATE 2: leak across an aborted turn", () => {
  let h: ReturnType<typeof harness>;
  beforeEach(async () => { h = harness(); await fire(h, "session_start", {}); });

  it("stale validated calls from an aborted turn survive into the next turn", async () => {
    // aborted turn that already issued Bash(ls -la)
    await fireToolCall(h, ls.name, ls.input);
    await fire(h, "turn_end", { message: { stopReason: "aborted", content: [] } });
    // now a real repeat loop on gcc
    await fireTurn(h, [gcc]);   // seed
    await fireTurn(h, [gcc]);   // fail 1
    await fireTurn(h, [gcc]);   // fail 2
    await fireTurn(h, [gcc]);   // fail 3 -> tier 2 arms
    const blockedLs = await fireToolCall(h, ls.name, ls.input);
    const blockedGcc = await fireToolCall(h, gcc.name, gcc.input);
    console.log("after abort — ls blocked:", JSON.stringify(blockedLs));
    console.log("after abort — gcc blocked:", JSON.stringify(blockedGcc));
    expect({ ls: !!blockedLs?.block, gcc: !!blockedGcc?.block }).toEqual({ ls: false, gcc: true });
  });
});
