import { describe, it, expect, afterEach } from "vitest";
import { findMarkerEchoIds } from "./retention.ts";
import setupShellRetention from "./index.ts";

// ── findMarkerEchoIds: pure extraction, liveness-agnostic ───────────────────

describe("findMarkerEchoIds", () => {
  const LIVE = "sr-f8112e58f6e68b7c";
  const OTHER = "sr-0011223344556677";

  it("extracts an id from the command-infix phrasing (em dash, the real byte)", () => {
    const text = "[... 1.6KB of command text demoted — ShellRecall id=sr-f8112e58f6e68b7c ...]";
    expect(findMarkerEchoIds({ command: text })).toEqual([LIVE]);
  });

  it("extracts an id from the result-prefix phrasing (em dash)", () => {
    const text =
      "[shell result demoted — 12.0KB originally; ShellRecall id=sr-f8112e58f6e68b7c pages back the full command+output]";
    expect(findMarkerEchoIds({ command: text })).toEqual([LIVE]);
  });

  // A small local model reproducing the marker from its own context may not
  // land the em dash byte-exact. The id-phrase match doesn't care.
  for (const dash of ["--", "-", "...", "—"]) {
    it(`tolerates a "${dash}" dash variant in the command-infix phrasing`, () => {
      const text = `[... 1.6KB of command text demoted ${dash} ShellRecall id=${LIVE} ...]`;
      expect(findMarkerEchoIds({ command: text })).toEqual([LIVE]);
    });

    it(`tolerates a "${dash}" dash variant in the result-prefix phrasing`, () => {
      const text = `[shell result demoted ${dash} 12.0KB originally; ShellRecall id=${LIVE} pages back the full command+output]`;
      expect(findMarkerEchoIds({ command: text })).toEqual([LIVE]);
    });
  }

  it("returns [] for text with no marker-shaped substring", () => {
    const bigHeredoc = `cat > file.c <<'EOF'\n${"int x;\n".repeat(2000)}EOF\n`;
    expect(findMarkerEchoIds({ command: bigHeredoc })).toEqual([]);
  });

  it("returns [] for null/undefined input without throwing", () => {
    expect(findMarkerEchoIds(null)).toEqual([]);
    expect(findMarkerEchoIds(undefined)).toEqual([]);
  });

  it("returns [] for non-object/non-string leaves (numbers, booleans)", () => {
    expect(findMarkerEchoIds({ timeout: 30, verbose: true })).toEqual([]);
  });

  it("dedupes an id echoed twice and collects distinct ids from nested leaves", () => {
    const text = `see ShellRecall id=${LIVE} and also ShellRecall id=${LIVE} again`;
    const nested = {
      command: text,
      meta: { notes: [`ShellRecall id=${OTHER}`, "plain text"] },
    };
    const ids = findMarkerEchoIds(nested);
    expect(ids.sort()).toEqual([LIVE, OTHER].sort());
  });

  it("walks arrays as well as objects", () => {
    const ids = findMarkerEchoIds(["no marker here", `ShellRecall id=${LIVE}`]);
    expect(ids).toEqual([LIVE]);
  });
});

// ── tool_call wiring: block genuine echoes, exempt ShellRecall, ignore rest ─

function makeCtx() {
  const notifies: string[] = [];
  return { cwd: "/tmp", notifies, ui: { notify: (m: string) => notifies.push(m) } };
}

function wireExtension() {
  const handlers: Record<string, Array<(event: any, ctx: any) => any>> = {};
  const pi = {
    on(name: string, h: (event: any, ctx: any) => any) {
      (handlers[name] ??= []).push(h);
    },
    registerTool() {},
  };
  setupShellRetention(pi as any);
  return handlers;
}

async function fireToolCall(handlers: Record<string, Array<(event: any, ctx: any) => any>>, event: any, ctx: any) {
  for (const h of handlers.tool_call ?? []) {
    const result = await h(event, ctx);
    if (result?.block) return result;
  }
  return undefined;
}

describe("shell-retention tool_call tripwire (wired)", () => {
  let handlers: Record<string, Array<(event: any, ctx: any) => any>>;

  afterEach(async () => {
    // hostArchive is module-level singleton state in index.ts; session_shutdown
    // is its own cleanup hook, the same one a real session fires on exit.
    for (const h of handlers?.session_shutdown ?? []) await h({}, makeCtx());
  });

  /** Demote a big stale pair through the real context hook to get a live id. */
  async function seedLiveId(): Promise<string> {
    handlers = wireExtension();
    const filler = "x".repeat(50) + "\n";
    const bigBody = filler.repeat(200); // well over the min-pair-bytes floor
    const msgs = [
      { role: "user", content: "go" },
      {
        role: "assistant",
        content: [{ type: "toolCall", id: "tc-1", name: "ShellSession", arguments: { command: "run big thing" } }],
      },
      { role: "toolResult", toolCallId: "tc-1", toolName: "ShellSession", content: [{ type: "text", text: bigBody }] },
      // Enough later traffic to push the pair past staleDistance/retainRaw.
      ...Array.from({ length: 60 }, (_, i) => ({ role: "user", content: `turn ${i}` })),
    ];
    const ctxEvent = { messages: msgs };
    let out: any;
    for (const h of handlers.context ?? []) {
      const r = await h(ctxEvent, makeCtx());
      if (r) out = r;
    }
    expect(out, "expected the context hook to demote the seeded pair").toBeTruthy();
    const demotedResult = out.messages[2];
    const text = demotedResult.content[0].text as string;
    const m = /ShellRecall id=(sr-[0-9a-f]{16})/.exec(text);
    expect(m, "expected a demotion marker with an id in the projected result").toBeTruthy();
    return m![1];
  }

  it("blocks a tool call echoing a live marker id (command-infix phrasing)", async () => {
    const id = await seedLiveId();
    const ctx = makeCtx();
    const event = {
      toolName: "ShellSession",
      input: { command: `cat > f.c <<'EOF'\n[... 1.6KB of command text demoted — ShellRecall id=${id} ...]\nEOF\n` },
    };
    const result = await fireToolCall(handlers, event, ctx);
    expect(result?.block).toBe(true);
    expect(result.reason).toContain(id);
    expect(result.reason).toContain("ShellRecall");
    // Repair-loop mitigation: the reason must cover fixing an already-corrupted
    // file without re-quoting the marker.
    expect(result.reason).toMatch(/line number/i);
    expect(result.reason).toMatch(/sed -i/i);
    expect(ctx.notifies[0]).toMatch(/harness intervention:.*placeholder/i);
  });

  it("blocks a tool call echoing a live marker id (result-prefix phrasing, dash variant)", async () => {
    const id = await seedLiveId();
    const event = {
      toolName: "ShellSession",
      input: {
        command: `echo '[shell result demoted -- 12.0KB originally; ShellRecall id=${id} pages back the full command+output]' >> notes.txt`,
      },
    };
    const result = await fireToolCall(handlers, event, makeCtx());
    expect(result?.block).toBe(true);
  });

  it("does not block ShellRecall itself even with a live id in its input", async () => {
    const id = await seedLiveId();
    const event = { toolName: "ShellRecall", input: { id } };
    const result = await fireToolCall(handlers, event, makeCtx());
    expect(result).toBeUndefined();
  });

  it("does not block a marker-shaped id that was never archived (test-fixture id)", async () => {
    handlers = wireExtension();
    const deadId = "sr-0000000000000000";
    const event = {
      toolName: "ShellSession",
      input: { command: `[... 1.6KB of command text demoted — ShellRecall id=${deadId} ...]` },
    };
    const result = await fireToolCall(handlers, event, makeCtx());
    expect(result).toBeUndefined();
  });

  it("does not block a normal, marker-free tool call, including a large heredoc", async () => {
    handlers = wireExtension();
    const heredoc = `cat > big.c <<'EOF'\n${"int main() { return 0; }\n".repeat(500)}EOF\n`;
    const event = { toolName: "ShellSession", input: { command: heredoc } };
    const result = await fireToolCall(handlers, event, makeCtx());
    expect(result).toBeUndefined();
  });

  it("does not crash on null/undefined tool-call input", async () => {
    handlers = wireExtension();
    await expect(fireToolCall(handlers, { toolName: "ShellSession", input: null }, makeCtx())).resolves.toBeUndefined();
    await expect(
      fireToolCall(handlers, { toolName: "ShellSession", input: undefined }, makeCtx()),
    ).resolves.toBeUndefined();
  });
});
