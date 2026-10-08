import { describe, it, expect, afterEach } from "vitest";
import {
  archiveId,
  demoteMessagesWithStats,
  demotionGate,
  resolveGateOptions,
  DEFAULT_CMD_KEEP,
  DEFAULT_DEMOTE_BATCH,
  DEFAULT_DEMOTE_FORCE_PENDING_BYTES,
  DEFAULT_DEMOTE_MIN_SAVE_RATIO,
  DEFAULT_DEMOTE_OPEN_AT_PERCENT,
  DEFAULT_DEMOTE_PENDING_BYTES,
  DEFAULT_KEEP_RESULT_HEAD,
  DEFAULT_KEEP_RESULT_TAIL,
  DEFAULT_MIN_PAIR_BYTES,
  DEFAULT_RETAIN_RAW,
  DEFAULT_STALE_DISTANCE,
  MIN_GATE_WINDOW,
  ENV_DEMOTE_FORCE_PENDING_BYTES,
  ENV_DEMOTE_MIN_SAVE_RATIO,
  ENV_DEMOTE_OPEN_AT_PERCENT,
  RESULT_DEMOTED_PREFIX,
  type GateOptions,
  type RetentionArchive,
  type RetentionOptions,
} from "./retention.ts";

const FOOTER = "[exit=0 cwd=/app timed_out=false backend=harbor-env]";

const OPTS: RetentionOptions = {
  retainRaw: DEFAULT_RETAIN_RAW,
  minPairBytes: DEFAULT_MIN_PAIR_BYTES,
  staleDistance: DEFAULT_STALE_DISTANCE,
  keepResultHeadBytes: DEFAULT_KEEP_RESULT_HEAD,
  keepResultTailBytes: DEFAULT_KEEP_RESULT_TAIL,
  cmdKeepBytes: DEFAULT_CMD_KEEP,
  demoteBatch: DEFAULT_DEMOTE_BATCH,
  demotePendingBytes: DEFAULT_DEMOTE_PENDING_BYTES,
};

const GATE: GateOptions = {
  minSaveRatio: DEFAULT_DEMOTE_MIN_SAVE_RATIO,
  openAtPercent: DEFAULT_DEMOTE_OPEN_AT_PERCENT,
  forcePendingBytes: DEFAULT_DEMOTE_FORCE_PENDING_BYTES,
};

const WINDOW = 262144;

function memArchive(): RetentionArchive & { entries: Map<string, string> } {
  const entries = new Map<string, string>();
  return {
    entries,
    save(id, text) {
      entries.set(id, text);
      return true;
    },
    size(id) {
      const t = entries.get(id);
      return t === undefined ? undefined : Buffer.byteLength(t, "utf-8");
    },
    readRange(id, start, length) {
      const t = entries.get(id);
      return t === undefined ? undefined : Buffer.from(t, "utf-8").subarray(start, start + length);
    },
  };
}

function body(bytes: number, tag: string): string {
  const line = `${tag} ${"x".repeat(60)}\n`;
  return line.repeat(Math.ceil(bytes / line.length)).slice(0, bytes);
}

function call(id: string, command: string) {
  return { role: "assistant", content: [{ type: "toolCall", id, name: "ShellSession", arguments: { command } }] };
}

function result(id: string, text: string) {
  return {
    role: "toolResult",
    toolCallId: id,
    toolName: "ShellSession",
    content: [{ type: "text", text: `${text}\n${FOOTER}` }],
    isError: false,
  };
}

/** `n` big pairs, then `tailChars` of plain conversation: the turn-122 shape when the tail is large. */
function history(n: number, pairBytes: number, tailChars: number, tag = "p"): any[] {
  const msgs: any[] = [{ role: "user", content: "go" }];
  for (let i = 0; i < n; i++) msgs.push(call(`${tag}${i}`, `echo ${i}`), result(`${tag}${i}`, body(pairBytes, `${tag}${i}`)));
  // Thinking-heavy turns, as with preserve_thinking: they grow the context but hold no shell output.
  const chunk = 40_000; // few messages, so no pair passes staleDistance
  for (let c = 0; c < tailChars; c += chunk) {
    msgs.push({ role: "assistant", content: [{ type: "thinking", thinking: "t".repeat(chunk) }, { type: "text", text: "ok" }] });
    msgs.push({ role: "user", content: "continue" });
  }
  return msgs;
}

function demotedIds(messages: any[]): string[] {
  return messages
    .filter((m) => m.role === "toolResult" && m.content[0].text.startsWith(RESULT_DEMOTED_PREFIX))
    .map((m) => m.toolCallId);
}

describe("demotionGate", () => {
  const base = { estSaveTokens: 9289, estReprefillTokens: 138377, contextTokens: 184059, contextWindow: WINDOW, pendingBytes: 35418 };

  it("defers the turn-122 jump: small saving, deep break, context under the open point", () => {
    expect(demotionGate(base, GATE)).toBeNull();
  });

  it("opens on the save ratio", () => {
    expect(demotionGate({ ...base, estSaveTokens: 0.1 * base.estReprefillTokens }, GATE)).toBe("ratio");
    expect(demotionGate({ ...base, estSaveTokens: 0.1 * base.estReprefillTokens - 1 }, GATE)).toBeNull();
  });

  it("opens once the context reaches openAtPercent of the window", () => {
    const at = Math.ceil((DEFAULT_DEMOTE_OPEN_AT_PERCENT / 100) * WINDOW);
    expect(demotionGate({ ...base, contextTokens: at }, GATE)).toBe("context");
    expect(demotionGate({ ...base, contextTokens: at - 1 }, GATE)).toBeNull();
  });

  it("opens on pending raw bytes", () => {
    expect(demotionGate({ ...base, pendingBytes: DEFAULT_DEMOTE_FORCE_PENDING_BYTES }, GATE)).toBe("bytes");
    expect(demotionGate({ ...base, pendingBytes: DEFAULT_DEMOTE_FORCE_PENDING_BYTES - 1 }, GATE)).toBeNull();
  });

  it("treats a non-positive ratio or byte budget as disabled, and openAtPercent >= 100 as never by context", () => {
    const off: GateOptions = { minSaveRatio: 0, openAtPercent: 100, forcePendingBytes: 0 };
    expect(demotionGate({ ...base, estSaveTokens: 1e9, contextTokens: WINDOW, pendingBytes: 1e9 }, off)).toBeNull();
  });

  it("opens when the break re-prefills nothing", () => {
    expect(demotionGate({ ...base, estSaveTokens: 1, estReprefillTokens: 0 }, GATE)).toBe("ratio");
  });
});

describe("resolveGateOptions", () => {
  const names = [ENV_DEMOTE_MIN_SAVE_RATIO, ENV_DEMOTE_OPEN_AT_PERCENT, ENV_DEMOTE_FORCE_PENDING_BYTES];
  const saved = Object.fromEntries(names.map((n) => [n, process.env[n]]));
  afterEach(() => {
    for (const n of names) {
      if (saved[n] === undefined) delete process.env[n];
      else process.env[n] = saved[n];
    }
  });

  it("reads the LITTLE_CODER_SHELL_DEMOTE_* knobs with conservative defaults", () => {
    expect(names).toEqual([
      "LITTLE_CODER_SHELL_DEMOTE_MIN_SAVE_RATIO",
      "LITTLE_CODER_SHELL_DEMOTE_OPEN_AT_PERCENT",
      "LITTLE_CODER_SHELL_DEMOTE_FORCE_PENDING_BYTES",
    ]);
    for (const n of names) delete process.env[n];
    expect(resolveGateOptions()).toEqual({ minSaveRatio: 0.1, openAtPercent: 75, forcePendingBytes: 262144 });
    process.env[ENV_DEMOTE_MIN_SAVE_RATIO] = "0.5";
    process.env[ENV_DEMOTE_OPEN_AT_PERCENT] = "0";
    process.env[ENV_DEMOTE_FORCE_PENDING_BYTES] = "-1";
    expect(resolveGateOptions()).toEqual({ minSaveRatio: 0.5, openAtPercent: 0, forcePendingBytes: -1 });
  });
});

describe("demoteMessagesWithStats with the cost gate", () => {
  const B = DEFAULT_DEMOTE_BATCH;
  const R = DEFAULT_RETAIN_RAW;
  // B + R big pairs: the oldest B are a due batch. 4KB each saves ~3.3KB, while
  // the 400K chars of thinking after them is the re-prefill the break would cost.
  const deep = () => history(B + R, 4096, 400_000);

  it("defers a due batch whose saving is small next to the re-prefill it forces", () => {
    const msgs = deep();
    const out = demoteMessagesWithStats(msgs, memArchive(), OPTS, {
      options: GATE,
      context: { contextTokens: 120_000, contextWindow: WINDOW },
    });
    expect(out.demotedCount).toBe(0);
    expect(out.messages).toEqual(msgs);
    expect(out.stats).toMatchObject({ due: B, prefix: 0, demoted: 0, gate: "deferred", gateReason: null, skippedCost: B });
    expect(out.stats.estReprefillTokens).toBeGreaterThan(100_000);
    expect(out.stats.estSaveTokens).toBeGreaterThan(0);
    expect(out.stats.estSaveTokens).toBeLessThan(0.1 * out.stats.estReprefillTokens);
    expect(out.stats.estContextTokens).toBe(120_000);
  });

  it("lets the same batch through near the compaction point", () => {
    const out = demoteMessagesWithStats(deep(), memArchive(), OPTS, {
      options: GATE,
      context: { contextTokens: 0.8 * WINDOW, contextWindow: WINDOW },
    });
    expect(out.demotedCount).toBe(B);
    expect(out.stats).toMatchObject({ prefix: B, demoted: B, gate: "open", gateReason: "context", skippedCost: 0 });
  });

  it("lets a batch through when its saving is large next to a shallow break", () => {
    const shallow = history(B + R, 4096, 0);
    const out = demoteMessagesWithStats(shallow, memArchive(), OPTS, {
      options: GATE,
      context: { contextTokens: 20_000, contextWindow: WINDOW },
    });
    expect(out.demotedCount).toBe(B);
    expect(out.stats).toMatchObject({ gate: "open", gateReason: "ratio" });
  });

  it("keeps an approved batch demoted on the next request even when the gate would now defer it", () => {
    const archive = memArchive();
    const msgs = deep();
    const first = demoteMessagesWithStats(msgs, archive, OPTS, {
      options: GATE,
      context: { contextTokens: 0.8 * WINDOW, contextWindow: WINDOW },
    });
    expect(first.demotedCount).toBe(B);

    // Next request: the context reading dropped below the open point (the
    // demotion shrank the prompt) and the history grew. Without the latch the
    // pairs would go back to raw and break the prefix a second time.
    const next = [...msgs, { role: "user", content: "more" }];
    const second = demoteMessagesWithStats(next, archive, OPTS, {
      options: GATE,
      context: { contextTokens: 0.7 * WINDOW, contextWindow: WINDOW },
    });
    expect(second.demotedCount).toBe(B);
    expect(demotedIds(second.messages)).toEqual(demotedIds(first.messages));
    expect(second.messages.slice(0, msgs.length)).toEqual(first.messages);
    expect(second.stats).toMatchObject({ prefix: B, gate: "none", skippedCost: 0 });
  });

  it("gates only the new part of the prefix once earlier pairs are latched", () => {
    const archive = memArchive();
    const msgs = deep();
    demoteMessagesWithStats(msgs, archive, OPTS, {
      options: GATE,
      context: { contextTokens: 0.8 * WINDOW, contextWindow: WINDOW },
    });
    // B more big pairs make a second batch due; under the open point it is deferred,
    // and the first batch stays exactly as it was.
    const grown = [...msgs, ...history(B, 4096, 0, "q").slice(1)];
    const out = demoteMessagesWithStats(grown, archive, OPTS, {
      options: GATE,
      context: { contextTokens: 0.7 * WINDOW, contextWindow: WINDOW },
    });
    expect(out.stats).toMatchObject({ due: 2 * B, prefix: B, demoted: B, gate: "deferred", skippedCost: B });
    expect(demotedIds(out.messages)).toEqual(Array.from({ length: B }, (_, i) => `p${i}`));
  });

  it("is off, and matches today's projection exactly, without a context window or with openAtPercent <= 0", () => {
    const msgs = deep();
    const today = demoteMessagesWithStats(msgs, memArchive(), OPTS);
    expect(today.demotedCount).toBe(B);
    expect(today.stats).toMatchObject({ gate: "off", skippedCost: 0 });

    const noWindow = demoteMessagesWithStats(msgs, memArchive(), OPTS, {
      options: GATE,
      context: { contextTokens: 120_000, contextWindow: null },
    });
    expect(noWindow.messages).toEqual(today.messages);
    expect(noWindow.stats.gate).toBe("off");

    const disabled = demoteMessagesWithStats(msgs, memArchive(), OPTS, {
      options: { ...GATE, openAtPercent: 0 },
      context: { contextTokens: 120_000, contextWindow: WINDOW },
    });
    expect(disabled.messages).toEqual(today.messages);
    expect(disabled.stats.gate).toBe("off");
  });

  it("opens on a cold cache: pi has no usage between a compaction and the next response", () => {
    const out = demoteMessagesWithStats(deep(), memArchive(), OPTS, {
      options: GATE,
      context: { contextTokens: null, contextWindow: WINDOW },
    });
    expect(out.demotedCount).toBe(B);
    expect(out.stats).toMatchObject({ gate: "open", gateReason: "cold" });
    // ~430K chars at 4 chars/token, from the messages themselves.
    expect(out.stats.estContextTokens).toBeGreaterThan(100_000);
  });

  it("stays off below MIN_GATE_WINDOW, where its defaults were never fitted", () => {
    const msgs = deep();
    const out = demoteMessagesWithStats(msgs, memArchive(), OPTS, {
      options: GATE,
      context: { contextTokens: 20_000, contextWindow: MIN_GATE_WINDOW - 1 },
    });
    expect(out.stats.gate).toBe("off");
    expect(out.demotedCount).toBe(B);
  });

  it("does not count history the server has not cached yet as re-prefill", () => {
    // The same due batch with the long tail arriving as new tool output after the
    // last response: the server prefills it on this request either way.
    const msgs: any[] = history(B + R, 4096, 0);
    msgs.push(call("late", "cat big"), result("late", "y".repeat(400_000)));
    const atEnd = demoteMessagesWithStats(msgs.slice(0, -1).concat([{ role: "user", content: "z".repeat(400_000) }]), memArchive(), OPTS, {
      options: GATE,
      context: { contextTokens: 50_000, contextWindow: WINDOW },
    });
    expect(atEnd.stats.estReprefillTokens).toBeLessThan(20_000);
    expect(atEnd.stats).toMatchObject({ gate: "open", gateReason: "ratio" });
  });

  it("never lets the latch reach past today's candidate (compaction, a changed knob, a reused id)", () => {
    const archive = memArchive();
    // Every pair archived by an earlier projection, but only the oldest B are due now.
    const msgs = history(B + R, 4096, 400_000);
    for (let i = 0; i < B + R; i++) archive.save(archiveId(`p${i}`), "x");
    const out = demoteMessagesWithStats(msgs, archive, OPTS, {
      options: GATE,
      context: { contextTokens: 120_000, contextWindow: WINDOW },
    });
    expect(out.stats).toMatchObject({ sticky: B, prefix: B, gate: "none" });
    expect(demotedIds(out.messages)).toEqual(Array.from({ length: B }, (_, i) => `p${i}`));
  });

  it("defers a pending-bytes flush like any other jump", () => {
    // Under the force budget but over the 64 KiB flush: the flush makes the jump
    // due, the gate still judges it.
    const msgs: any[] = [{ role: "user", content: "go" }];
    msgs.push(call("big", `cat > /tmp/a <<'EOF'\n${body(70 * 1024, "src")}\nEOF`), result("big", "ok"));
    for (let i = 0; i < 30; i++) msgs.push({ role: "user", content: `u${i}` }, { role: "assistant", content: [{ type: "thinking", thinking: "t".repeat(40_000) }] });
    const out = demoteMessagesWithStats(msgs, memArchive(), OPTS, {
      options: GATE,
      context: { contextTokens: 150_000, contextWindow: WINDOW },
    });
    expect(out.stats).toMatchObject({ due: 1, flushed: true, gate: "deferred", skippedCost: 1, prefix: 0 });
    // The break would start inside the call message, at its toolCall block.
    expect(out.stats.estReprefillTokens).toBeGreaterThan(290_000);
  });

  it("ignores pairs that rewrite nothing when placing the break", () => {
    // A signed heredoc whose result is tiny cannot shrink; the break is the next pair's result.
    const msgs: any[] = [{ role: "user", content: "go" }];
    msgs.push(
      { role: "assistant", content: [{ type: "toolCall", id: "s0", name: "ShellSession", arguments: { command: body(5000, "h") }, thoughtSignature: "sig" }] },
      result("s0", "ok"),
    );
    msgs.push({ role: "assistant", content: [{ type: "thinking", thinking: "t".repeat(200_000) }] });
    for (let i = 1; i < B + R; i++) msgs.push(call(`p${i}`, `echo ${i}`), result(`p${i}`, body(4096, `p${i}`)));
    const out = demoteMessagesWithStats(msgs, memArchive(), OPTS, {
      options: GATE,
      context: { contextTokens: 60_000, contextWindow: WINDOW },
    });
    expect(out.stats.estReprefillTokens).toBeLessThan(20_000);
    expect(out.stats).toMatchObject({ gate: "open", skippedNoShrink: 1, signed: 1 });
  });

  it("archives nothing for a deferred pair, so the latch cannot claim it", () => {
    const archive = memArchive();
    demoteMessagesWithStats(deep(), archive, OPTS, {
      options: GATE,
      context: { contextTokens: 120_000, contextWindow: WINDOW },
    });
    expect(archive.entries.has(archiveId("p0"))).toBe(false);
  });
});
