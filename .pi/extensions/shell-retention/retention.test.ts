import { describe, it, expect } from "vitest";
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  archiveId,
  archiveText,
  demoteMessages,
  demotedPrefixLength,
  recallSlice,
  resolveOptions,
  CMD_DEMOTED_INFIX,
  DEFAULT_CMD_KEEP,
  DEFAULT_DEMOTE_BATCH,
  DEFAULT_KEEP_RESULT_HEAD,
  DEFAULT_KEEP_RESULT_TAIL,
  DEFAULT_MIN_PAIR_BYTES,
  DEFAULT_RECALL_BYTES,
  DEFAULT_RECALL_MAX,
  DEFAULT_RETAIN_RAW,
  DEFAULT_STALE_DISTANCE,
  ENV_DEMOTE_BATCH,
  ENV_MIN_PAIR_BYTES,
  ENV_RETAIN_RAW,
  REASONING_FIELD_NAMES,
  RESULT_DEMOTED_PREFIX,
  type RetentionArchive,
  type RetentionOptions,
} from "./retention.ts";

// Canned message shapes mirror pi's AgentMessage / ToolResultMessage.
// See node_modules/@earendil-works/pi-ai/dist/types.d.ts for the real types.

const FOOTER = "[exit=0 cwd=/app timed_out=false backend=harbor-env]";

// Existing fixtures pin per-pair demotion (demoteBatch 1, today's behaviour);
// batched behaviour is exercised in its own describe blocks with explicit opts.
const BASE: RetentionOptions = {
  retainRaw: DEFAULT_RETAIN_RAW,
  minPairBytes: DEFAULT_MIN_PAIR_BYTES,
  staleDistance: DEFAULT_STALE_DISTANCE,
  keepResultHeadBytes: DEFAULT_KEEP_RESULT_HEAD,
  keepResultTailBytes: DEFAULT_KEEP_RESULT_TAIL,
  cmdKeepBytes: DEFAULT_CMD_KEEP,
  demoteBatch: 1,
};

function opts(over: Partial<RetentionOptions> = {}): RetentionOptions {
  return { ...BASE, ...over };
}

function userMsg(text: string) {
  return { role: "user", content: text };
}

function assistantShell(id: string, command: string, extra: Record<string, unknown> = {}) {
  return {
    role: "assistant",
    content: [
      { type: "toolCall", id, name: "ShellSession", arguments: { command }, ...extra },
    ],
  };
}

function shellResult(
  id: string,
  body: string,
  footer: string | null = FOOTER,
  toolName = "ShellSession",
) {
  return {
    role: "toolResult",
    toolCallId: id,
    toolName,
    content: [{ type: "text", text: footer === null ? body : `${body}\n${footer}` }],
    isError: false,
    timestamp: 1700000000000,
  };
}

function textOf(m: any): string {
  return m.content[0].text;
}

function commandOf(m: any): string {
  return m.content[0].arguments.command;
}

/** Filler of roughly `bytes` bytes, line-broken like real command output. */
function filler(bytes: number, tag: string): string {
  const out: string[] = [];
  let total = 0;
  for (let i = 0; total < bytes; i++) {
    const line = `${tag} line ${i} ${"=".repeat(50)}`;
    out.push(line);
    total += line.length + 1;
  }
  return out.join("\n");
}

function memArchive(): RetentionArchive & { entries: Map<string, string> } {
  const entries = new Map<string, string>();
  return {
    entries,
    save(id, text) {
      entries.set(id, text);
      return true;
    },
    size(id) {
      const text = entries.get(id);
      return text === undefined ? undefined : Buffer.byteLength(text, "utf-8");
    },
    readRange(id, start, length) {
      const text = entries.get(id);
      if (text === undefined) return undefined;
      return Buffer.from(text, "utf-8").subarray(start, start + length);
    },
  };
}

const failingArchive: RetentionArchive = {
  save: () => false,
  size: () => undefined,
  readRange: () => undefined,
};

/** `count` pairs of a small command and a `bodyBytes` result, ids `${tag}N`. */
function pairs(count: number, tag: string, bodyBytes = 4096): any[] {
  const msgs: any[] = [];
  for (let i = 0; i < count; i++) {
    msgs.push(assistantShell(`${tag}${i}`, `echo run-${tag}-${i}`));
    msgs.push(shellResult(`${tag}${i}`, filler(bodyBytes, `${tag}${i}`)));
  }
  return msgs;
}

describe("demoteMessages", () => {
  it("no-op on history without shell results", () => {
    const msgs = [userMsg("hello"), { role: "assistant", content: [{ type: "text", text: "hi" }] }];
    const out = demoteMessages(msgs, memArchive(), opts());
    expect(out.demotedCount).toBe(0);
    expect(out.messages).toEqual(msgs);
  });

  it("keeps the newest retainRaw qualifying pairs raw and demotes older ones", () => {
    const msgs = [userMsg("build it"), ...pairs(6, "p")];
    const archive = memArchive();
    const out = demoteMessages(msgs, archive, opts());

    expect(out.demotedCount).toBe(2);
    expect(textOf(out.messages[2])).toContain(RESULT_DEMOTED_PREFIX);
    expect(textOf(out.messages[4])).toContain(RESULT_DEMOTED_PREFIX);
    for (let i = 2; i < 6; i++) {
      const resultIdx = 2 + 2 * i;
      expect(out.messages[resultIdx]).toEqual(msgs[resultIdx]);
      expect(out.messages[resultIdx - 1]).toEqual(msgs[resultIdx - 1]);
    }
  });

  it("leaves pairs below the size floor alone regardless of age", () => {
    const msgs = [userMsg("t"), ...pairs(10, "s", 120)];
    expect(demoteMessages(msgs, memArchive(), opts()).demotedCount).toBe(0);
    expect(demoteMessages(msgs, memArchive(), opts({ staleDistance: 1 })).demotedCount).toBe(0);
  });

  it("sizes on the pair, so a huge command with a tiny result still demotes", () => {
    const command = `cat > /tmp/a.c <<'EOF'\n${filler(8192, "src")}\nEOF\ngcc -o /tmp/a /tmp/a.c && echo compiled`;
    const msgs = [
      userMsg("t"),
      assistantShell("big", command),
      shellResult("big", "compiled"),
      ...pairs(4, "n"),
    ];
    const archive = memArchive();
    const out = demoteMessages(msgs, archive, opts());

    expect(out.demotedCount).toBe(1);
    // Result side would grow, so only the call demotes.
    expect(out.messages[2]).toEqual(msgs[2]);
    const demotedCmd = commandOf(out.messages[1]);
    expect(demotedCmd).toContain("cat > /tmp/a.c <<'EOF'");
    expect(demotedCmd).toContain("gcc -o /tmp/a /tmp/a.c && echo compiled");
    expect(demotedCmd).toContain(CMD_DEMOTED_INFIX);
    expect(demotedCmd.length).toBeLessThan(command.length);
    expect(archive.entries.get(archiveId("big"))).toContain(command);
  });

  it("demotes the write-compressor shape and recalls it byte-exact", () => {
    const source = [
      "#include <stdio.h>",
      "#include <stdlib.h>",
      filler(3500, "src-a"),
      "int MIDDLE_SENTINEL = 42;",
      filler(3500, "src-b"),
      "int main(void) { return decode(); }",
    ].join("\n");
    const command =
      `cd /app && cat > /tmp/redec.c <<'EOF'\n${source}\nEOF\n` +
      `gcc -O0 -o /tmp/redec /tmp/redec.c && echo "compiled" && /tmp/redec 2>&1; echo "exit=$?"`;
    const firstWarning = "/tmp/redec.c:12:5: warning: implicit declaration of function 'decode'";
    const body = [
      "compiled",
      firstWarning,
      filler(6000, "warn"),
      "bash: line 87:  1104 Segmentation fault      /tmp/redec 2>&1",
      "exit=139",
    ].join("\n");
    const resultText = `${body}\n${FOOTER}`;

    const msgs = [userMsg("compress it"), assistantShell("wc1", command), shellResult("wc1", body), ...pairs(4, "n")];
    const archive = memArchive();
    const out = demoteMessages(msgs, archive, opts());

    expect(out.demotedCount).toBe(1);
    const id = archiveId("wc1");

    const demotedResult = textOf(out.messages[2]);
    expect(demotedResult).toContain(RESULT_DEMOTED_PREFIX);
    expect(demotedResult).toContain(id);
    expect(demotedResult).toContain(firstWarning);
    expect(demotedResult.split("\n").at(-1)).toBe(FOOTER);
    // The harness footer says exit=0; the real failure is in the body tail.
    expect(demotedResult).toContain("Segmentation fault");
    expect(demotedResult).toContain("exit=139");
    expect(demotedResult).not.toContain("warn line 40 ");

    const demotedCmd = commandOf(out.messages[1]);
    expect(demotedCmd.split("\n")[0]).toBe("cd /app && cat > /tmp/redec.c <<'EOF'");
    expect(demotedCmd.split("\n").at(-1)).toBe(
      `gcc -O0 -o /tmp/redec /tmp/redec.c && echo "compiled" && /tmp/redec 2>&1; echo "exit=$?"`,
    );
    expect(demotedCmd).not.toContain("MIDDLE_SENTINEL");
    expect(demotedCmd).toContain(id);

    const archived = archive.entries.get(id)!;
    expect(archived).toBe(archiveText(command, resultText));
    expect(archived).toContain(command);
    expect(archived).toContain(resultText);

    const sentinelOffset = Buffer.byteLength(archived.slice(0, archived.indexOf("int MIDDLE_SENTINEL")), "utf-8");
    const recalled = recallSlice(id, archive, sentinelOffset, 200, {
      defaultBytes: DEFAULT_RECALL_BYTES,
      maxBytes: DEFAULT_RECALL_MAX,
    });
    expect(recalled.isError).toBe(false);
    expect(recalled.text).toContain("int MIDDLE_SENTINEL = 42;");
  });

  it("keeps the real failure signal when the harness footer reports exit=0", () => {
    const body = [
      "compiled",
      filler(6000, "warn"),
      "bash: line 87:  1104 Segmentation fault      /tmp/redec 2>&1",
      "exit=139",
    ].join("\n");
    const msgs = [userMsg("t"), assistantShell("sf", "gcc x.c && ./a.out; echo \"exit=$?\""), shellResult("sf", body), ...pairs(4, "n")];
    const out = demoteMessages(msgs, memArchive(), opts());

    const demoted = textOf(out.messages[2]);
    expect(demoted).toContain("Segmentation fault");
    expect(demoted).toContain("exit=139");
    expect(demoted.split("\n").at(-1)).toBe(FOOTER);
  });

  it("keeps the last literal line when there is no harness footer", () => {
    const body = `${filler(5000, "gaia")}\nDONE-LAST-LINE`;
    const msgs = [
      userMsg("t"),
      {
        role: "assistant",
        content: [{ type: "toolCall", id: "g1", name: "bash", arguments: { command: "ls -R /" } }],
      },
      shellResult("g1", body, null, "bash"),
      ...pairs(4, "n"),
    ];
    const out = demoteMessages(msgs, memArchive(), opts());
    expect(textOf(out.messages[2]).split("\n").at(-1)).toBe("DONE-LAST-LINE");
  });

  it("demotes a rank-0 pair once it is far enough from the end", () => {
    const msgs = [userMsg("t"), assistantShell("g", "echo go"), shellResult("g", filler(8192, "big")), ...pairs(60, "tiny", 60)];
    const out = demoteMessages(msgs, memArchive(), opts());
    expect(out.demotedCount).toBe(1);
    expect(textOf(out.messages[2])).toContain(RESULT_DEMOTED_PREFIX);
  });

  it("leaves a pair raw when the archive cannot store it", () => {
    const msgs = [userMsg("t"), ...pairs(6, "p")];
    const out = demoteMessages(msgs, failingArchive, opts());
    expect(out.demotedCount).toBe(0);
    expect(out.messages).toEqual(msgs);
  });

  it("is idempotent when its own output is fed back in", () => {
    const msgs = [userMsg("t"), ...pairs(6, "p")];
    const first = demoteMessages(msgs, memArchive(), opts());
    expect(first.demotedCount).toBe(2);
    expect(demoteMessages(first.messages, memArchive(), opts()).demotedCount).toBe(0);
  });

  it("is deterministic and stable under a trailing append", () => {
    const msgs = [userMsg("t"), ...pairs(6, "p")];
    const a = demoteMessages(msgs, memArchive(), opts());
    const b = demoteMessages(msgs, memArchive(), opts());
    expect(JSON.stringify(b.messages)).toBe(JSON.stringify(a.messages));

    const extended = demoteMessages([...msgs, userMsg("and now?")], memArchive(), opts());
    expect(JSON.stringify(extended.messages.slice(0, msgs.length))).toBe(JSON.stringify(a.messages));
  });

  it("preserves message structure and ignores non-shell results", () => {
    const msgs: any[] = [
      userMsg("t"),
      {
        role: "toolResult",
        toolCallId: "b9",
        toolName: "BrowserNavigate",
        content: [{ type: "text", text: `navigated\n${filler(8192, "page")}` }],
        isError: false,
        timestamp: 1,
      },
      {
        role: "assistant",
        content: [
          { type: "text", text: "running it" },
          { type: "toolCall", id: "k1", name: "ShellSession", arguments: { command: "make", timeout: 120 } },
        ],
      },
      shellResult("k1", filler(8192, "make")),
      ...pairs(4, "n"),
    ];
    const out = demoteMessages(msgs, memArchive(), opts());

    expect(out.messages[1]).toEqual(msgs[1]);
    const call = out.messages[2] as any;
    expect(call.content[0]).toEqual({ type: "text", text: "running it" });
    expect(call.content[1].id).toBe("k1");
    expect(call.content[1].name).toBe("ShellSession");
    expect(call.content[1].arguments.timeout).toBe(120);
    const res = out.messages[3] as any;
    expect(res.toolCallId).toBe("k1");
    expect(res.toolName).toBe("ShellSession");
    expect(res.isError).toBe(false);
    expect(res.timestamp).toBe(1700000000000);
  });

  it("demotes a ShellLog result and notes that it re-pages", () => {
    const msgs = [
      userMsg("t"),
      {
        role: "assistant",
        content: [{ type: "toolCall", id: "L1", name: "ShellLog", arguments: { id: "job-1", lines: 400 } }],
      },
      shellResult("L1", filler(12000, "job"), null, "ShellLog"),
      ...pairs(4, "n"),
    ];
    const out = demoteMessages(msgs, memArchive(), opts());
    const demoted = textOf(out.messages[2]);
    expect(demoted).toContain(RESULT_DEMOTED_PREFIX);
    expect(demoted).toContain("re-pages the live job buffer");
    expect((out.messages[1] as any).content[0].arguments).toEqual({ id: "job-1", lines: 400 });
  });

  it("never rewrites the command of a tool call carrying a thoughtSignature", () => {
    const command = `cat > /tmp/b.c <<'EOF'\n${filler(8192, "src")}\nEOF\ngcc -o /tmp/b /tmp/b.c`;
    const msgs = [
      userMsg("t"),
      assistantShell("sig", command, { thoughtSignature: "CtYBAbc123" }),
      shellResult("sig", filler(8192, "out")),
      ...pairs(4, "n"),
    ];
    const out = demoteMessages(msgs, memArchive(), opts());

    expect(commandOf(out.messages[1])).toBe(command);
    expect((out.messages[1] as any).content[0].thoughtSignature).toBe("CtYBAbc123");
    // Result-side demotion stays allowed for a signed call.
    const demoted = textOf(out.messages[2]);
    expect(demoted).toContain(RESULT_DEMOTED_PREFIX);
    expect(demoted).toContain(archiveId("sig"));
  });

  it("handles a result whose tool call is missing from history", () => {
    const msgs = [userMsg("t"), shellResult("orphan", filler(8192, "o")), ...pairs(4, "n")];
    const out = demoteMessages(msgs, memArchive(), opts());
    expect(textOf(out.messages[1])).toContain(RESULT_DEMOTED_PREFIX);
  });

  it("keeps only the tail when the head budget is set to zero", () => {
    const body = `${filler(6000, "warn")}\nexit=139`;
    const msgs = [userMsg("t"), assistantShell("z", "./a.out"), shellResult("z", body), ...pairs(4, "n")];
    const out = demoteMessages(msgs, memArchive(), opts({ keepResultHeadBytes: 0 }));

    const demoted = textOf(out.messages[2]);
    expect(demoted).toContain("exit=139");
    expect(demoted).not.toContain("warn line 0 ");
    expect(Buffer.byteLength(demoted)).toBeLessThan(1024);
  });

  // A final content line at/near the tail budget's own byte size makes the
  // line-boundary search land exactly at buf.length, which can collapse the
  // kept tail to nothing even though the tail-keep budget is non-zero.
  it("keeps a real failure signal even when it sits on a line near the tail budget's size", () => {
    // Marker at the line's END: it must survive even under the raw-byte-slice
    // fallback a line longer than the tail budget falls back to, isolating
    // the boundary bug (tail collapsing to nothing) from that separate,
    // already-correct partial-slice behavior.
    for (const len of [254, 255, 256, 300]) {
      const failureLine = "x".repeat(Math.max(0, len - 16)) + "SEGFAULT-MARKER";
      // Trailing "\n" matters: real ShellSession output has a blank line
      // before its footer (formatOutput preserves split("\n")'s trailing
      // empty element), which is exactly the shape the boundary bug needs.
      const body = `${filler(10000, "out")}\n${failureLine}\n`;
      const msgs = [userMsg("t"), assistantShell("z", "./a.out"), shellResult("z", body), ...pairs(4, "n")];
      const out = demoteMessages(msgs, memArchive(), opts());
      expect(textOf(out.messages[2])).toContain("SEGFAULT-MARKER");
    }
  });

  // An unbounded "last line is the footer" assumption lets a giant single
  // line (no newline at all, or a huge final line) get treated as the
  // footer and copied verbatim, defeating demotion entirely.
  it("still demotes a giant single-line result with no footer at all", () => {
    const body = "z".repeat(200000); // no newline anywhere — GAIA's bash has no [exit=…] footer
    const msgs = [userMsg("t"), assistantShell("z", "minify.sh"), shellResult("z", body, null, "bash"), ...pairs(4, "n")];
    const out = demoteMessages(msgs, memArchive(), opts());
    expect(out.demotedCount).toBeGreaterThan(0);
    expect(Buffer.byteLength(textOf(out.messages[2]))).toBeLessThan(2000);
  });

  it("still demotes a result whose final line is itself huge", () => {
    const body = `${filler(2000, "out")}\n${"z".repeat(300000)}`;
    const msgs = [userMsg("t"), assistantShell("z", "cat recalled"), shellResult("z", body, null, "ShellRecall"), ...pairs(4, "n")];
    const out = demoteMessages(msgs, memArchive(), opts());
    expect(Buffer.byteLength(textOf(out.messages[2]))).toBeLessThan(2000);
  });

  // This repo is self-hosted, so a command that heredocs retention.ts's own
  // source can contain CMD_DEMOTED_INFIX as plain text. That must not be
  // mistaken for an actual demotion marker and permanently skip the pair.
  it("does not mistake a command that merely quotes the demotion marker text for an already-demoted one", () => {
    const command = `cat > retention.ts <<'EOF'\nexport const CMD_DEMOTED_INFIX = "${CMD_DEMOTED_INFIX}";\n${filler(4000, "src")}\nEOF`;
    const msgs = [userMsg("t"), assistantShell("z", command), shellResult("z", "ok"), ...pairs(4, "n")];
    const out = demoteMessages(msgs, memArchive(), opts());
    expect(out.demotedCount).toBe(1);
    expect(commandOf(out.messages[1])).toContain(CMD_DEMOTED_INFIX + archiveId("z"));
  });

  // A complete, validly-shaped marker copied from a DIFFERENT pair (e.g.
  // quoted as an example) must not be mistaken for this pair's own — the
  // marker's id has to match archiveId(p.toolCallId), not just look right.
  it("does not mistake another pair's marker, copied verbatim, for this pair's own", () => {
    const copiedMarker = `[... 5.2KB ${CMD_DEMOTED_INFIX}sr-1234567890abcdef ...]`;
    const command = `cat > notes.md <<EOF\nExample:\n${copiedMarker}\n${filler(4000, "src")}\nEOF`;
    const msgs = [userMsg("t"), assistantShell("victim", command), shellResult("victim", "ok"), ...pairs(4, "n")];
    const out = demoteMessages(msgs, memArchive(), opts());
    expect(commandOf(out.messages[1])).not.toBe(command);
    expect(commandOf(out.messages[1])).toContain(CMD_DEMOTED_INFIX + archiveId("victim"));
  });

  // Real output that merely starts with the prefix text and separately
  // mentions this pair's own id must still be demoted — the check requires
  // the complete, exact marker line, not "starts with X and contains Y".
  it("does not mistake real output that merely resembles the result marker for an already-demoted one", () => {
    const id = archiveId("victim2");
    const resultText =
      `${RESULT_DEMOTED_PREFIX} mode] test output for debugging shell-retention.ts\n` +
      `checking id format: ShellRecall id=${id} looks right\n` +
      filler(5000, "out");
    const msgs = [userMsg("t"), assistantShell("victim2", "debug-test"), shellResult("victim2", resultText, null), ...pairs(4, "n")];
    const out = demoteMessages(msgs, memArchive(), opts());
    expect(textOf(out.messages[2])).not.toBe(resultText);
  });

  // A thoughtSignature on a sibling text/thinking block (not the toolCall
  // block itself) must still block the command rewrite — pi's
  // google-shared.js states the signature "can appear on ANY part type".
  it("skips the command rewrite when the signature sits on a sibling text block, not the tool call", () => {
    const command = `cat > /tmp/a.c <<'EOF'\n${filler(8192, "src")}\nEOF\ngcc -o /tmp/a /tmp/a.c && echo compiled`;
    const msgs = [
      userMsg("t"),
      {
        role: "assistant",
        content: [
          { type: "text", text: "Let's compile it.", textSignature: "CtYBsibling123" },
          { type: "toolCall", id: "sig2", name: "ShellSession", arguments: { command } },
        ],
      },
      shellResult("sig2", filler(8192, "out")),
      ...pairs(4, "n"),
    ];
    const out = demoteMessages(msgs, memArchive(), opts());
    const assistantOut = out.messages[1] as any;
    expect(assistantOut.content[1].arguments.command).toBe(command);
    // Result-side demotion stays allowed regardless of the sibling signature.
    expect(textOf(out.messages[2])).toContain(RESULT_DEMOTED_PREFIX);
  });

  // pi-ai's openai-completions provider stores the name of the delta field
  // the reasoning arrived in as thinkingSignature — a replay hint, not a
  // signature.
  function thinkingThenShell(
    id: string,
    command: string,
    thinkingSignature: string,
    callExtra = {},
    api: string | null = "openai-completions",
  ) {
    return {
      role: "assistant",
      ...(api === null ? {} : { api }),
      content: [
        { type: "thinking", thinking: "Compile it next.", thinkingSignature },
        { type: "toolCall", id, name: "ShellSession", arguments: { command }, ...callExtra },
      ],
    };
  }

  for (const field of ["reasoning_content", "reasoning", "reasoning_text"]) {
    it(`demotes the command when thinkingSignature is openai-completions' "${field}" field name`, () => {
      const command = `cat > /tmp/c.c <<'EOF'\n${filler(8192, "src")}\nEOF\ngcc -o /tmp/c /tmp/c.c`;
      const msgs = [
        userMsg("t"),
        thinkingThenShell("rc", command, field),
        shellResult("rc", filler(8192, "out")),
        ...pairs(4, "n"),
      ];
      const out = demoteMessages(msgs, memArchive(), opts());
      const assistantOut = out.messages[1] as any;
      expect(assistantOut.content[1].arguments.command).toContain(archiveId("rc"));
      expect(assistantOut.content[0]).toEqual(msgs[1].content[0]);
    });
  }

  for (const [label, api] of [["another provider", "anthropic-messages"], ["no api field", null]] as const) {
    it(`never rewrites the command for a field-name thinkingSignature from ${label}`, () => {
      const command = `cat > /tmp/g.c <<'EOF'\n${filler(8192, "src")}\nEOF\ngcc -o /tmp/g /tmp/g.c`;
      const msgs = [
        userMsg("t"),
        thinkingThenShell("xp", command, "reasoning", {}, api),
        shellResult("xp", filler(8192, "out")),
        ...pairs(4, "n"),
      ];
      const out = demoteMessages(msgs, memArchive(), opts());
      expect((out.messages[1] as any).content[1].arguments.command).toBe(command);
      expect(textOf(out.messages[2])).toContain(RESULT_DEMOTED_PREFIX);
    });
  }

  it("never rewrites the command when thinkingSignature is an opaque signature", () => {
    const command = `cat > /tmp/d.c <<'EOF'\n${filler(8192, "src")}\nEOF\ngcc -o /tmp/d /tmp/d.c`;
    const msgs = [
      userMsg("t"),
      thinkingThenShell("op", command, "EqQBCkgIAxABGAIiQL2x9signedBlob"),
      shellResult("op", filler(8192, "out")),
      ...pairs(4, "n"),
    ];
    const out = demoteMessages(msgs, memArchive(), opts());
    expect((out.messages[1] as any).content[1].arguments.command).toBe(command);
    expect(textOf(out.messages[2])).toContain(RESULT_DEMOTED_PREFIX);
  });

  it("never rewrites the command when a field-name thinkingSignature sits beside an encrypted reasoning detail", () => {
    const command = `cat > /tmp/e.c <<'EOF'\n${filler(8192, "src")}\nEOF\ngcc -o /tmp/e /tmp/e.c`;
    const detail = JSON.stringify({ type: "reasoning.encrypted", id: "en", data: "gAAAAB" });
    const msgs = [
      userMsg("t"),
      thinkingThenShell("en", command, "reasoning", { thoughtSignature: detail }),
      shellResult("en", filler(8192, "out")),
      ...pairs(4, "n"),
    ];
    const out = demoteMessages(msgs, memArchive(), opts());
    expect((out.messages[1] as any).content[1].arguments.command).toBe(command);
  });

  it("honors env overrides for retainRaw and the size floor", () => {
    const prevRetain = process.env[ENV_RETAIN_RAW];
    const prevFloor = process.env[ENV_MIN_PAIR_BYTES];
    const prevBatch = process.env[ENV_DEMOTE_BATCH];
    try {
      // Per-pair, so retainRaw=1 shows as exactly 5 of 6 demoted.
      process.env[ENV_DEMOTE_BATCH] = "1";
      const msgs = [userMsg("t"), ...pairs(6, "p")];
      process.env[ENV_RETAIN_RAW] = "1";
      expect(demoteMessages(msgs, memArchive(), resolveOptions()).demotedCount).toBe(5);

      process.env[ENV_MIN_PAIR_BYTES] = "1000000";
      expect(demoteMessages(msgs, memArchive(), resolveOptions()).demotedCount).toBe(0);
    } finally {
      if (prevRetain === undefined) delete process.env[ENV_RETAIN_RAW];
      else process.env[ENV_RETAIN_RAW] = prevRetain;
      if (prevFloor === undefined) delete process.env[ENV_MIN_PAIR_BYTES];
      else process.env[ENV_MIN_PAIR_BYTES] = prevFloor;
      if (prevBatch === undefined) delete process.env[ENV_DEMOTE_BATCH];
      else process.env[ENV_DEMOTE_BATCH] = prevBatch;
    }
  });

});

describe("demotedPrefixLength", () => {
  it("rounds the per-pair target down to whole batches", () => {
    expect([0, 3, 4, 7, 8, 11].map((t) => demotedPrefixLength(t, 4))).toEqual([0, 0, 4, 4, 8, 8]);
  });

  it("is the per-pair target itself for a batch of 1 or anything not a finite number >= 1", () => {
    for (const b of [1, 0, -3, 0.5, NaN, Infinity]) {
      expect(demotedPrefixLength(7, b)).toBe(7);
    }
  });

  it("floors a fractional batch", () => {
    expect(demotedPrefixLength(7, 2.9)).toBe(6);
    expect(demotedPrefixLength(5, 4.5)).toBe(4);
  });

  it("never exceeds the target, never trails it by a whole batch, and never shrinks as the target grows", () => {
    for (let b = 1; b <= 8; b++) {
      let prev = 0;
      for (let t = 0; t <= 40; t++) {
        const m = demotedPrefixLength(t, b);
        expect(m % b).toBe(0);
        expect(m).toBeLessThanOrEqual(t);
        expect(t - m).toBeLessThan(b);
        expect(m).toBeGreaterThanOrEqual(prev);
        prev = m;
      }
    }
  });
});

describe("resolveOptions demoteBatch", () => {
  it("resolves the batch from LITTLE_CODER_SHELL_DEMOTE_BATCH, defaulting to 4", () => {
    const prev = process.env[ENV_DEMOTE_BATCH];
    try {
      expect(ENV_DEMOTE_BATCH).toBe("LITTLE_CODER_SHELL_DEMOTE_BATCH");
      expect(DEFAULT_DEMOTE_BATCH).toBe(4);
      delete process.env[ENV_DEMOTE_BATCH];
      expect(resolveOptions().demoteBatch).toBe(4);
      // envNumber's convention: unset/blank/non-numeric fall back, any finite
      // value passes through; demotedPrefixLength gives <= 0 its meaning.
      for (const [raw, want] of [["1", 1], ["16", 16], ["0", 0], ["-2", -2], ["", 4], ["  ", 4], ["abc", 4]] as const) {
        process.env[ENV_DEMOTE_BATCH] = raw;
        expect(resolveOptions().demoteBatch).toBe(want);
      }
    } finally {
      if (prev === undefined) delete process.env[ENV_DEMOTE_BATCH];
      else process.env[ENV_DEMOTE_BATCH] = prev;
    }
  });
});

describe("demoteMessages with the default batch", () => {
  const B = DEFAULT_DEMOTE_BATCH;
  const R = DEFAULT_RETAIN_RAW;
  const BATCHED = opts({ demoteBatch: DEFAULT_DEMOTE_BATCH });

  it("demotes only whole batches, keeping the newest retainRaw pairs raw", () => {
    // Per-pair rule wants the oldest B + 1; the batch takes the oldest B.
    const msgs = [userMsg("t"), ...pairs(R + B + 1, "p")];
    const out = demoteMessages(msgs, memArchive(), BATCHED);
    expect(out.demotedCount).toBe(B);
    for (let i = 0; i < R + B + 1; i++) {
      const resultIdx = 2 + 2 * i;
      if (i < B) expect(textOf(out.messages[resultIdx])).toContain(RESULT_DEMOTED_PREFIX);
      else expect(out.messages[resultIdx]).toEqual(msgs[resultIdx]);
    }
  });

  it("leaves displaced and stale pairs raw until a whole batch is due (the stranded case)", () => {
    const displaced = [userMsg("t"), ...pairs(R + B - 1, "p")];
    expect(demoteMessages(displaced, memArchive(), BATCHED)).toEqual({ messages: displaced, demotedCount: 0 });

    const loneStale = [userMsg("t"), assistantShell("g", "echo go"), shellResult("g", filler(8192, "big")), ...pairs(60, "tiny", 60)];
    expect(demoteMessages(loneStale, memArchive(), BATCHED).demotedCount).toBe(0);

    const staleBatch = [userMsg("t"), ...pairs(B, "old"), ...pairs(60, "tiny", 60)];
    expect(demoteMessages(staleBatch, memArchive(), BATCHED).demotedCount).toBe(B);
  });

  it("leaves a refused pair raw in its slot without pulling the next pair into the batch", () => {
    const msgs = [userMsg("t"), ...pairs(R + B + 1, "p")];
    const archive = memArchive();
    const refusing: RetentionArchive = {
      ...archive,
      save: (id, text) => id !== archiveId("p0") && archive.save(id, text),
    };
    const out = demoteMessages(msgs, refusing, BATCHED);
    expect(out.demotedCount).toBe(B - 1);
    expect(out.messages[1]).toEqual(msgs[1]);
    expect(out.messages[2]).toEqual(msgs[2]);
    for (let i = 1; i < B; i++) expect(textOf(out.messages[2 + 2 * i])).toContain(RESULT_DEMOTED_PREFIX);
    // Slot B stays raw: the boundary does not move to make up for p0.
    expect(out.messages[2 + 2 * B]).toEqual(msgs[2 + 2 * B]);
    expect(archive.entries.has(archiveId("p0"))).toBe(false);

    expect(demoteMessages(msgs, failingArchive, BATCHED)).toEqual({ messages: msgs, demotedCount: 0 });
  });

  it("is idempotent on its own output even when a non-shrinking pair sits inside the batch", () => {
    // Signed (command never rewritten) and a one-word result (cannot shrink):
    // it qualifies on size, occupies slot 0, and is skipped.
    const command = `cat > /tmp/s.c <<'EOF'\n${filler(4000, "src")}\nEOF\ngcc /tmp/s.c`;
    const msgs = [
      userMsg("t"),
      assistantShell("sig", command, { thoughtSignature: "CtYBsig" }),
      shellResult("sig", "ok"),
      ...pairs(R + 2 * B - 2, "p"),
    ];
    // Slots: sig, p0..p(R+2B-3) -> per-pair target 2B-1 -> prefix B.
    const first = demoteMessages(msgs, memArchive(), BATCHED);
    expect(first.demotedCount).toBe(B - 1);
    expect(first.messages[1]).toEqual(msgs[1]);
    expect(first.messages[2]).toEqual(msgs[2]);
    expect(textOf(first.messages[4 + 2 * (B - 2)])).toContain(RESULT_DEMOTED_PREFIX);
    expect(first.messages[4 + 2 * (B - 1)]).toEqual(msgs[4 + 2 * (B - 1)]);

    const second = demoteMessages(first.messages, memArchive(), BATCHED);
    expect(second.demotedCount).toBe(0);
    expect(second.messages).toEqual(first.messages);
  });

  it("keeps the ShellLog note and never rewrites a signed command under the default batch", () => {
    const log = [
      userMsg("t"),
      { role: "assistant", content: [{ type: "toolCall", id: "L1", name: "ShellLog", arguments: { id: "job-1", lines: 400 } }] },
      shellResult("L1", filler(12000, "job"), null, "ShellLog"),
      ...pairs(R + B, "n"),
    ];
    const logOut = demoteMessages(log, memArchive(), BATCHED);
    expect(logOut.demotedCount).toBe(B);
    expect(textOf(logOut.messages[2])).toContain("re-pages the live job buffer");
    expect((logOut.messages[1] as any).content[0].arguments).toEqual({ id: "job-1", lines: 400 });

    const command = `cat > /tmp/b.c <<'EOF'\n${filler(8192, "src")}\nEOF\ngcc -o /tmp/b /tmp/b.c`;
    const signed = [
      userMsg("t"),
      assistantShell("sig", command, { thoughtSignature: "CtYBAbc123" }),
      shellResult("sig", filler(8192, "out")),
      ...pairs(R + B, "n"),
    ];
    const signedOut = demoteMessages(signed, memArchive(), BATCHED);
    expect(signedOut.demotedCount).toBe(B);
    expect(commandOf(signedOut.messages[1])).toBe(command);
    expect(textOf(signedOut.messages[2])).toContain(archiveId("sig"));
  });

  it("does not treat another pair's copied marker as this pair's own under the default batch", () => {
    const copiedMarker = `[... 5.2KB ${CMD_DEMOTED_INFIX}sr-1234567890abcdef ...]`;
    const command = `cat > notes.md <<EOF\nExample:\n${copiedMarker}\n${filler(4000, "src")}\nEOF`;
    const msgs = [userMsg("t"), assistantShell("victim", command), shellResult("victim", "ok"), ...pairs(R + B, "n")];
    const out = demoteMessages(msgs, memArchive(), BATCHED);
    expect(out.demotedCount).toBe(B);
    expect(commandOf(out.messages[1])).toContain(CMD_DEMOTED_INFIX + archiveId("victim"));
  });

  it("applies LITTLE_CODER_SHELL_DEMOTE_BATCH through resolveOptions, treating <= 0 as per-pair", () => {
    const prev = process.env[ENV_DEMOTE_BATCH];
    const msgs = [userMsg("t"), ...pairs(R + B + 1, "p")]; // per-pair rule wants B + 1
    const count = () => demoteMessages(msgs, memArchive(), resolveOptions()).demotedCount;
    try {
      delete process.env[ENV_DEMOTE_BATCH];
      expect(count()).toBe(B);
      for (const v of ["1", "0", "-2"]) {
        process.env[ENV_DEMOTE_BATCH] = v;
        expect(count()).toBe(B + 1);
      }
      process.env[ENV_DEMOTE_BATCH] = "abc";
      expect(count()).toBe(B);
      process.env[ENV_DEMOTE_BATCH] = String(2 * B);
      expect(count()).toBe(0);
    } finally {
      if (prev === undefined) delete process.env[ENV_DEMOTE_BATCH];
      else process.env[ENV_DEMOTE_BATCH] = prev;
    }
  });
});

// ── Turn-by-turn prefix-break simulation ───────────────────────────────────
// 120 turns: every 6th is [user, assistant text]; the rest append one shell
// pair [toolCall, toolResult], 40% of them large (3.5–23.5KB by a fixed
// formula), a quarter of the large ones a heredoc'd command with a one-word
// result (so its demotion rewrites the call, not the result). A turn appends at
// most 2 messages and shell results sit at least 2 apart, so the per-pair
// target grows by at most 1 per turn: per-pair demotion breaks the prefix once
// per newly demoted pair, and batched breaks are exactly floor(perPair / B).
function simulatedTurns(): any[][] {
  const turns: any[][] = [];
  let s = 0;
  for (let i = 0; i < 120; i++) {
    if (i % 6 === 5) {
      turns.push([userMsg(`check ${i}`), { role: "assistant", content: [{ type: "text", text: `looks fine at ${i}` }] }]);
      continue;
    }
    const id = `sim${i}`;
    const large = s % 5 < 2;
    const bytes = 3500 + ((s * 7919) % 20000);
    if (large && s % 10 === 1) {
      turns.push([
        assistantShell(id, `cat > /tmp/s${i}.c <<'EOF'\n${filler(bytes, `src${i}`)}\nEOF\ngcc /tmp/s${i}.c`),
        shellResult(id, "compiled"),
      ]);
    } else {
      turns.push([assistantShell(id, `make step-${i}`), shellResult(id, filler(large ? bytes : 120, `out${i}`))]);
    }
    s++;
  }
  return turns;
}

/** Pairs at or over the size floor, oldest first — computed from the fixture, not the code under test. */
function qualifyingSlots(history: any[], o: RetentionOptions): { id: string; resultIdx: number }[] {
  const commands = new Map<string, string>();
  for (const m of history) {
    if (m.role !== "assistant") continue;
    for (const b of m.content) if (b.type === "toolCall") commands.set(b.id, b.arguments.command);
  }
  const slots: { id: string; resultIdx: number }[] = [];
  history.forEach((m, resultIdx) => {
    if (m.role !== "toolResult") return;
    const bytes = Buffer.byteLength(commands.get(m.toolCallId) ?? "") + Buffer.byteLength(textOf(m));
    if (bytes >= o.minPairBytes) slots.push({ id: m.toolCallId, resultIdx });
  });
  return slots;
}

/** Today's per-pair rule, restated independently: rank >= retainRaw or distance > staleDistance. */
function legacyDemotedIds(history: any[], o: RetentionOptions): string[] {
  const slots = qualifyingSlots(history, o);
  return slots
    .filter((s, k) => slots.length - 1 - k >= o.retainRaw || history.length - 1 - s.resultIdx > o.staleDistance)
    .map((s) => s.id);
}

/** Ids whose result or command carries this pair's own demotion marker, in history order. */
function demotedIdsIn(projection: any[]): string[] {
  const commands = new Map<string, string>();
  for (const m of projection) {
    if (m.role !== "assistant") continue;
    for (const b of m.content) if (b.type === "toolCall") commands.set(b.id, b.arguments.command);
  }
  return projection
    .filter((m) => m.role === "toolResult")
    .filter(
      (m) =>
        textOf(m).startsWith(RESULT_DEMOTED_PREFIX) ||
        (commands.get(m.toolCallId) ?? "").includes(CMD_DEMOTED_INFIX + archiveId(m.toolCallId)),
    )
    .map((m) => m.toolCallId);
}

/** The first message a pair's demotion rewrites: its call if the command changed, else its result. */
function firstTouchedIndex(history: any[], projection: any[], id: string): number {
  const callIdx = history.findIndex(
    (m) => m.role === "assistant" && m.content.some((b: any) => b.type === "toolCall" && b.id === id),
  );
  const resultIdx = history.findIndex((m) => m.role === "toolResult" && m.toolCallId === id);
  return JSON.stringify(projection[callIdx]) !== JSON.stringify(history[callIdx]) ? callIdx : resultIdx;
}

interface SimStep {
  history: any[];
  projection: any[];
  demoted: string[];
  breakAt: number | null;
}

/** Grow the conversation turn by turn, projecting each prefix the way pi does (fresh copy per call). */
function runSimulation(o: RetentionOptions): { breaks: number; steps: SimStep[] } {
  const archive = memArchive();
  const history: any[] = [];
  const steps: SimStep[] = [];
  let breaks = 0;
  for (const turn of simulatedTurns()) {
    history.push(...turn);
    const projection = demoteMessages([...history], archive, o).messages;
    const prev = steps.at(-1)?.projection;
    let breakAt: number | null = null;
    if (prev) {
      for (let i = 0; i < prev.length; i++) {
        if (projection[i] !== prev[i] && JSON.stringify(projection[i]) !== JSON.stringify(prev[i])) {
          breakAt = i;
          break;
        }
      }
    }
    if (breakAt !== null) breaks++;
    steps.push({ history: [...history], projection, demoted: demotedIdsIn(projection), breakAt });
  }
  return { breaks, steps };
}

describe("demoteMessages prefix-cache stability (simulated session)", () => {
  it("cuts mid-history prefix breaks by k = DEFAULT_DEMOTE_BATCH against per-pair demotion", () => {
    const perPair = runSimulation(opts({ demoteBatch: 1 }));
    const batched = runSimulation(opts({ demoteBatch: DEFAULT_DEMOTE_BATCH }));

    // batch=1 is today's behaviour: the same demoted set as the per-pair rule
    // on every turn, and one break per turn at which that set grows.
    let growth = 0;
    for (let t = 0; t < perPair.steps.length; t++) {
      const { history, demoted } = perPair.steps[t];
      const legacy = legacyDemotedIds(history, opts());
      expect(demoted).toEqual(legacy);
      if (t > 0 && legacy.length > perPair.steps[t - 1].demoted.length) growth++;
    }
    expect(perPair.breaks).toBe(growth);
    expect(perPair.breaks).toBeGreaterThanOrEqual(30);

    expect(batched.breaks).toBe(Math.floor(perPair.breaks / DEFAULT_DEMOTE_BATCH));
    expect(batched.breaks * DEFAULT_DEMOTE_BATCH).toBeLessThanOrEqual(perPair.breaks);
  });

  it("grows the demoted set only in whole-batch prefixes and keeps earlier history byte-stable between jumps", () => {
    const o = opts({ demoteBatch: DEFAULT_DEMOTE_BATCH });
    const { steps } = runSimulation(o);
    let jumps = 0;
    for (let t = 1; t < steps.length; t++) {
      const { history, projection, demoted, breakAt } = steps[t];
      const before = steps[t - 1].demoted;
      const slots = qualifyingSlots(history, o).map((s) => s.id);
      const legacy = legacyDemotedIds(history, o);

      expect(demoted).toEqual(slots.slice(0, demoted.length));
      expect(demoted.length).toBe(demotedPrefixLength(legacy.length, DEFAULT_DEMOTE_BATCH));
      expect(legacy.slice(0, demoted.length)).toEqual(demoted);
      expect(slots.slice(-DEFAULT_RETAIN_RAW).some((id) => demoted.includes(id))).toBe(false);
      expect(demoted.slice(0, before.length)).toEqual(before);

      if (demoted.length === before.length) {
        expect(breakAt).toBeNull();
      } else {
        jumps++;
        expect(breakAt).toBe(firstTouchedIndex(history, projection, demoted[before.length]));
      }
    }
    expect(jumps).toBeGreaterThanOrEqual(5);
  });
});

describe("demoteMessages per-pair (batch 1) byte identity", () => {
  // Hashes captured against the pre-batching implementation: demoteBatch 1 must
  // keep producing exactly those bytes, not merely the same demoted set.
  it("matches the legacy per-pair output on representative histories", () => {
    const heredoc = `cat > /tmp/redec.c <<'EOF'\n${filler(6000, "src")}\nEOF\ngcc -O2 -o /tmp/redec /tmp/redec.c`;
    const histories: Record<string, any[]> = {
      sixPairs: [userMsg("build it"), ...pairs(6, "p")],
      heredoc: [userMsg("t"), assistantShell("h", heredoc), shellResult("h", "compiled"), ...pairs(4, "n")],
      shellLog: [
        userMsg("t"),
        { role: "assistant", content: [{ type: "toolCall", id: "L1", name: "ShellLog", arguments: { id: "job-1" } }] },
        shellResult("L1", filler(12000, "job"), null, "ShellLog"),
        ...pairs(4, "n"),
      ],
      signed: [
        userMsg("t"),
        assistantShell("sig", heredoc, { thoughtSignature: "CtYBAbc123" }),
        shellResult("sig", filler(8192, "out")),
        ...pairs(4, "n"),
      ],
      loneStale: [userMsg("t"), assistantShell("g", "echo go"), shellResult("g", filler(8192, "big")), ...pairs(60, "tiny", 60)],
    };
    const digests = Object.fromEntries(
      Object.entries(histories).map(([name, msgs]) => {
        const out = demoteMessages(msgs, memArchive(), opts({ demoteBatch: 1 }));
        return [name, `${out.demotedCount}:${createHash("sha256").update(JSON.stringify(out.messages)).digest("hex").slice(0, 16)}`];
      }),
    );
    expect(digests).toMatchInlineSnapshot(`
      {
        "heredoc": "1:6466a4e3e5aaca2c",
        "loneStale": "1:29d5243032ffeaab",
        "shellLog": "1:77ebc93cc7dbaeb7",
        "signed": "1:1d6b1acbd560c65d",
        "sixPairs": "2:d6d011296659d274",
      }
    `);
  });
});

describe("recallSlice", () => {
  const recallOpts = { defaultBytes: DEFAULT_RECALL_BYTES, maxBytes: DEFAULT_RECALL_MAX };
  const archived = archiveText("echo hi", filler(60000, "body"));

  function archiveOf(id: string, text: string): RetentionArchive {
    const a = memArchive();
    a.save(id, text);
    return a;
  }

  it("returns the default slice from the start", () => {
    const out = recallSlice("sr-abcd1234", archiveOf("sr-abcd1234", archived), undefined, undefined, recallOpts);
    expect(out.isError).toBe(false);
    expect(out.text.split("\n")[0]).toBe(
      `[sr-abcd1234 bytes 0–${DEFAULT_RECALL_BYTES} of ${Buffer.byteLength(archived)}]`,
    );
    expect(out.text).toContain("has_more=true");
  });

  it("honors explicit offset and byte count", () => {
    const out = recallSlice("sr-abcd1234", archiveOf("sr-abcd1234", archived), 100, 50, recallOpts);
    expect(out.text.split("\n")[0]).toBe(`[sr-abcd1234 bytes 100–150 of ${Buffer.byteLength(archived)}]`);
  });

  it("clamps an oversized request to the recall max", () => {
    const out = recallSlice("sr-abcd1234", archiveOf("sr-abcd1234", archived), 0, 999999, recallOpts);
    expect(out.text.split("\n")[0]).toBe(
      `[sr-abcd1234 bytes 0–${DEFAULT_RECALL_MAX} of ${Buffer.byteLength(archived)}]`,
    );
  });

  it("returns an empty slice for an offset past the end", () => {
    const total = Buffer.byteLength(archived);
    const out = recallSlice("sr-abcd1234", archiveOf("sr-abcd1234", archived), total + 5000, 100, recallOpts);
    expect(out.isError).toBe(false);
    expect(out.text).toBe(`[sr-abcd1234 bytes ${total}–${total} of ${total}]\n`);
  });

  it("errors on an unknown id", () => {
    const out = recallSlice("sr-deadbeef", memArchive(), 0, 100, recallOpts);
    expect(out.isError).toBe(true);
    expect(out.text).toContain("sr-deadbeef");
  });

  it("errors on a missing id", () => {
    expect(recallSlice("", memArchive(), 0, 100, recallOpts).isError).toBe(true);
  });

  // The bounded-read fix must not change what a caller receives: reading a
  // window slightly wider than requested (for UTF-8 boundary alignment) is
  // an internal implementation detail, not a change to the returned slice.
  it("reads only a bounded window from the archive, not the whole thing", () => {
    let maxRequested = 0;
    const bounded: RetentionArchive = {
      save: () => true,
      size: () => Buffer.byteLength(archived),
      readRange(_id, start, length) {
        maxRequested = Math.max(maxRequested, length);
        return Buffer.from(archived, "utf-8").subarray(start, start + length);
      },
    };
    const out = recallSlice("sr-x", bounded, 100, 50, recallOpts);
    expect(out.text.split("\n")[0]).toBe(`[sr-x bytes 100–150 of ${Buffer.byteLength(archived)}]`);
    // A handful of boundary-padding bytes, nowhere near the archive's real size.
    expect(maxRequested).toBeLessThan(200);
  });

  // stat succeeding but the read itself failing (e.g. the file vanished, or a
  // transient fs error) must surface as an error, not a fake empty page that
  // reads as "this archived pair had no content."
  it("errors when the archive can be sized but not read", () => {
    const flaky: RetentionArchive = {
      save: () => true,
      size: () => Buffer.byteLength(archived),
      readRange: () => undefined,
    };
    const out = recallSlice("sr-x", flaky, 0, 100, recallOpts);
    expect(out.isError).toBe(true);
    expect(out.text).toContain("sr-x");
  });
});

// The field-name exemption mirrors pi-ai internals that are not an exported
// contract. These run the installed provider so a pi-ai change that alters
// what lands in thinkingSignature fails here instead of silently bringing
// back undemotable commands.
describe("pi-ai openai-completions thinkingSignature", () => {
  const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "../../..");
  const piAiDir = [
    "node_modules/@earendil-works/pi-coding-agent/node_modules/@earendil-works/pi-ai",
    "node_modules/@earendil-works/pi-ai",
  ].map((p) => join(repoRoot, p)).find((p) => existsSync(join(p, "dist/api/openai-completions.js")));
  const providerFile = piAiDir ? join(piAiDir, "dist/api/openai-completions.js") : "";

  it("finds the installed provider", () => {
    expect(piAiDir).toBeDefined();
  });

  it("pins the provider's reasoning field list to REASONING_FIELD_NAMES", () => {
    const src = readFileSync(providerFile, "utf8");
    const m = /const reasoningFields = (\[[^\]]*\]);/.exec(src);
    expect(m, "reasoningFields literal not found in openai-completions.js").not.toBeNull();
    const fields: string[] = JSON.parse(m![1]);
    expect(fields.length).toBeGreaterThan(0);
    for (const f of fields) expect(REASONING_FIELD_NAMES.has(f)).toBe(true);
  });

  function sse(chunks: unknown[]): Response {
    const body = [...chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`), "data: [DONE]\n\n"].join("");
    return new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } });
  }

  async function streamedAssistant(field: string, id: string, command: string): Promise<any> {
    const { stream } = await import(pathToFileURL(providerFile).href);
    const chunk = (delta: Record<string, unknown>, finish: string | null = null) => ({
      id: "c1", object: "chat.completion.chunk", created: 0, model: "local",
      choices: [{ index: 0, delta, finish_reason: finish }],
    });
    const model = {
      id: "local", name: "local", api: "openai-completions", provider: "omlx",
      baseUrl: "http://127.0.0.1:1/v1", reasoning: true, input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 32768, maxTokens: 4096,
    };
    const fetch = async () => sse([
      chunk({ role: "assistant", [field]: "Compile it next." }),
      chunk({ tool_calls: [{ index: 0, id, type: "function", function: { name: "ShellSession", arguments: JSON.stringify({ command }) } }] }),
      chunk({}, "tool_calls"),
    ]);
    const s = stream(model, { messages: [{ role: "user", content: "t", timestamp: 0 }] }, { apiKey: "x", fetch });
    const msg = await s.result();
    expect(msg.stopReason).toBe("toolUse");
    return msg;
  }

  for (const field of REASONING_FIELD_NAMES) {
    it(`demotes the command of a real streamed "${field}" turn`, async () => {
      const command = `cat > /tmp/f.c <<'EOF'\n${filler(8192, "src")}\nEOF\ngcc -o /tmp/f /tmp/f.c`;
      const assistant = await streamedAssistant(field, "live", command);
      expect(assistant.content.some((b: any) => b.type === "thinking")).toBe(true);
      const msgs = [userMsg("t"), assistant, shellResult("live", filler(8192, "out")), ...pairs(4, "n")];
      const out = demoteMessages(msgs, memArchive(), opts());
      const call = (out.messages[1] as any).content.find((b: any) => b.type === "toolCall");
      expect(call.arguments.command).toContain(archiveId("live"));
    });
  }
});
