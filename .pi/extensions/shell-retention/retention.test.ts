import { describe, it, expect } from "vitest";
import {
  archiveId,
  archiveText,
  demoteMessages,
  recallSlice,
  resolveOptions,
  CMD_DEMOTED_INFIX,
  DEFAULT_CMD_KEEP,
  DEFAULT_KEEP_RESULT_HEAD,
  DEFAULT_KEEP_RESULT_TAIL,
  DEFAULT_MIN_PAIR_BYTES,
  DEFAULT_RECALL_BYTES,
  DEFAULT_RECALL_MAX,
  DEFAULT_RETAIN_RAW,
  DEFAULT_STALE_DISTANCE,
  ENV_MIN_PAIR_BYTES,
  ENV_RETAIN_RAW,
  RESULT_DEMOTED_PREFIX,
  type RetentionArchive,
  type RetentionOptions,
} from "./retention.ts";

// Canned message shapes mirror pi's AgentMessage / ToolResultMessage.
// See node_modules/@earendil-works/pi-ai/dist/types.d.ts for the real types.

const FOOTER = "[exit=0 cwd=/app timed_out=false backend=harbor-env]";

const BASE: RetentionOptions = {
  retainRaw: DEFAULT_RETAIN_RAW,
  minPairBytes: DEFAULT_MIN_PAIR_BYTES,
  staleDistance: DEFAULT_STALE_DISTANCE,
  keepResultHeadBytes: DEFAULT_KEEP_RESULT_HEAD,
  keepResultTailBytes: DEFAULT_KEEP_RESULT_TAIL,
  cmdKeepBytes: DEFAULT_CMD_KEEP,
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

  it("honors env overrides for retainRaw and the size floor", () => {
    const prevRetain = process.env[ENV_RETAIN_RAW];
    const prevFloor = process.env[ENV_MIN_PAIR_BYTES];
    try {
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
    }
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
