import { describe, it, expect, afterEach } from "vitest";
import setupTruncatedView from "./index.ts";
import {
  annotate,
  detectTrailingTruncator,
  partialViewNote,
  reconstructLineCount,
  splitFooter,
  MIN_BYTES_ENV,
  MIN_LINES_ENV,
} from "./truncation.ts";
import { MAX_RAW_HEAD_BYTES, MAX_RAW_TAIL_BYTES } from "../shell-session/helpers.ts";

// The command from the failing raman-fitting trial (harbor run
// 2026-09-14__23-29-55, live log line 534), with a decoy `head -999` added to
// the heredoc body. `head -50` cut the printed rows 130 x-units short of the
// peak the script's own comment calls real, and the model's next message
// declared that peak noise.
const EVIDENCE_COMMAND = [
  "cd /app",
  "cat > /tmp/explore2.py << 'PYEOF'",
  "import numpy as np",
  "data = np.loadtxt('/tmp/data.txt')",
  "x = data[:,0]; y = data[:,1]",
  "idx = np.argsort(x); x = x[idx]; y = y[idx]",
  "# The huge peak at 19140 dominates everything.",
  "# The peak at 3745 (y=12920) is a real peak!",
  "# decoy for the parser: foo | head -999",
  'print("=== Peak at 3745 ===")',
  "m=(x>=3500)&(x<=4000)",
  "sm=x[m]; vm=y[m]",
  "for i in range(len(sm)):",
  '    print(f"x={sm[i]:.1f} y={vm[i]:.1f}")',
  "PYEOF",
  "python3 /tmp/explore2.py 2>&1 | head -50",
].join("\n");

const HARBOR_FOOTER = "[exit=0 cwd=/app timed_out=false backend=harbor-env]";

/** `n` rows shaped like the evidence output, followed by a trailing newline. */
function rows(n: number): string {
  return Array.from({ length: n }, (_, i) => `x=${3501 + i * 2.3} y=${1100 + i}`).join("\n") + "\n";
}

describe("detectTrailingTruncator", () => {
  // One vector pinning four things at once: the pipeline form, the
  // obsolescent `-50`, heredoc stripping (the body holds a decoy `head -999`,
  // an `&`, and a `|`), and `2>&1` not splitting the final segment.
  it("finds the trailing head in the real evidence command", () => {
    expect(detectTrailingTruncator(EVIDENCE_COMMAND)).toEqual({
      tool: "head",
      unit: "lines",
      limit: 50,
    });
  });

  it("finds the direct-file form and resolves a path to its basename", () => {
    expect(detectTrailingTruncator("head -50 file.txt")).toEqual({
      tool: "head",
      unit: "lines",
      limit: 50,
    });
    expect(detectTrailingTruncator("/usr/bin/head -50 f")).toEqual({
      tool: "head",
      unit: "lines",
      limit: 50,
    });
  });

  it.each([
    ["head -n 50 f", { tool: "head", unit: "lines", limit: 50 }],
    ["head -n50 f", { tool: "head", unit: "lines", limit: 50 }],
    ["head --lines=50 f", { tool: "head", unit: "lines", limit: 50 }],
    ["head --lines 50 f", { tool: "head", unit: "lines", limit: 50 }],
    ["head -q -50 f g", { tool: "head", unit: "lines", limit: 50 }],
    ["foo | head", { tool: "head", unit: "lines", limit: 10 }],
    ["tail -20 log", { tool: "tail", unit: "lines", limit: 20 }],
    ["head -c 4096 f", { tool: "head", unit: "bytes", limit: 4096 }],
    ["head -c4096 f", { tool: "head", unit: "bytes", limit: 4096 }],
    ["head --bytes=4096 f", { tool: "head", unit: "bytes", limit: 4096 }],
    // Last count flag wins, as in coreutils.
    ["head -10 -n 50 f", { tool: "head", unit: "lines", limit: 50 }],
  ])("parses %j", (command, expected) => {
    expect(detectTrailingTruncator(command)).toEqual(expected);
  });

  it("takes the trailing stage when two truncators are chained", () => {
    expect(detectTrailingTruncator("python3 g.py | head -60 | tail -50")).toEqual({
      tool: "tail",
      unit: "lines",
      limit: 50,
    });
  });

  it.each([
    "tail -n +5 f", // prints from line 5 to EOF, not an end-truncation
    "head -n -5 f", // GNU all-but-last
    "tail -f log", // streaming
    "tail -f -n 50 log",
    "tail --follow=name log",
    "head -c 1K f", // size suffixes deferred
    'head -n "50" f', // splitWords keeps the quotes, so the count won't parse
    "head -qn50 f", // a count hides in a flag this parser can't read
    "cat f | head -50 | grep x", // not the final segment
    "head -5 a.txt && echo done", // not the final segment
    'echo "try: foo | head -50"', // quoted, so the segment is the whole echo
    "ls -la",
    "",
  ])("declines %j", (command) => {
    expect(detectTrailingTruncator(command)).toBeNull();
  });
});

describe("reconstructLineCount", () => {
  it("counts a plain body, ignoring the single trailing newline", () => {
    expect(reconstructLineCount(rows(50))).toEqual({ count: 50, unknownCut: false });
    expect(reconstructLineCount("")).toEqual({ count: 0, unknownCut: false });
  });

  it("credits a line-truncation marker with the lines it replaced", () => {
    const body = ["a", "b", "  [... 100 lines truncated ...]", "c"].join("\n");
    expect(reconstructLineCount(body)).toEqual({ count: 103, unknownCut: false });
  });

  it("credits a duplicate-collapse marker with the lines it replaced", () => {
    const body = ["a", "  [... 30 duplicate line(s) collapsed ...]", "b"].join("\n");
    expect(reconstructLineCount(body)).toEqual({ count: 32, unknownCut: false });
  });

  it("flags a byte-cap marker as an uncountable cut", () => {
    const body = ["a", "  [... 12.3KB truncated ...]", "b"].join("\n");
    expect(reconstructLineCount(body)).toEqual({ count: 2, unknownCut: true });
  });

  it("drops the overflow-file note, which is not command output", () => {
    const body = ["a", "b", "Full output: /tmp/lc-shell-x/1-2-3.log"].join("\n");
    expect(reconstructLineCount(body)).toEqual({ count: 2, unknownCut: false });
  });
});

describe("splitFooter", () => {
  it("separates the shell-session footer from the body", () => {
    expect(splitFooter(`a\nb\n${HARBOR_FOOTER}`)).toEqual({
      body: "a\nb",
      footer: HARBOR_FOOTER,
    });
  });

  it("treats a footerless result as all body", () => {
    expect(splitFooter("a\nb")).toEqual({ body: "a\nb", footer: null });
  });
});

describe("annotate", () => {
  afterEach(() => {
    delete process.env[MIN_LINES_ENV];
    delete process.env[MIN_BYTES_ENV];
  });

  const HEAD_50_NOTE =
    "[partial view: this command ends in 'head -50' and the output reaches 50 lines, " +
    "so the source may continue past the last line shown. Don't conclude something is " +
    "absent from this view — re-run with a larger limit or a targeted filter (grep/awk) " +
    "if it matters.]";

  it("annotates the real evidence result above the footer", () => {
    const text = `=== Peak at 3745 ===\n${rows(49)}${HARBOR_FOOTER}`;
    const out = annotate(EVIDENCE_COMMAND, text);
    expect(out).not.toBeNull();
    const lines = out!.split("\n");
    expect(lines[lines.length - 1]).toBe(HARBOR_FOOTER);
    expect(lines[lines.length - 2]).toBe(HEAD_50_NOTE);
    // The body is untouched and the footer stays the only `[exit=` match.
    expect(out!.startsWith(text.slice(0, text.length - HARBOR_FOOTER.length))).toBe(true);
    expect(out!.split("[exit=")).toHaveLength(2);
  });

  // Finding A. This command's own output is three stages deep, so how many
  // lines the trailing `head -20` returned is unknowable from the text — the
  // note must claim only the command's shape and the output's size. Matches
  // the shape at live log line 605 of the evidence trial.
  it("claims nothing about what the truncator stage itself returned", () => {
    const command = [
      "ls -la /app",
      "find /app -name '*.csv' 2>/dev/null",
      "find /app -name '*.txt' 2>/dev/null | head -20",
    ].join("\n");
    const out = annotate(command, `${rows(25)}${HARBOR_FOOTER}`);
    expect(out).not.toBeNull();
    const note = out!.split("\n").at(-2)!;
    expect(note).toContain("this command ends in 'head -20'");
    expect(note).toContain("the output reaches 20 lines");
    expect(note).not.toMatch(/returned/);
    expect(note).not.toMatch(/full 20/);
  });

  // The certain negative: `head -50` that gave back fewer than 50 lines did
  // not truncate, and no marker or raw_bytes is present to say otherwise.
  it("leaves a result under the limit byte-for-byte unchanged", () => {
    const text = `=== Peak at 3745 ===\n${rows(48)}${HARBOR_FOOTER}`;
    expect(annotate(EVIDENCE_COMMAND, text)).toBeNull();
  });

  // The exactly-at-the-limit boundary, on a body whose last line is real
  // output rather than the empty string a trailing newline leaves behind.
  it("counts a body with no trailing newline as 50 lines, not 51", () => {
    const body = rows(50).trimEnd();
    expect(annotate(EVIDENCE_COMMAND, `${body}\n${HARBOR_FOOTER}`)).not.toBeNull();
    expect(
      annotate(EVIDENCE_COMMAND, `${rows(49).trimEnd()}\n${HARBOR_FOOTER}`),
    ).toBeNull();
  });

  it("stays silent when the limit is below the configured floor", () => {
    process.env[MIN_LINES_ENV] = "60";
    expect(annotate(EVIDENCE_COMMAND, `${rows(50)}${HARBOR_FOOTER}`)).toBeNull();
  });

  it("does not fire for a peek below the default floor", () => {
    expect(annotate("seq 100 | head -5", `${rows(50)}${HARBOR_FOOTER}`)).toBeNull();
  });

  it("fires on bare head, whose coreutils default sits exactly on the floor", () => {
    expect(annotate("seq 100 | head", `${rows(50)}${HARBOR_FOOTER}`)).not.toBeNull();
  });

  // The harness cut 100 lines out of a 300-line output, so `head -500` never
  // reached its limit; counting only the visible 200 would get this backwards.
  it("reads through a harness marker to see the limit was not reached", () => {
    const body = `${rows(100)}  [... 100 lines truncated ...]\n${rows(100)}`;
    const footer = "[exit=0 cwd=/app timed_out=false output_truncated=true raw_bytes=9000]";
    expect(annotate("seq 300 | head -500", `${body}${footer}`)).toBeNull();
  });

  it("reads through a harness marker to see the limit was reached", () => {
    const body = `${rows(100)}  [... 300 lines truncated ...]\n${rows(100)}`;
    const footer = "[exit=0 cwd=/app timed_out=false output_truncated=true raw_bytes=9000]";
    expect(annotate("seq 800 | head -500", `${body}${footer}`)).not.toBeNull();
  });

  // Finding B: the pre-dedup cap fires iff the raw output exceeded its budget,
  // and the marker it leaves can be dropped by the later line cut.
  it("annotates an uncountable result and says the count is what is missing", () => {
    const raw = MAX_RAW_HEAD_BYTES + MAX_RAW_TAIL_BYTES + 1;
    const footer = `[exit=0 cwd=/app timed_out=false output_truncated=true raw_bytes=${raw}]`;
    const out = annotate("seq 1000000 | head -50", `${rows(10)}${footer}`);
    expect(out).not.toBeNull();
    expect(out!.split("\n").at(-2)).toBe(
      "[partial view: this command ends in 'head -50' and the output was truncated " +
        "before its line count could be established, so the source may continue past " +
        "the last line shown. Don't conclude something is absent from this view — " +
        "re-run with a larger limit or a targeted filter (grep/awk) if it matters.]",
    );
  });

  it("stays silent when raw_bytes is within the pre-dedup budget", () => {
    const raw = MAX_RAW_HEAD_BYTES + MAX_RAW_TAIL_BYTES;
    const footer = `[exit=0 cwd=/app timed_out=false output_truncated=true raw_bytes=${raw}]`;
    expect(annotate("seq 1000000 | head -50", `${rows(10)}${footer}`)).toBeNull();
  });

  // GAIA's bash results carry no `[exit=` footer, so there is nothing to sit above.
  it("appends the note as the last line when there is no footer", () => {
    const out = annotate("seq 1000 | head -50", rows(50));
    expect(out).not.toBeNull();
    expect(out!.split("\n").at(-1)).toBe(HEAD_50_NOTE);
    expect(out!.split("\n").at(-2)).toBe(rows(50).trimEnd().split("\n").at(-1));
  });

  it("tells tail's story in the other direction", () => {
    const out = annotate("tail -50 log", `${rows(50)}${HARBOR_FOOTER}`);
    expect(out!.split("\n").at(-2)).toBe(
      "[partial view: this command ends in 'tail -50' and the output reaches 50 lines, " +
        "so the source may have earlier lines before the first line shown. Don't conclude " +
        "something is absent from this view — re-run with a larger limit or a targeted " +
        "filter (grep/awk) if it matters.]",
    );
  });

  describe("byte limits", () => {
    afterEach(() => {
      delete process.env[MIN_BYTES_ENV];
    });

    const big = "z".repeat(2048);

    it("annotates a body that reaches the byte limit", () => {
      const out = annotate("head -c 2048 f", `${big}\n${HARBOR_FOOTER}`);
      expect(out!.split("\n").at(-2)).toBe(
        "[partial view: this command ends in 'head -c 2048' and the output reaches 2048 " +
          "bytes, so the source may continue past what is shown.]",
      );
    });

    it("stays silent below the byte limit", () => {
      expect(annotate("head -c 2048 f", `${"z".repeat(1000)}\n${HARBOR_FOOTER}`)).toBeNull();
    });

    it("stays silent when the byte limit is below the configured floor", () => {
      process.env[MIN_BYTES_ENV] = "4096";
      expect(annotate("head -c 2048 f", `${big}\n${HARBOR_FOOTER}`)).toBeNull();
    });

    it("prefers the footer's raw_bytes over the capped body's own size", () => {
      const footer = "[exit=0 cwd=/app timed_out=false output_truncated=true raw_bytes=9000]";
      const out = annotate("head -c 4096 f", `short\n${footer}`);
      expect(out!.split("\n").at(-2)).toContain("the output reaches 4096 bytes");
    });

    it("tells tail's story in the other direction", () => {
      const out = annotate("tail -c 2048 f", `${big}\n${HARBOR_FOOTER}`);
      expect(out!.split("\n").at(-2)).toBe(
        "[partial view: this command ends in 'tail -c 2048' and the output reaches 2048 " +
          "bytes, so the source may have earlier content before what is shown.]",
      );
    });
  });
});

describe("partialViewNote", () => {
  // The footer must stay the last `[exit=` match for the harbor adapter's
  // exit-code parser and the transcript previewers.
  it("never contains an exit= substring", () => {
    for (const tool of ["head", "tail"] as const) {
      for (const unit of ["lines", "bytes"] as const) {
        for (const unknownCut of [false, true]) {
          expect(partialViewNote({ tool, unit, limit: 50 }, unknownCut)).not.toContain("exit=");
        }
      }
    }
  });

  // A byte limit has no line count to be missing, whatever the caller passes.
  it("never describes a byte limit in terms of lines", () => {
    for (const tool of ["head", "tail"] as const) {
      for (const unknownCut of [false, true]) {
        expect(
          partialViewNote({ tool, unit: "bytes", limit: 2048 }, unknownCut),
        ).not.toContain("line");
      }
    }
  });
});

// ── tool_result handler ─────────────────────────────────────────────────────

function getToolResultHandler() {
  let handler: ((event: any) => any) | undefined;
  const pi = {
    on(name: string, h: (event: any) => any) {
      if (name === "tool_result") handler = h;
    },
  };
  setupTruncatedView(pi as any);
  if (!handler) throw new Error("truncated-view did not register a tool_result handler");
  return handler;
}

function shellEvent(overrides: Record<string, unknown> = {}) {
  return {
    toolName: "ShellSession",
    isError: false,
    input: { command: EVIDENCE_COMMAND },
    content: [{ type: "text", text: `${rows(50)}${HARBOR_FOOTER}` }],
    ...overrides,
  };
}

describe("truncated-view tool_result handler", () => {
  it("replaces a qualifying ShellSession result with the annotated text", async () => {
    const result = await getToolResultHandler()(shellEvent());
    expect(result.content).toHaveLength(1);
    expect(result.content[0].text).toContain("partial view");
    expect(result.content[0].text.split("\n").at(-1)).toBe(HARBOR_FOOTER);
  });

  it("annotates GAIA's built-in bash tool too", async () => {
    const result = await getToolResultHandler()(
      shellEvent({ toolName: "bash", content: [{ type: "text", text: rows(50) }] }),
    );
    expect(result.content[0].text).toContain("partial view");
  });

  it.each([
    ["a non-shell tool", { toolName: "Read" }],
    // ShellStart returns a job-started acknowledgment, not command output.
    ["ShellStart", { toolName: "ShellStart" }],
    ["an error result", { isError: true }],
    ["an image block", { content: [{ type: "image", data: "…" }] }],
    ["empty content", { content: [] }],
    ["a command with no trailing truncator", { input: { command: "ls -la" } }],
  ])("returns nothing for %s", async (_label, overrides) => {
    expect(await getToolResultHandler()(shellEvent(overrides))).toBeUndefined();
  });

  it("returns nothing when the input carries no command", async () => {
    expect(await getToolResultHandler()(shellEvent({ input: {} }))).toBeUndefined();
  });
});
