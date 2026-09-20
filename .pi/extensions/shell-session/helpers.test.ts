import { describe, it, expect, afterAll } from "vitest";
import { readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname } from "node:path";
import {
  stripAnsi,
  dedupLines,
  truncateLines,
  formatOutput,
  capBytesHeadTail,
  byteLen,
  MAX_LINES,
  SMALL_OUTPUT_FLOOR_BYTES,
} from "./helpers.ts";

describe("stripAnsi", () => {
  it("removes SGR sequences", () => {
    expect(stripAnsi("\x1b[31mred\x1b[0m text")).toBe("red text");
  });
  it("passes through ansi-free text", () => {
    expect(stripAnsi("hello")).toBe("hello");
  });
});

describe("dedupLines", () => {
  it("collapses consecutive duplicates", () => {
    const out = dedupLines(["a", "b", "b", "b", "c"]);
    expect(out).toEqual(["a", "b", "  [... 2 duplicate line(s) collapsed ...]", "c"]);
  });
  it("handles trailing duplicates", () => {
    const out = dedupLines(["a", "a", "a"]);
    expect(out).toEqual(["a", "  [... 2 duplicate line(s) collapsed ...]"]);
  });
});

describe("truncateLines", () => {
  it("passes through short output", () => {
    const { lines, truncated } = truncateLines(["a", "b"], 10);
    expect(lines).toEqual(["a", "b"]);
    expect(truncated).toBe(false);
  });
  it("truncates long output with head/tail", () => {
    const input = Array.from({ length: 20 }, (_, i) => `line${i}`);
    const { lines, truncated } = truncateLines(input, 8);
    expect(truncated).toBe(true);
    // cap=8 → head=4, tail=2, skipped=14
    expect(lines.length).toBe(4 + 1 + 2);
    expect(lines[0]).toBe("line0");
    expect(lines[4]).toContain("lines truncated");
  });
});

describe("formatOutput", () => {
  it("formats basic exit=0 output", () => {
    const out = formatOutput("hello\nworld\n", 0, "/tmp", false, "backend=subprocess");
    expect(out).toContain("hello");
    expect(out).toContain("world");
    expect(out).toContain("[exit=0 cwd=/tmp timed_out=false backend=subprocess]");
  });
  it("emits footer only for empty body", () => {
    const out = formatOutput("", 0, "/tmp", false, "");
    expect(out.startsWith("[")).toBe(true);
  });
  it("appends output_truncated when head/tail cut", () => {
    // Padded past the 4KB small-output floor so the line cap still fires.
    const big = Array.from({ length: 500 }, (_, i) => `line${i}`.padEnd(20, "x")).join("\n");
    const out = formatOutput(big, 0, "/", false, "");
    expect(out).toContain("output_truncated=true");
  });
  it("strips ANSI before line processing", () => {
    const out = formatOutput("\x1b[32mgreen\x1b[0m", 0, "/", false, "");
    expect(out).toContain("green");
    expect(out).not.toContain("\x1b");
  });
});

describe("capBytesHeadTail", () => {
  it("passes through content within budget", () => {
    const { text, dropped } = capBytesHeadTail("hello", 10, 10);
    expect(text).toBe("hello");
    expect(dropped).toBe(0);
  });

  it("cuts at the LAST newline in the head window", () => {
    // A first-newline rule would cut after "a" and throw away the budget.
    const { text } = capBytesHeadTail("a\nbb\n" + "c".repeat(40), 16, 8);
    expect(text.startsWith("a\nbb\n  [... ")).toBe(true);
  });

  it("cuts at the FIRST newline in the tail window", () => {
    // A last-newline rule would keep only "ee".
    const { text } = capBytesHeadTail("x".repeat(40) + "\ndd\nee", 8, 12);
    expect(text.endsWith(" truncated ...]\ndd\nee")).toBe(true);
  });

  // Shared parity vectors: benchmarks/test_format_output.py asserts these
  // exact strings against both Python _cap_bytes_head_tail implementations.
  // There is no shared code across the three, only this spec.
  it("cuts 2-byte characters on a character boundary (parity vector)", () => {
    const { text } = capBytesHeadTail("é".repeat(30), 11, 11);
    expect(text).toBe("é".repeat(5) + "\n  [... 40B truncated ...]\n" + "é".repeat(5));
  });

  it("cuts 3-byte characters on a character boundary (parity vector)", () => {
    const { text } = capBytesHeadTail("日".repeat(20), 11, 11);
    expect(text).toBe("日".repeat(3) + "\n  [... 42B truncated ...]\n" + "日".repeat(3));
  });
});

describe("formatOutput byte cap", () => {
  const GIANT_LEN = 1042304; // the real vocab.json grep result that crashed a trial

  // Cleanup runs even when an assertion above it throws, which an unlinkSync
  // at the end of a test body does not -- leaked overflow files accumulate
  // across CI retries.
  const overflowDirs = new Set<string>();
  const overflowPath = (out: string, prefix: string): string => {
    const lines = out.split("\n");
    const noteLine = lines[lines.length - 2];
    expect(noteLine.startsWith(prefix)).toBe(true);
    const path = noteLine.slice(prefix.length);
    overflowDirs.add(dirname(path));
    return path;
  };
  afterAll(() => {
    for (const dir of overflowDirs) rmSync(dir, { recursive: true, force: true });
  });

  it("caps a single giant line with no newlines anywhere", () => {
    const line = "x".repeat(GIANT_LEN);
    const out = formatOutput(line, 0, "/tmp", false, "backend=subprocess");

    expect(byteLen(out)).toBeLessThan(64 * 1024);
    expect(out).toContain("truncated ...]");
    expect(out).toContain("output_truncated=true");
    expect(out).toContain(`raw_bytes=${GIANT_LEN}`);

    // The guard against "simplifying" this to pi's truncateHead, which
    // returns EMPTY content for exactly this input.
    const head = out.split("\n")[0];
    expect(head.length).toBe(32 * 1024);
    expect(line.startsWith(head)).toBe(true);
  });

  it("caps many short lines by line count, staying under the byte cap", () => {
    const big = Array.from({ length: 300000 }, (_, i) => String(i).padStart(39, "0")).join("\n");
    const out = formatOutput(big, 0, "/tmp", false, "");

    expect(out).toContain("lines truncated");
    expect(out).toContain("output_truncated=true");
    expect(byteLen(out)).toBeLessThanOrEqual(48 * 1024);
    expect(out.split("\n").length).toBeLessThanOrEqual(MAX_LINES);
  });

  it("skips the line cap for small-but-many-lines output (below the 4KB floor)", () => {
    // 300 lines, ~1.5KB total -- e.g. `ls -1` on a big sparse directory.
    const lines = Array.from({ length: 300 }, (_, i) => String(i).padStart(4, "0"));
    const text = lines.join("\n");
    expect(byteLen(text)).toBeLessThan(SMALL_OUTPUT_FLOOR_BYTES);

    const out = formatOutput(text, 0, "/tmp", false, "");
    expect(out).not.toContain("lines truncated");
    expect(out).not.toContain("output_truncated=true");
    for (const line of lines) expect(out).toContain(line);
  });

  it("still applies the line cap once padded past the 4KB floor", () => {
    // Same 300 lines, padded to ~6KB total -- above the floor.
    const text = Array.from({ length: 300 }, (_, i) => String(i).padStart(4, "0") + "x".repeat(15)).join(
      "\n",
    );
    expect(byteLen(text)).toBeGreaterThan(SMALL_OUTPUT_FLOOR_BYTES);

    const out = formatOutput(text, 0, "/tmp", false, "");
    expect(out).toContain("lines truncated");
    expect(out).toContain("output_truncated=true");
  });

  it("still lets dedup rescue a flood of identical lines", () => {
    const out = formatOutput("same\n".repeat(100000), 0, "/tmp", false, "");
    expect(out).toContain("duplicate line(s) collapsed");
    expect(byteLen(out)).toBeLessThan(1024);
  });

  it("never cuts a multi-byte line mid-character", () => {
    const line = "é".repeat(600000);
    const out = formatOutput(line, 0, "/tmp", false, "");
    expect(out).not.toContain("�");
    expect(line.startsWith(out.split("\n")[0])).toBe(true);
  });

  it("leaves normal output byte-for-byte unchanged", () => {
    expect(formatOutput("hello\nworld\n", 0, "/tmp", false, "backend=subprocess")).toBe(
      "hello\nworld\n\n[exit=0 cwd=/tmp timed_out=false backend=subprocess]",
    );
  });

  it("writes an overflow file when the local backend asks for one", () => {
    const line = "y".repeat(GIANT_LEN);
    const out = formatOutput(line, 0, "/tmp", false, "backend=subprocess", { overflowFile: true });

    const path = overflowPath(out, "Full output: ");
    expect(readFileSync(path, "utf-8")).toBe(line);
  });

  it("keeps overflow files private to this process", () => {
    // The content is arbitrary command output, so on a shared /tmp neither the
    // file nor the listing that reveals its name may be world-readable.
    const out = formatOutput("y".repeat(GIANT_LEN), 0, "/tmp", false, "backend=subprocess", {
      overflowFile: true,
    });
    const path = overflowPath(out, "Full output: ");
    const dir = dirname(path);

    expect(statSync(dir).mode & 0o777).toBe(0o700);
    expect(statSync(path).mode & 0o777).toBe(0o600);
    expect(dir).not.toBe(tmpdir());
    expect(dirname(dir)).toBe(tmpdir());
  });

  it("never writes a file when overflowFile is unset", () => {
    // Narrower than it looks: this is formatOutput's own default, not proof of
    // anything backend-specific. That the tmux/harbor proxies never pass
    // overflowFile is the caller's invariant, pinned in index.test.ts.
    const out = formatOutput("z".repeat(GIANT_LEN), 0, "/app", false, "backend=tmux-proxy");
    expect(out).not.toContain("Full output:");
  });
});
