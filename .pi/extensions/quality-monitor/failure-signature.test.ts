import { describe, it, expect } from "vitest";
import {
  FailureSignatureTracker,
  normalizeTail,
  readResult,
  signatureOf,
} from "./failure-signature.ts";

// What ShellSession actually returns: the failure is in the footer, and the
// result is never flagged as an error.
const SHELL_FOOTER = "[exit=139 cwd=/app timed_out=false backend=subprocess]";
const SHELL_FAIL = `reading corpus from /data\nSegmentation fault (core dumped)\n${SHELL_FOOTER}`;
// What pi's built-in bash returns: it throws, and the status line is text.
const BASH_FAIL = "reading corpus from /data\nSegmentation fault (core dumped)\nCommand exited with code 139";

function trace(addr: string, stamp: string): string {
  const lines = [`[${stamp}] compressor: starting run`];
  for (let i = 0; i < 36; i++) lines.push(`  #${i} 0x00005612aa${(i * 7).toString(16).padStart(4, "0")} in stage_${i} () at ring.c:${i * 3}`);
  lines.push(`  malloc_error_break: heap block at ${addr} was modified after being freed`);
  lines.push("Abort trap: 6");
  lines.push("[exit=134 cwd=/app timed_out=false backend=subprocess]");
  return lines.join("\n");
}

describe("readResult", () => {
  it("reads failure from the footer when nothing threw", () => {
    // The Terminal-Bench shell path: gating on isError alone leaves this
    // watchdog silent for every TB failure.
    expect(readResult(SHELL_FAIL, false, 4)).toEqual({ failed: true, hasContent: true });
  });
  it("reads failure from the thrown-result flag", () => {
    expect(readResult(BASH_FAIL, true, 4)).toEqual({ failed: true, hasContent: true });
  });
  it("does not call a zero exit a failure", () => {
    const ok = "all 42 tests passed in 3 seconds\n[exit=0 cwd=/app timed_out=false backend=subprocess]";
    expect(readResult(ok, false, 4).failed).toBe(false);
  });
  it("does not call a footerless success a failure", () => {
    expect(readResult("all 42 tests passed", false, 4).failed).toBe(false);
  });
  it("rejects a silent nonzero exit as contentless", () => {
    // Three unrelated no-match greps all produce exactly this.
    expect(readResult("[exit=1 cwd=/app timed_out=false backend=subprocess]", false, 4)).toEqual({
      failed: true,
      hasContent: false,
    });
  });
  it("rejects a bare status line as contentless", () => {
    expect(readResult("Command exited with code 1", true, 4).hasContent).toBe(false);
    expect(readResult("Command timed out after 30 seconds", true, 4).hasContent).toBe(false);
  });
  it("accepts a two-line failure that says what went wrong", () => {
    expect(readResult(`Segmentation fault (core dumped)\n${SHELL_FOOTER}`, false, 4).hasContent).toBe(true);
  });
});

describe("normalizeTail", () => {
  it("is idempotent", () => {
    const once = normalizeTail("a  \n\n\n b\t\n", 40);
    expect(normalizeTail(once, 40)).toBe(once);
  });
  it("ignores trailing whitespace and repeated blank lines", () => {
    expect(normalizeTail("a   \n\n\nb", 40)).toBe(normalizeTail("a\n\nb   ", 40));
  });
  it("keeps only the last maxLines lines", () => {
    const text = Array.from({ length: 50 }, (_, i) => `line ${i}`).join("\n");
    expect(normalizeTail(text, 40).split("\n")).toHaveLength(40);
    expect(normalizeTail(text, 40).startsWith("line 10")).toBe(true);
  });
  it("gives a 39-line and a 41-line run with the same tail the same signature", () => {
    const tail = Array.from({ length: 39 }, (_, i) => `tail ${i}`);
    const short = tail.join("\n");
    const long = ["noise a", "noise b", ...tail].join("\n");
    expect(normalizeTail(short, 39)).toBe(normalizeTail(long, 39));
  });
});

describe("FailureSignatureTracker", () => {
  const obs = (command: string, text: string, isError = false) => ({
    toolName: "ShellSession",
    input: { command },
    text,
    isError,
  });

  it("fires after `streak` differing attempts with the identical failure", () => {
    const t = new FailureSignatureTracker();
    expect(t.record(obs("./run --a", SHELL_FAIL), 1)).toBeNull();
    expect(t.record(obs("./run --b", SHELL_FAIL), 2)).toBeNull();
    expect(t.record(obs("./run --c", SHELL_FAIL), 3)).toMatchObject({
      reason: "repeated_failure_signature",
      count: 3,
      toolName: "ShellSession",
    });
  });

  it("matches a noisy trace whose addresses and timestamps move", () => {
    const t = new FailureSignatureTracker();
    t.record(obs("./run --a", trace("0x7f8a1c00", "10:00:01")), 1);
    t.record(obs("./run --b", trace("0x7f8a2d40", "10:04:55")), 2);
    expect(t.record(obs("./run --c", trace("0x7f8a3e80", "10:09:12")), 3)).toMatchObject({ count: 3 });
  });

  it("absorbs the per-call overflow path ShellSession embeds in the text", () => {
    // Level 1 cannot survive a temp path that differs every call; level 2 sees
    // one changed line in forty.
    const withOverflow = (n: number) => trace("0x7f8a1c00", "10:00:01").replace("Abort trap: 6", `Full output: /tmp/shell-${n}.log\nAbort trap: 6`);
    const t = new FailureSignatureTracker();
    expect(signatureOf("shellsession", withOverflow(1), 40).hash).not.toBe(
      signatureOf("shellsession", withOverflow(2), 40).hash,
    );
    t.record(obs("./run --a", withOverflow(1)), 1);
    t.record(obs("./run --b", withOverflow(2)), 2);
    expect(t.record(obs("./run --c", withOverflow(3)), 3)).toMatchObject({ count: 3 });
  });

  it("does not match a genuinely different error", () => {
    const t = new FailureSignatureTracker();
    t.record(obs("./run --a", SHELL_FAIL), 1);
    t.record(obs("./run --b", SHELL_FAIL), 2);
    const other = `./run: cannot open /data/corpus.bin: No such file or directory\n${SHELL_FOOTER}`;
    expect(t.record(obs("./run --c", other), 3)).toBeNull();
    // ...and the streak restarts from the new outcome.
    expect(t.record(obs("./run --d", other), 4)).toBeNull();
    expect(t.record(obs("./run --e", other), 5)).toMatchObject({ count: 3 });
  });

  it("ignores a repeat of the identical input", () => {
    const t = new FailureSignatureTracker();
    t.record(obs("./run --a", SHELL_FAIL), 1);
    t.record(obs("./run --a", SHELL_FAIL), 2);
    expect(t.record(obs("./run --a", SHELL_FAIL), 3)).toBeNull();
  });

  it("restarts a streak whose last match aged out of the window", () => {
    const t = new FailureSignatureTracker({ window: 8 });
    t.record(obs("./run --a", SHELL_FAIL), 1);
    t.record(obs("./run --b", SHELL_FAIL), 2);
    expect(t.record(obs("./run --c", SHELL_FAIL), 30)).toBeNull();
  });

  it("ignores failures with nothing to identify them by", () => {
    const t = new FailureSignatureTracker();
    const silent = "[exit=1 cwd=/app timed_out=false backend=subprocess]";
    expect(t.record(obs("grep alpha src", silent), 1)).toBeNull();
    expect(t.record(obs("grep beta src", silent), 2)).toBeNull();
    expect(t.record(obs("grep gamma src", silent), 3)).toBeNull();
  });

  it("ignores a passing result that no cluster corroborates", () => {
    const t = new FailureSignatureTracker();
    const passing = "all 42 tests passed in 3 seconds\n[exit=0 cwd=/app timed_out=false backend=subprocess]";
    expect(t.record(obs("make test", passing), 1)).toBeNull();
    expect(t.record(obs("make check", passing), 2)).toBeNull();
    expect(t.record(obs("make test-all", passing), 3)).toBeNull();
  });

  it("fires one attempt sooner when a fuzzy cluster corroborates", () => {
    const t = new FailureSignatureTracker();
    const c = (command: string) => ({ ...obs(command, SHELL_FAIL), clusterId: 7, corroborated: true });
    expect(t.record(c("./run --a"), 1)).toBeNull();
    expect(t.record(c("./run --b"), 2)).toMatchObject({ count: 2, corroborated: true, clusterId: 7 });
  });

  it("keeps a corroborated passing outcome off the error slot", () => {
    const t = new FailureSignatureTracker();
    const passing = "compressed size 512 bytes, expected under 400\n[exit=0 cwd=/app timed_out=false backend=subprocess]";
    t.record(obs("./run --a", SHELL_FAIL), 1);
    t.record({ ...obs("./run --b", passing), clusterId: 3, corroborated: true }, 2);
    t.record(obs("./run --c", SHELL_FAIL), 3);
    // The live error signature kept counting through the passing result.
    expect(t.record(obs("./run --d", SHELL_FAIL), 4)).toMatchObject({ count: 3 });
  });

  it("is disabled by a streak of 0", () => {
    const t = new FailureSignatureTracker({ streak: 0 });
    t.record(obs("./run --a", SHELL_FAIL), 1);
    t.record(obs("./run --b", SHELL_FAIL), 2);
    expect(t.record(obs("./run --c", SHELL_FAIL), 3)).toBeNull();
  });

  it("speaks once, then once more at streak + 2, then not again", () => {
    const t = new FailureSignatureTracker();
    const fire = (turn: number) => {
      const d = t.record(obs(`./run --${turn}`, SHELL_FAIL), turn);
      if (d) t.markNotified(d.sigKey);
      return d;
    };
    expect(fire(1)).toBeNull();
    expect(fire(2)).toBeNull();
    expect(fire(3)).toMatchObject({ count: 3, escalated: false });
    expect(fire(4)).toBeNull();
    expect(fire(5)).toMatchObject({ count: 5, escalated: true });
    expect(fire(6)).toBeNull();
    expect(fire(7)).toBeNull();
  });

  it("tracks each tool's outcome separately", () => {
    const t = new FailureSignatureTracker();
    t.record(obs("./run --a", SHELL_FAIL), 1);
    t.record({ toolName: "Bash", input: { command: "./run --b" }, text: BASH_FAIL, isError: true }, 2);
    expect(t.record(obs("./run --c", SHELL_FAIL), 3)).toBeNull();
  });
});
