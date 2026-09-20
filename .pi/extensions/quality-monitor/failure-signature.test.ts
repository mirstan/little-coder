import { describe, it, expect, afterEach, beforeEach } from "vitest";
import {
  boundResultText,
  FAILSIG_ENV,
  FailureSignatureTracker,
  normalizeTail,
  readResult,
  signatureOf,
} from "./failure-signature.ts";
import { jaccard } from "./similarity.ts";
import { pinEnv } from "../_shared/env-pin.ts";

// The assertions below are written against the built-in defaults (streak 3,
// threshold 0.95, window 8, tail 40 lines), which failsigOptionsFromEnv reads
// out of the environment. These knobs are advertised for CI and local tuning,
// so a shell that sets one would otherwise retune the watchdog under the
// tests instead of being pinned out of them.
const pinned = pinEnv(Object.values(FAILSIG_ENV));
beforeEach(() => pinned.clear());
afterEach(() => pinned.restore());

// What ShellSession actually returns: the failure is in the footer, and the
// result is never flagged as an error.
const SHELL_FOOTER = "[exit=139 cwd=/app timed_out=false backend=subprocess]";
const SHELL_FAIL = `reading corpus from /data\nSegmentation fault (core dumped)\n${SHELL_FOOTER}`;
// What pi's built-in bash returns: it throws, and the status line is text.
const BASH_FAIL = "reading corpus from /data\nSegmentation fault (core dumped)\nCommand exited with code 139";

// Frames carry their arguments and source path, as a real backtrace does.
// That is also what gives the fallback room: the two varying elements (the
// timestamp and the heap address) are a smaller share of a denser tail, so
// the pair scores ~0.977 against the 0.95 threshold rather than ~0.960. At
// the tighter spacing a reworded frame line flipped the test.
function trace(addr: string, stamp: string): string {
  const lines = [`[${stamp}] compressor: starting run`];
  for (let i = 0; i < 36; i++)
    lines.push(
      `  #${i} 0x00005612aa${(i * 7).toString(16).padStart(4, "0")} in stage_${i} ` +
        `(ctx=ctx_${i}, flags=flags_${i}) at src/ring.c:${i * 3} discriminator ${i % 4}`,
    );
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
  it("does not let the status line pay for output that is missing", () => {
    // Three real tokens plus pi's five-token status line cleared a floor that
    // merely made an allowance for the line instead of removing it.
    expect(readResult("no such file\n\nCommand exited with code 1", true, 4).hasContent).toBe(false);
  });
  it("accepts a short failure that says something, whatever the status line costs", () => {
    // The mirror case: "Command aborted" is two tokens, so a fixed allowance
    // over-charged this one and dropped a failure that does identify itself.
    expect(readResult("cannot bind port 8080\n\nCommand aborted", true, 4).hasContent).toBe(true);
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

describe("boundResultText", () => {
  const huge = (footer: string) =>
    `${Array.from({ length: 4000 }, (_, i) => `line ${i} of a very verbose build log`).join("\n")}\n${footer}`;
  it("leaves an ordinary result untouched", () => {
    expect(boundResultText(SHELL_FAIL)).toBe(SHELL_FAIL);
  });
  it("bounds an oversized result and keeps its end", () => {
    const text = huge(SHELL_FOOTER);
    expect(text.length).toBeGreaterThan(64 * 1024);
    const bounded = boundResultText(text);
    expect(bounded.length).toBeLessThanOrEqual(64 * 1024);
    expect(bounded.endsWith(SHELL_FOOTER)).toBe(true);
  });
  it("starts what it keeps at a line boundary", () => {
    expect(boundResultText(huge(SHELL_FOOTER)).split("\n")[0]).toMatch(/^line \d+ of a very verbose build log$/);
  });
  it("leaves the watchdog's reading of an oversized failure unchanged", () => {
    const text = huge(SHELL_FOOTER);
    expect(readResult(boundResultText(text), false, 4)).toEqual(readResult(text, false, 4));
    expect(signatureOf("shellsession", boundResultText(text), 40).hash).toBe(
      signatureOf("shellsession", text, 40).hash,
    );
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
    // Exercising the Jaccard fallback, not hash equality, and doing so with
    // room to spare on both counts.
    const shingles = (text: string) => signatureOf("shellsession", text, 40).shingles;
    expect(signatureOf("shellsession", trace("0x7f8a1c00", "10:00:01"), 40).hash).not.toBe(
      signatureOf("shellsession", trace("0x7f8a2d40", "10:04:55"), 40).hash,
    );
    expect(
      jaccard(shingles(trace("0x7f8a1c00", "10:00:01")), shingles(trace("0x7f8a2d40", "10:04:55"))),
    ).toBeGreaterThan(0.97);
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

  it("spans `window` turns inclusive, as the fuzzy tracker's window does", () => {
    // Turns 1 and 8 are eight turns of history; turns 1 and 9 are nine, and
    // the fuzzy tracker has already dropped turn 1 by then. Counting them
    // together here would let one detector corroborate a span the other no
    // longer holds.
    const inside = new FailureSignatureTracker({ window: 8, streak: 2 });
    inside.record(obs("./run --a", SHELL_FAIL), 1);
    expect(inside.record(obs("./run --b", SHELL_FAIL), 8)).toMatchObject({ count: 2 });
    const outside = new FailureSignatureTracker({ window: 8, streak: 2 });
    outside.record(obs("./run --a", SHELL_FAIL), 1);
    expect(outside.record(obs("./run --b", SHELL_FAIL), 9)).toBeNull();
  });

  // Enough other tools to overflow MAX_TRACKED (16) and force one eviction.
  const fillOtherTools = (t: FailureSignatureTracker, from: number, count: number, turn: number) => {
    for (let i = 0; i < count; i++) {
      t.record(
        { toolName: `Tool${from + i}`, input: { command: "./x" }, text: SHELL_FAIL, isError: false },
        turn + i,
      );
    }
  };

  it("keeps a still-counting streak through an eviction", () => {
    // Eviction drops the front of the map, so an entry that keeps matching
    // has to be moved to the back as it counts -- otherwise the tool that
    // started the session is the first one dropped, however live its streak.
    const t = new FailureSignatureTracker({ window: 50 });
    t.record(obs("./run --a", SHELL_FAIL), 1);
    fillOtherTools(t, 1, 8, 2);
    t.record(obs("./run --b", SHELL_FAIL), 10);
    fillOtherTools(t, 9, 8, 11);
    expect(t.record(obs("./run --c", SHELL_FAIL), 19)).toMatchObject({ count: 3 });
  });

  it("forgets an evicted signature's spent messages along with it", () => {
    // The message budget is keyed by signature, so a record outliving its
    // entry both grows without bound and silences the same failure if it
    // comes back.
    const t = new FailureSignatureTracker({ window: 50 });
    for (const n of [1, 2]) t.record(obs(`./run --${n}`, SHELL_FAIL), n);
    const first = t.record(obs("./run --3", SHELL_FAIL), 3);
    expect(first).toMatchObject({ count: 3 });
    t.markNotified(first!.sigKey);
    fillOtherTools(t, 1, 16, 4);
    for (const n of [4, 5] as const) t.record(obs(`./run --${n}`, SHELL_FAIL), 20 + n);
    expect(t.record(obs("./run --6", SHELL_FAIL), 26)).toMatchObject({ count: 3 });
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

  it("reports a corroborated passing streak as output, not failure", () => {
    // Tracked deliberately (an attempt that "succeeds" with the same
    // unsatisfying output is not progress), but it is not a failure, and the
    // detection has to say which it is.
    const t = new FailureSignatureTracker();
    const passing = "compressed size 512 bytes, expected under 400\n[exit=0 cwd=/app timed_out=false backend=subprocess]";
    const c = (command: string) => ({ ...obs(command, passing), clusterId: 4, corroborated: true });
    expect(t.record(c("./run --a"), 1)).toBeNull();
    expect(t.record(c("./run --b"), 2)).toMatchObject({
      reason: "repeated_output_signature",
      failed: false,
      count: 2,
    });
  });

  it("reports a failure streak as a failure", () => {
    const t = new FailureSignatureTracker();
    for (const n of [1, 2]) t.record(obs(`./run --${n}`, SHELL_FAIL), n);
    expect(t.record(obs("./run --3", SHELL_FAIL), 3)).toMatchObject({
      reason: "repeated_failure_signature",
      failed: true,
    });
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
