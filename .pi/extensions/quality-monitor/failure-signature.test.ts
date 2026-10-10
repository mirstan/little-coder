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
    expect(readResult(SHELL_FAIL, false, 4)).toEqual({ failed: true, hasContent: true, failedBy: "exit" });
  });
  it("reads failure from the thrown-result flag", () => {
    expect(readResult(BASH_FAIL, true, 4)).toEqual({ failed: true, hasContent: true, failedBy: "exit" });
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
      failedBy: "exit",
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

// A psrn/psm keyword-argument typo repeated across six ShellSession calls.
// Five piped through `head`, so bash's default (non-pipefail) semantics
// report [exit=0] over the Python TypeError; only D ran unpiped and failed
// at the bash level.
const EXIT0 = "[exit=0 cwd=/app timed_out=false backend=harbor-env]";
const PSRN_TRACE = (n = 1) =>
  `Traceback (most recent call last):\n  File "<string>", line 5, in <module>\nTypeError: image_to_data() got an unexpected keyword argument 'psrn'`.repeat(n);
const CALL_A = `${PSRN_TRACE()}\ntry2\n${PSRN_TRACE()}\n${EXIT0}`;
const CALL_B = `${PSRN_TRACE()}\n${EXIT0}`;
const CALL_D = `bash: line 7: warning: here-document at line 1 delimited by end-of-file (wanted \`EOF')\nbash: -c: line 8: syntax error: unexpected end of file\n\n[exit=2 cwd=/app timed_out=false backend=harbor-env]`;
const CALL_E = `written\nTraceback (most recent call last):\n  File "/app/probe.py", line 4, in <module>\n    data = pytesseract.image_to_data(img, psrn=11)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nTypeError: image_to_data() got an unexpected keyword argument 'psrn'\n${EXIT0}`;
const CALL_F = `4:data = pytesseract.image_to_data(img, psm=11)\nTraceback (most recent call last):\n  File "/app/probe.py", line 4, in <module>\n    data = pytesseract.image_to_data(img, psm=11)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nTypeError: image_to_data() got an unexpected keyword argument 'psm'\n${EXIT0}`;

describe("readResult — exit-code masking (Fix 1)", () => {
  it("calls a piped Python TypeError failed even though the footer reports exit=0", () => {
    expect(readResult(CALL_A, false, 4).failed).toBe(true);
    expect(readResult(CALL_E, false, 4).failed).toBe(true);
    expect(readResult(CALL_F, false, 4).failed).toBe(true);
  });
  it("still reads D's genuine bash-level failure from its exit code, unaffected", () => {
    // D has no recognizable exception line -- it keeps failing via exit=2.
    expect(readResult(CALL_D, false, 4).failed).toBe(true);
  });

  it("marks a masked-exit-0 failure as content-only, and D's real exit code as exit-proven (Finding 1)", () => {
    expect(readResult(CALL_A, false, 4).failedBy).toBe("content");
    expect(readResult(CALL_D, false, 4).failedBy).toBe("exit");
  });

  it("recognizes a qualified/dotted exception name as a failure signature (Finding 5)", () => {
    const dotted =
      "reading /data/payload.json\n" +
      "json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)\n" +
      "[exit=0 cwd=/app timed_out=false backend=harbor-env]";
    // `\w*` before `Error:` cannot match the dots in a qualified type name,
    // so this used to be neither exit-proven (masked at 0) nor content-proven
    // -- exactly the blind spot this whole PR exists to close.
    expect(readResult(dotted, false, 4)).toMatchObject({ failed: true, failedBy: "content" });
    const httpError =
      "GET https://api.example.com/v1/items failed\n" +
      "requests.exceptions.HTTPError: 503 Server Error: Service Unavailable\n" +
      "[exit=0 cwd=/app timed_out=false backend=harbor-env]";
    expect(readResult(httpError, false, 4).failed).toBe(true);
  });
  it("does not call a benign 'errors' mention a failure", () => {
    const ok = "no errors found, all clear\n[exit=0 cwd=/app timed_out=false backend=harbor-env]";
    expect(readResult(ok, false, 4).failed).toBe(false);
    const ok2 = "0 errors, 2 warnings\n[exit=0 cwd=/app timed_out=false backend=harbor-env]";
    expect(readResult(ok2, false, 4).failed).toBe(false);
  });

  it("does not call an exit=0 result failed for a mid-body line-initial Error: or Warning: that is not the last line", () => {
    // The trailing line only: a mid-body "Error:"/"Warning:" status message
    // must not flip an otherwise-successful run to failed.
    const errNotLast =
      "Error: skipping bad row 7, continuing\nprocessed 41 of 42 rows\nwrote output.csv\n[exit=0 cwd=/app timed_out=false backend=harbor-env]";
    expect(readResult(errNotLast, false, 4).failed).toBe(false);
    const warnNotLast =
      "Warning: Permanently added 'github.com' (ED25519) to the list of known hosts.\nclone complete\nbuild finished successfully\n[exit=0 cwd=/app timed_out=false backend=harbor-env]";
    expect(readResult(warnNotLast, false, 4).failed).toBe(false);
  });

  it("does not call an exit=0 passing pytest run failed because of an exception line inside its captured-log section", () => {
    // A RuntimeError/ValueError line buried in captured-log output (not the
    // trailing line) was still enough to flip `failed` to true even though
    // the run passed and the real trailing line says so.
    const passingWithCapturedLog =
      "============================= test session starts ==============================\n" +
      "collected 3 items\n\n" +
      "test_foo.py::test_a PASSED\n" +
      "--- Captured log call ---\n" +
      "ValueError: bad input during warmup, retried and succeeded\n" +
      "test_foo.py::test_b PASSED\n" +
      "test_foo.py::test_c PASSED\n\n" +
      "============================== 3 passed in 0.12s ==============================\n" +
      "[exit=0 cwd=/app timed_out=false backend=harbor-env]";
    expect(readResult(passingWithCapturedLog, false, 4).failed).toBe(false);
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

  describe("exception-line fallback (Fix 2)", () => {
    it("continues the streak across A-B and, after D resets it, across E-F -- both pairs score below the 0.95 Jaccard bar (0.9310, 0.7381) and are not byte-identical, so only the exception-line fallback keeps them counted as the same mistake instead of resetting", () => {
      const t = new FailureSignatureTracker();
      expect(t.record(obs("call-a", CALL_A), 1)).toBeNull();
      // B continues A's streak (count 2) via the exception-line match --
      // Jaccard alone (0.9310) falls just short of the 0.95 bar, so without
      // Fix 2 this pair would already reset instead of combining.
      expect(t.record(obs("call-b", CALL_B), 2)).toBeNull();
      // C is a verbatim repeat of B's own command+text in the real log --
      // deduped by the tracker's own "identical input" rule, count stays 2.
      expect(t.record(obs("call-b", CALL_B), 3)).toBeNull();
      // D is a genuinely different failure (no exception line at all) and
      // correctly resets the streak.
      expect(t.record(obs("call-d", CALL_D), 4)).toBeNull();
      expect(t.record(obs("call-e", CALL_E), 5)).toBeNull();
      // F continues E's streak (count 2) the same way B continued A's --
      // this pair scores 0.7381 Jaccard, the one whole-tail similarity alone
      // cannot catch at this text length.
      expect(t.record(obs("call-f", CALL_F), 6)).toBeNull();
      // The real trial had only these 6 calls, so the streak never crosses
      // the default streak=3 firing bar within them -- extend the same
      // recurring mistake by one more attempt and it fires immediately,
      // proving the streak that got this far is the live, counting one:
      const detection = t.record(obs("call-g", CALL_F), 7);
      expect(detection).toMatchObject({ reason: "repeated_failure_signature", failed: true, count: 3 });
    });

    it("regression pin: the E-F pair (0.7381 Jaccard, nearest the 0.5 floor) still counts as a match under the conjunctive rule", () => {
      // The fix must not regress the case the PR was built for. E and F
      // score below the 0.95 whole-tail threshold but inside the [0.57,
      // 0.95) band the conjunctive rule admits at the default threshold
      // (excLineJaccardFloor(0.95) = 0.57) -- jaccard >=
      // excLineJaccardFloor(threshold) -- distinct from the KeyError test
      // above, whose ~0.368 score falls below the floor.
      const eShingles = signatureOf("shellsession", CALL_E, 40).shingles;
      const fShingles = signatureOf("shellsession", CALL_F, 40).shingles;
      const score = jaccard(eShingles, fShingles);
      expect(score).toBeGreaterThanOrEqual(0.5);
      expect(score).toBeLessThan(0.95);

      const t = new FailureSignatureTracker({ streak: 2 });
      expect(t.record(obs("call-e", CALL_E), 1)).toBeNull();
      expect(t.record(obs("call-f", CALL_F), 2)).toMatchObject({
        reason: "repeated_failure_signature",
        count: 2,
      });
    });

    it("does not match two different exception types", () => {
      const t = new FailureSignatureTracker();
      const valueErr = `Traceback (most recent call last):\n  File "x.py", line 1\nValueError: invalid literal for int() with base 10: 'abc'\n${EXIT0}`;
      const keyErr = `Traceback (most recent call last):\n  File "x.py", line 1\nKeyError: 'foo'\n${EXIT0}`;
      t.record(obs("a", valueErr), 1);
      t.record(obs("b", keyErr), 2);
      expect(t.record(obs("c", valueErr), 3)).toBeNull();
    });

    it("does not match two genuinely different TypeErrors", () => {
      const t = new FailureSignatureTracker();
      const noneType = `Traceback (most recent call last):\n  File "x.py", line 1\nTypeError: 'NoneType' object is not subscriptable\n${EXIT0}`;
      t.record(obs("a", CALL_E), 1);
      t.record(obs("b", noneType), 2);
      expect(t.record(obs("c", CALL_E), 3)).toBeNull();
    });

    it("does not let Fix 2 paper over D's genuinely different (exit-code) failure", () => {
      // D has no recognizable exception line, so the exception-line fallback
      // simply doesn't apply to it -- it can only match via hash/Jaccard,
      // same as before this fix.
      const t = new FailureSignatureTracker();
      t.record(obs("a", CALL_E), 1);
      t.record(obs("b", CALL_D), 2);
      expect(t.record(obs("c", CALL_E), 3)).toBeNull();
    });

    it("does not accumulate a streak across KeyErrors that share only the exception type, not the underlying cause", () => {
      // Two KeyErrors differing only in the quoted key are different bugs:
      // the keys are far apart (edit distance > 2) and the tails score
      // ~0.368 Jaccard, under the fallback floor.
      const t = new FailureSignatureTracker();
      const keyErr = (file: string, line: number, fn: string, expr: string, key: string) =>
        `Traceback (most recent call last):\n  File "${file}", line ${line}, in ${fn}\n    ${expr}\nKeyError: '${key}'\n${EXIT0}`;
      expect(
        t.record(obs("a", keyErr("handlers/users.py", 42, "load_profile", "profile = cache[record_id]", "user_id")), 1),
      ).toBeNull();
      expect(
        t.record(obs("b", keyErr("workers/ingest.py", 118, "flush_batch", "row = batch[cursor]", "timestamp")), 2),
      ).toBeNull();
      expect(
        t.record(obs("c", keyErr("api/routes.py", 7, "handle_request", "ctx = session[token]", "csrf_token")), 3),
      ).toBeNull();
      expect(
        t.record(obs("d", keyErr("jobs/reindex.py", 265, "rebuild", "doc = index[doc_id]", "shard_id")), 4),
      ).toBeNull();
    });

    it("does not accumulate a streak across FileNotFoundErrors on two demonstrably different paths (Findings 2+3)", () => {
      const t = new FailureSignatureTracker();
      const notFound = (path: string) =>
        `Traceback (most recent call last):\n  File "probe.py", line 9, in <module>\nFileNotFoundError: [Errno 2] No such file or directory: '${path}'\n${EXIT0}`;
      expect(t.record(obs("a", notFound("/data/alpha_2023_run_input.csv")), 1)).toBeNull();
      expect(t.record(obs("b", notFound("/var/lib/reports/beta_final_output.parquet")), 2)).toBeNull();
      expect(t.record(obs("c", notFound("/srv/cache/gamma_manifest_shard.json")), 3)).toBeNull();
    });

    it("does not accumulate a streak across AttributeErrors on two unrelated modules (Findings 2+3)", () => {
      const t = new FailureSignatureTracker();
      const attrErr = (mod: string, attr: string) =>
        `Traceback (most recent call last):\n  File "probe.py", line 3, in <module>\nAttributeError: module '${mod}' has no attribute '${attr}'\n${EXIT0}`;
      expect(t.record(obs("a", attrErr("numpy", "nanstd")), 1)).toBeNull();
      expect(t.record(obs("b", attrErr("pandas", "read_parquet")), 2)).toBeNull();
      expect(t.record(obs("c", attrErr("requests", "adapters")), 3)).toBeNull();
    });

    it("does not let an incidental mid-tail Warning: line supply a false excLine match between unrelated failures (Finding 4)", () => {
      // excLine comes from the trailing line only, so an incidental mid-tail
      // "Warning:" supplies none.
      const attempt = (job: number) =>
        [
          "Warning: cache miss, rebuilding index",
          ...Array.from({ length: 20 }, (_, i) => `processing job ${job * 1000 + i} of 999`),
          "unexpected worker termination",
          "[exit=1 cwd=/app timed_out=false backend=subprocess]",
        ].join("\n");
      expect(signatureOf("shellsession", attempt(1), 40).excLine).toBeUndefined();
      expect(signatureOf("shellsession", attempt(9), 40).excLine).toBeUndefined();
      const t = new FailureSignatureTracker();
      expect(t.record(obs("job-1", attempt(1)), 1)).toBeNull();
      expect(t.record(obs("job-9", attempt(9)), 2)).toBeNull();
      expect(t.record(obs("job-42", attempt(42)), 3)).toBeNull();
    });

    it("lets a qualified/dotted exception name participate in signature matching (Finding 5)", () => {
      const t = new FailureSignatureTracker();
      const dotted = (path: string) =>
        `reading ${path}\njson.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)\n${EXIT0}`;
      expect(t.record(obs("a", dotted("/data/a.json")), 1)).toBeNull();
      expect(t.record(obs("b", dotted("/data/b.json")), 2)).toBeNull();
      expect(t.record(obs("c", dotted("/data/c.json")), 3)).toMatchObject({
        reason: "repeated_failure_signature",
        count: 3,
      });
    });

    it("does not let a masked-exit-0 content-only mismatch clobber a live exit-code-proven streak (Finding 1)", () => {
      const t = new FailureSignatureTracker();
      expect(t.record(obs("./run --a", SHELL_FAIL), 1)).toBeNull();
      expect(t.record(obs("./run --b", SHELL_FAIL), 2)).toBeNull();
      // Turn 3: the model reruns its own failing command as
      // `... 2>&1 | tail -1` -- an ordinary debugging move. Still exit 0
      // (piped), and its one-line tail happens to read as an unrelated
      // exception, so readResult calls it failed via content alone; its
      // hash/Jaccard score against the tracked SHELL_FAIL streak is nowhere
      // near a match.
      const rerunTail =
        "TypeError: unrelated_probe() got an unexpected keyword argument 'zzz'\n" +
        "[exit=0 cwd=/app timed_out=false backend=subprocess]";
      expect(readResult(rerunTail, false, 4)).toMatchObject({ failed: true, failedBy: "content" });
      expect(t.record(obs("./run --a 2>&1 | tail -1", rerunTail), 3)).toBeNull();
      // Turn 4 resumes the ORIGINAL exit-code-proven failure. Pre-fix, turn
      // 3's mismatch would have forget()-ed the tracked entry and restarted
      // it at count 1; count is 3, proving the streak survived intact.
      expect(t.record(obs("./run --c", SHELL_FAIL), 4)).toMatchObject({
        reason: "repeated_failure_signature",
        count: 3,
      });
    });
  });
});
