// Watchdog for the inverse of a near-duplicate loop: the attempts differ, the
// outcome does not. A model rewriting its code three times and getting the
// same crash each time is making no progress, and nothing else in
// quality-monitor looks at tool output.
//
// The signature is a hash of a normalized output tail with a similarity
// fallback, not a scrubber that strips timestamps/PIDs/addresses: every scrub
// pattern encodes an assumption about noise FORMAT and silently stops
// matching when the tooling under test changes. Noise is tolerated
// proportionally instead — one varying line in forty barely moves Jaccard,
// while a genuinely different error falls far below the threshold.

import { createHash } from "node:crypto";
import { envNumber } from "../_shared/env-number.ts";
import { splitFooter } from "../truncated-view/truncation.ts";
import { footerExit } from "../syntax-check/helpers.ts";
import { bigrams, jaccard, stableStringify, tokenize } from "./similarity.ts";

export interface FailsigOptions {
  tailLines: number;
  threshold: number;
  streak: number;
  window: number;
  minTokens: number;
}

// Named here, beside the only code that reads them -- see similarity.ts's
// FUZZY_ENV.
export const FAILSIG_ENV = {
  tailLines: "LITTLE_CODER_FAILSIG_TAIL_LINES",
  threshold: "LITTLE_CODER_FAILSIG_THRESHOLD",
  streak: "LITTLE_CODER_FAILSIG_STREAK",
  window: "LITTLE_CODER_FAILSIG_WINDOW",
  minTokens: "LITTLE_CODER_FAILSIG_MIN_TOKENS",
} as const;

export function failsigOptionsFromEnv(overrides: Partial<FailsigOptions> = {}): FailsigOptions {
  return {
    tailLines: envNumber(FAILSIG_ENV.tailLines, 40),
    // Tighter than the fuzzy detector's 0.85: "the identical outcome" is a
    // stronger claim than "a similar attempt".
    threshold: envNumber(FAILSIG_ENV.threshold, 0.95),
    streak: envNumber(FAILSIG_ENV.streak, 3),
    window: envNumber(FAILSIG_ENV.window, 8),
    minTokens: envNumber(FAILSIG_ENV.minTokens, 4),
    ...overrides,
  };
}

// pi's own bash ends a failed result with one of these as its last line
// (core/tools/bash.ts appendStatus, verified against the installed
// package). It is the harness reporting the failure, not the command saying
// anything about it, so it is removed before the floor is applied rather
// than paid for with a token allowance -- an allowance is wrong in both
// directions at once, letting "no such file" (3 tokens) through on the
// strength of five status-line tokens while over-charging the 2-token
// "Command aborted". A footered result has already had its harness text
// split off by splitFooter.
const STATUS_LINE_RE =
  /^Command (?:exited with code -?\d+|timed out after [\d.]+ seconds?|aborted)$/;

function stripStatusLine(body: string): string {
  const lines = body.split("\n");
  const last = lines[lines.length - 1] ?? "";
  if (!STATUS_LINE_RE.test(last.trim())) return body;
  return lines.slice(0, -1).join("\n");
}

export interface ResultFacts {
  /** The command failed, however this tool reports failure. */
  failed: boolean;
  /**
   * Enough output beyond the harness's own report of the failure to identify
   * WHICH failure this is.
   */
  hasContent: boolean;
  /**
   * How `failed` was established, when it was. `"exit"` covers both the
   * thrown-result flag and a nonzero footer exit -- either one PROVES the
   * command failed. `"content"` means only the trailing-exception-line check
   * called it failed while the exit signal (masked, or simply absent) did
   * not. record() uses this so a content-only match never resets a live,
   * exit-code-proven streak it merely fails to match (see matches() and the
   * `else` branch below).
   */
  failedBy?: "exit" | "content";
}

// A trailing exception line, independent of exit code. TB's ShellSession
// wraps the model's command in `{ command ; } ; __rc=$?` with no `pipefail`
// (benchmarks/harbor_adapter/little_coder_agent.py's _wrap_command) — under
// bash's default semantics a piped command (`... | head`) reports the LAST
// stage's exit code, so `python3 ... | head -40` reports 0 even when python3
// raised. Content-based detection catches what the (masked) exit code can't.
//
// Applied only to the last non-blank line (see lastNonBlankLine below, used
// by both readResult and signatureOf), and excludes Warning from the
// alternation, so a benign "Error: skipping bad row 7, continuing" earlier
// in the output, or a captured-log ValueError/RuntimeError line inside an
// otherwise-passing run, does not flip a result to failed or supply a false
// excLine. `[\w.]*`, not `\w*`: `\w` does not match `.`, so a qualified type
// name -- `json.decoder.JSONDecodeError`, `requests.exceptions.HTTPError` --
// was invisible to this regex even though it ends in `Error:` like any
// other, and combined with a masked exit code such a failure went entirely
// undetected.
const TRAILING_EXCEPTION_LINE_RE = /^([\w.]*(?:Error|Exception)):\s*(.+)$/;

function lastNonBlankLine(text: string): string {
  const lines = text.split("\n");
  for (let i = lines.length - 1; i >= 0; i--) {
    if (lines[i].trim() !== "") return lines[i];
  }
  return "";
}

/**
 * Whether a tool result is a failure, and whether it says anything.
 *
 * `isError` alone is not enough: it is set only when a tool throws, which
 * pi's built-in bash does but ShellSession — the shell path under
 * Terminal-Bench — never does. ShellSession catches every failure and reports
 * the exit code in its `[exit=N …]` footer instead, so a TB failure arrives
 * with `isError: false` and gating on the flag alone would leave this
 * watchdog silent for that whole benchmark.
 *
 * The content requirement is the counterweight: a silent nonzero exit carries
 * only the harness's status line or footer, which is byte-identical across
 * three unrelated no-match greps and would otherwise read as one repeated
 * failure.
 *
 * Exit code alone is also not enough: it can itself be wrong (see
 * TRAILING_EXCEPTION_LINE_RE above), so a recognizable trailing exception
 * line counts as failed even when the reported exit is 0.
 */
export function readResult(text: string, isError: boolean, minTokens: number): ResultFacts {
  const { body, footer } = splitFooter(text);
  const exit = footerExit(footer);
  const stripped = stripStatusLine(body.trimEnd());
  const looksLikeUncaughtException = TRAILING_EXCEPTION_LINE_RE.test(lastNonBlankLine(stripped));
  const exitProven = isError === true || (exit !== null && exit !== 0);
  const failed = exitProven || looksLikeUncaughtException;
  return {
    failed,
    hasContent: tokenize(stripped).length >= minTokens,
    failedBy: failed ? (exitProven ? "exit" : "content") : undefined,
  };
}

function trimLineEnd(line: string): string {
  let end = line.length;
  while (end > 0) {
    const c = line[end - 1];
    if (c !== " " && c !== "\t" && c !== "\r") break;
    end--;
  }
  return line.slice(0, end);
}

/**
 * The last `maxLines` lines, per-line trailing whitespace removed and runs of
 * blank lines collapsed. Where a failure concludes, and it carries the exit
 * status without anything having to parse it out.
 */
export function normalizeTail(text: string, maxLines: number): string {
  const lines = text.split("\n").map(trimLineEnd);
  const out: string[] = [];
  for (const line of lines) {
    if (line === "" && out[out.length - 1] === "") continue;
    out.push(line);
  }
  while (out.length > 0 && out[0] === "") out.shift();
  while (out.length > 0 && out[out.length - 1] === "") out.pop();
  return out.slice(Math.max(0, out.length - maxLines)).join("\n");
}

// Everything this module reads from a result is at its end: splitFooter
// takes the last line, normalizeTail the last `tailLines`, and the content
// floor a token count any output this size clears many times over. Holding
// more buys nothing and keeps a whole turn's tool output -- every parallel
// call's -- alive until turn_end.
const MAX_RESULT_CHARS = 64 * 1024;

/**
 * The tail of a tool result, cut at a line boundary where there is one.
 * Everything downstream reads the end of the text, so what this drops is
 * output nothing looks at — except where 64 KiB does not reach back the
 * `tailLines` the signature wants, and the signature is then taken over
 * less.
 */
export function boundResultText(text: string): string {
  if (text.length <= MAX_RESULT_CHARS) return text;
  const cut = text.length - MAX_RESULT_CHARS;
  const nl = text.indexOf("\n", cut);
  return text.slice(nl === -1 ? cut : nl + 1);
}

// Bounded edit distance between two short strings (quoted literals, so at
// most a handful of characters) -- classic DP, no early-exit needed at this
// length.
function levenshtein(a: string, b: string): number {
  if (a === b) return 0;
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const curr = [i];
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost);
    }
    prev = curr;
  }
  return prev[b.length];
}

/** A line with every quoted literal blanked, so two lines compare equal iff
 * everything OUTSIDE the quotes is identical. */
function excLineShape(line: string): string {
  return line.replace(/'[^']*'/g, "'\0'").replace(/"[^"]*"/g, '"\0"');
}

/** The contents of each quoted literal, in order. */
function quotedLiterals(line: string): string[] {
  return [...line.matchAll(/'([^']*)'|"([^"]*)"/g)].map((m) => m[1] ?? m[2] ?? "");
}

// A typo'd literal differs from its original by a couple of characters
// (psrn/psm, the motivating case, is 2); two UNRELATED literals -- a
// different file path, a different dict key, a different module name --
// differ by far more than that, however short they are individually.
const LITERAL_EDIT_DISTANCE_FLOOR = 2;

// Finding 3: the previous version blanked every quoted literal outright
// (`'psrn'` and `'timestamp'` both became `'<TOK>'`), so two trailing
// exception lines that differed ONLY in a quoted literal always compared
// equal regardless of how unrelated that literal was -- a `KeyError` on
// 'user_id' and a `KeyError` on 'timestamp' are different bugs, not the same
// bug typo'd. This still lets two lines match on a REPEATED mistake (the
// psrn/psm keyword-argument typo this fallback exists for): identical
// outside the quotes, and the quoted contents themselves close enough
// (bounded edit distance) to be "the same word, misspelled" rather than two
// different words.
function sameExcLine(a: string, b: string): boolean {
  if (a === b) return true;
  if (excLineShape(a) !== excLineShape(b)) return false;
  const litsA = quotedLiterals(a);
  const litsB = quotedLiterals(b);
  return litsA.length === litsB.length && litsA.every((lit, i) => levenshtein(lit, litsB[i]) <= LITERAL_EDIT_DISTANCE_FLOOR);
}

export interface Signature {
  hash: string;
  shingles: Set<string>;
  /**
   * The raw trailing exception line, if the tail has one. Compared by
   * matches() via sameExcLine's shape-plus-edit-distance rule (Finding 3),
   * never by plain string equality.
   */
  excLine?: string;
}

export function signatureOf(toolName: string, text: string, tailLines: number): Signature {
  const tail = normalizeTail(text, tailLines);
  // Finding 4: excLine extraction uses the SAME trailing-line semantics as
  // readResult -- the footer and any harness status line stripped off first,
  // then only the last non-blank line of what's left -- rather than scanning
  // the whole tail for the last match anywhere. Scanning the whole tail let
  // an incidental mid-tail Warning:/Error: line (a benign log line, a
  // caught-and-logged exception earlier in otherwise-passing output) supply
  // an excLine that then matched an unrelated failure via the fallback
  // below.
  const { body } = splitFooter(text);
  const trailingLine = lastNonBlankLine(stripStatusLine(body.trimEnd()));
  const excMatch = TRAILING_EXCEPTION_LINE_RE.exec(trailingLine);
  return {
    hash: createHash("sha1").update(`${toolName.toLowerCase()}\n${tail}`).digest("hex"),
    shingles: bigrams(tokenize(tail)),
    excLine: excMatch ? excMatch[0] : undefined,
  };
}

export interface ResultObservation {
  toolName: string;
  input: unknown;
  text: string;
  isError: boolean;
  /** Set when the producing call is in a fuzzy cluster of its own. */
  clusterId?: number;
  /** That cluster already holds near-identical attempts. */
  corroborated?: boolean;
}

export interface FailureSignatureDetection {
  /**
   * Which of the two things repeated. A corroborated PASSING result is
   * tracked too (see `record`), and calling that a failure would steer the
   * model to explain an error it never got.
   */
  reason: "repeated_failure_signature" | "repeated_output_signature";
  failed: boolean;
  toolName: string;
  count: number;
  sigKey: string;
  corroborated: boolean;
  /** The fuzzy cluster that corroborated it, so one steer can stand for both. */
  clusterId?: number;
  /** Second and final message for this signature. */
  escalated: boolean;
}

interface Entry {
  toolName: string;
  /** Constant per entry: failures and passing results take separate keys. */
  failed: boolean;
  /** How `failed` was established (see ResultFacts.failedBy); undefined for a passing entry. */
  failedBy?: "exit" | "content";
  sig: Signature;
  count: number;
  lastInput: string;
  lastTurn: number;
  corroborated: boolean;
  clusterId?: number;
}

// Bounded because a long session touches many tools; the least recently
// updated slot is the least likely to be mid-loop.
const MAX_TRACKED = 16;

// Finding 2: the floor `matches()` requires alongside an excLine match (see
// below) used to be a bare constant, so tightening the env-configurable
// `threshold` never tightened this fallback path at all. Deriving it from
// `threshold` instead means a stricter threshold genuinely means a stricter
// watchdog everywhere, not just on the whole-tail check. The 0.6 multiplier
// keeps the motivating psrn/psm pair matching (0.9310, 0.7381 Jaccard, per
// the test fixtures) at the 0.95 default, with headroom below the ~0.368
// Jaccard two KeyErrors that share only an exception TYPE score; 0.5 is a
// hard floor so a heavily loosened threshold cannot suppress this check
// outright.
function excLineJaccardFloor(threshold: number): number {
  return Math.max(0.5, threshold * 0.6);
}

/**
 * Counts how many differing attempts produced the same outcome, per tool.
 *
 * Steer-only by contract, like the fuzzy tracker: a matching signature cannot
 * prove the NEXT attempt fails, so it must never gate execution.
 */
export class FailureSignatureTracker {
  private readonly opts: FailsigOptions;
  private entries = new Map<string, Entry>();
  private notified = new Map<string, number>();

  constructor(overrides: Partial<FailsigOptions> = {}) {
    this.opts = failsigOptionsFromEnv(overrides);
  }

  /**
   * Fold one result into its tool's streak, returning the detection due a
   * message. Successes are tracked only when the producing call's fuzzy
   * cluster already holds near-identical attempts (`corroborated`), and then
   * under a cluster-scoped key so a passing result can never overwrite a live
   * error signature.
   */
  record(obs: ResultObservation, turn: number): FailureSignatureDetection | null {
    if (this.opts.streak <= 0) return null;
    const facts = readResult(obs.text, obs.isError, this.opts.minTokens);
    if (!facts.hasContent) return null;
    if (!facts.failed && !(obs.corroborated && obs.clusterId !== undefined)) return null;

    const tool = obs.toolName.toLowerCase();
    // Finding 1: an exit-code-proven failure and a content-only one (a
    // masked exit-0 result whose trailing line merely LOOKS like an
    // exception -- e.g. a model rerunning its own failing command as
    // `... 2>&1 | tail -1`, an ordinary debugging move) took the SAME key,
    // so a content-only observation that failed to match the tracked
    // signature would forget() and reset a live, genuine exit-code-proven
    // streak -- the mismatch says only that THIS attempt isn't a repeat, not
    // that the streak it failed to match has ended. Giving content-only
    // failures their own key makes that structurally impossible: they can
    // never look up, compare against, or forget() the exit-proven entry.
    const contentKey = `${tool}#content`;
    const key = !facts.failed ? `${tool}#c${obs.clusterId}` : facts.failedBy === "content" ? contentKey : tool;
    const sig = signatureOf(tool, obs.text, this.opts.tailLines);
    const input = stableStringify(obs.input);
    const prev = this.entries.get(key);

    // `window` turns INCLUSIVE of both ends, matching the fuzzy tracker's
    // rolling span: with the default 8, turns 1 and 8 still count together
    // and turns 1 and 9 do not.
    if (prev && this.matches(prev.sig, sig) && turn - prev.lastTurn < this.opts.window) {
      // Identical input producing identical output is the verbatim
      // loop-breaker's case; counting it here would double-cover it. The
      // point of this watchdog is changed attempt, unchanged outcome.
      if (input !== prev.lastInput) {
        prev.count++;
        prev.lastInput = input;
        prev.lastTurn = turn;
        prev.corroborated = prev.corroborated || obs.corroborated === true;
        if (obs.corroborated) prev.clusterId = obs.clusterId;
        // Re-inserted to keep Map order recency order: evict() drops the
        // front, so an entry that only ever matches would be evicted
        // mid-streak.
        this.entries.delete(key);
        this.entries.set(key, prev);
      }
    } else {
      this.forget(key);
      // The reverse direction is NOT protected, and deliberately so: an
      // exit-code-proven failure (or reset) is the strongest signal there is
      // that the situation has genuinely changed, so it also invalidates any
      // content-only streak that happened to be accumulating alongside it --
      // otherwise a later content-only observation could resume matching a
      // stale pre-reset streak instead of starting fresh from the new
      // context.
      if (key === tool) this.forget(contentKey);
      this.entries.set(key, {
        toolName: obs.toolName,
        failed: facts.failed,
        failedBy: facts.failedBy,
        sig,
        count: 1,
        lastInput: input,
        lastTurn: turn,
        corroborated: obs.corroborated === true,
        clusterId: obs.corroborated ? obs.clusterId : undefined,
      });
      this.evict();
    }

    return this.due(key);
  }

  /** Consume this signature's message slot. */
  markNotified(sigKey: string): void {
    this.notified.set(sigKey, (this.notified.get(sigKey) ?? 0) + 1);
  }

  private matches(a: Signature, b: Signature): boolean {
    if (a.hash === b.hash) return true;
    if (jaccard(a.shingles, b.shingles) >= this.opts.threshold) return true;
    // OR, not a replacement: a short, information-dense tail (e.g. a 5-line
    // traceback) can fail the whole-tail Jaccard bar on a single differing
    // token while still being "the same mistake" by its exception line. But
    // excLine equality alone is too weak to stand on its own -- sameExcLine
    // already rejects two failures that only share an exception TYPE
    // (Finding 3), and requiring a threshold-derived similarity floor
    // alongside it (Finding 2) still catches the motivating psrn/psm-typo
    // pairs (0.9310, 0.7381 Jaccard) while rejecting unrelated failures on a
    // second, independent axis.
    return (
      a.excLine !== undefined &&
      b.excLine !== undefined &&
      sameExcLine(a.excLine, b.excLine) &&
      jaccard(a.shingles, b.shingles) >= excLineJaccardFloor(this.opts.threshold)
    );
  }

  private notifyKey(key: string, sig: Signature): string {
    return `${key}:${sig.hash.slice(0, 8)}`;
  }

  private due(key: string): FailureSignatureDetection | null {
    const entry = this.entries.get(key);
    if (!entry) return null;
    // Near-identical attempts producing the identical outcome is the same
    // conclusion reached twice over, so it needs one attempt less to be worth
    // saying.
    const streak = entry.corroborated ? this.opts.streak - 1 : this.opts.streak;
    const sigKey = this.notifyKey(key, entry.sig);
    const sent = this.notified.get(sigKey) ?? 0;
    const dueNow = (sent === 0 && entry.count >= streak) || (sent === 1 && entry.count >= streak + 2);
    if (!dueNow) return null;
    return {
      reason: entry.failed ? "repeated_failure_signature" : "repeated_output_signature",
      failed: entry.failed,
      toolName: entry.toolName,
      count: entry.count,
      sigKey,
      corroborated: entry.corroborated,
      clusterId: entry.clusterId,
      escalated: sent === 1,
    };
  }

  private evict(): void {
    while (this.entries.size > MAX_TRACKED) {
      const oldest = this.entries.keys().next().value as string | undefined;
      if (oldest === undefined) return;
      this.forget(oldest);
    }
  }

  /**
   * Drop an entry and the message budget spent on it, so `notified` stays
   * bounded by the tracked entries rather than by how long the session runs.
   * Safe because a live streak's `sig` is never replaced, so its notify key
   * does not move while it is being counted.
   */
  private forget(key: string): void {
    const entry = this.entries.get(key);
    if (entry) this.notified.delete(this.notifyKey(key, entry.sig));
    this.entries.delete(key);
  }
}
