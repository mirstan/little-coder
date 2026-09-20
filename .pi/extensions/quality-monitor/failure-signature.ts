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

export function failsigOptionsFromEnv(overrides: Partial<FailsigOptions> = {}): FailsigOptions {
  return {
    tailLines: envNumber("LITTLE_CODER_FAILSIG_TAIL_LINES", 40),
    // Tighter than the fuzzy detector's 0.85: "the identical outcome" is a
    // stronger claim than "a similar attempt".
    threshold: envNumber("LITTLE_CODER_FAILSIG_THRESHOLD", 0.95),
    streak: envNumber("LITTLE_CODER_FAILSIG_STREAK", 3),
    window: envNumber("LITTLE_CODER_FAILSIG_WINDOW", 8),
    minTokens: envNumber("LITTLE_CODER_FAILSIG_MIN_TOKENS", 4),
    ...overrides,
  };
}

// Without a footer the text still ends in pi's own appended status line
// ("Command exited with code N", "Command timed out after N seconds"), so the
// floor has to clear that before any real output counts. A footered result
// has already had its harness text split off.
const STATUS_LINE_ALLOWANCE = 4;

export interface ResultFacts {
  /** The command failed, however this tool reports failure. */
  failed: boolean;
  /**
   * Enough output beyond the harness's own report of the failure to identify
   * WHICH failure this is.
   */
  hasContent: boolean;
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
 */
export function readResult(text: string, isError: boolean, minTokens: number): ResultFacts {
  const { body, footer } = splitFooter(text);
  const exit = footerExit(footer);
  const floor = footer === null ? minTokens + STATUS_LINE_ALLOWANCE : minTokens;
  return {
    failed: isError === true || (exit !== null && exit !== 0),
    hasContent: tokenize(body).length >= floor,
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

export interface Signature {
  hash: string;
  shingles: Set<string>;
}

export function signatureOf(toolName: string, text: string, tailLines: number): Signature {
  const tail = normalizeTail(text, tailLines);
  return {
    hash: createHash("sha1").update(`${toolName.toLowerCase()}\n${tail}`).digest("hex"),
    shingles: bigrams(tokenize(tail)),
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
  reason: "repeated_failure_signature";
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
  sig: Signature;
  count: number;
  lastInput: string;
  lastTurn: number;
  corroborated: boolean;
  clusterId?: number;
}

// Bounded because a long session touches many tools; the oldest slot is the
// least likely to be mid-loop.
const MAX_TRACKED = 16;

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
   * message. Successes are tracked only when the producing call is already in
   * a fuzzy cluster, and then under a cluster-scoped key so a passing result
   * can never overwrite a live error signature.
   */
  record(obs: ResultObservation, turn: number): FailureSignatureDetection | null {
    if (this.opts.streak <= 0) return null;
    const facts = readResult(obs.text, obs.isError, this.opts.minTokens);
    if (!facts.hasContent) return null;
    if (!facts.failed && !(obs.corroborated && obs.clusterId !== undefined)) return null;

    const tool = obs.toolName.toLowerCase();
    const key = facts.failed ? tool : `${tool}#c${obs.clusterId}`;
    const sig = signatureOf(tool, obs.text, this.opts.tailLines);
    const input = stableStringify(obs.input);
    const prev = this.entries.get(key);

    if (prev && this.matches(prev.sig, sig) && turn - prev.lastTurn <= this.opts.window) {
      // Identical input producing identical output is the verbatim
      // loop-breaker's case; counting it here would double-cover it. The
      // point of this watchdog is changed attempt, unchanged outcome.
      if (input !== prev.lastInput) {
        prev.count++;
        prev.lastInput = input;
        prev.lastTurn = turn;
        prev.corroborated = prev.corroborated || obs.corroborated === true;
        if (obs.corroborated) prev.clusterId = obs.clusterId;
      }
    } else {
      this.entries.delete(key);
      this.entries.set(key, {
        toolName: obs.toolName,
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
    return jaccard(a.shingles, b.shingles) >= this.opts.threshold;
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
      reason: "repeated_failure_signature",
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
      this.entries.delete(oldest);
    }
  }
}
