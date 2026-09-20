// Near-duplicate detection for tool calls whose inputs vary but whose content
// repeats. The verbatim loop-breaker in quality.ts compares
// JSON.stringify(input) for equality, so a probe loop that changes one
// constant per attempt, or three rewrites of one script with cosmetic
// differences, is invisible to it by construction.
//
// Scoring is a similarity score rather than normalize-then-compare because a
// normalizer has to know in advance WHICH detail varies (a number, a path, a
// flag, an identifier) and silently stops matching when a new variation class
// appears. A score degrades toward "not similar", i.e. toward not firing.

import { createHash } from "node:crypto";
import { envNumber } from "../_shared/env-number.ts";
import type { ToolCall } from "./quality.ts";

// The only pattern in this module, and deliberately so: a total lexer over
// arbitrary text, not a recognizer of any format. Every input maps to a token
// list, so there is no "stopped matching" state to drift into. Non-Latin
// identifiers (CJK source, emoji) collapse to few or no tokens, which puts
// such calls below the min-token floor and turns fuzzy detection off for them
// rather than making it wrong.
const WORD_RE = /[a-z0-9_]+/g;

export function tokenize(text: string): string[] {
  return text.toLowerCase().match(WORD_RE) ?? [];
}

/**
 * Adjacent token pairs. Unigrams over-match — two unrelated perl scripts share
 * most keywords — while a single changed token only moves two bigrams, so a
 * long near-duplicate still scores ~0.98.
 */
export function bigrams(tokens: string[]): Set<string> {
  const out = new Set<string>();
  for (let i = 0; i + 1 < tokens.length; i++) out.add(`${tokens[i]} ${tokens[i + 1]}`);
  return out;
}

/** |a ∩ b| / |a ∪ b|. Both empty → 1; exactly one empty → 0. */
export function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) return 1;
  if (a.size === 0 || b.size === 0) return 0;
  let shared = 0;
  const [small, large] = a.size <= b.size ? [a, b] : [b, a];
  for (const s of small) if (large.has(s)) shared++;
  return shared / (a.size + b.size - shared);
}

/** Fraction of `part` also present in `whole`. */
export function containment(part: Set<string>, whole: Set<string>): number {
  if (part.size === 0) return 1;
  let shared = 0;
  for (const s of part) if (whole.has(s)) shared++;
  return shared / part.size;
}

/**
 * JSON with object keys sorted, so two inputs differing only in key order
 * produce identical text. pi's coercion paths do not guarantee key order, and
 * reordering alone would otherwise deflate similarity.
 */
export function stableStringify(value: unknown, depth = 0): string {
  if (depth > 12 || value === null || typeof value !== "object") return JSON.stringify(value) ?? "null";
  if (Array.isArray(value)) return `[${value.map((v) => stableStringify(v, depth + 1)).join(",")}]`;
  const keys = Object.keys(value as Record<string, unknown>).sort();
  const parts = keys.map(
    (k) => `${JSON.stringify(k)}:${stableStringify((value as Record<string, unknown>)[k], depth + 1)}`,
  );
  return `{${parts.join(",")}}`;
}

// Tokenizing a whole multi-megabyte Write bounds nothing. Head AND tail,
// rather than a prefix: two files sharing a long generated preamble and
// nothing else would compare as that preamble alone and score 1.0.
const MAX_COMPARE_CHARS = 16 * 1024;
const SAMPLE_HALF = MAX_COMPARE_CHARS / 2;

/**
 * At most `MAX_COMPARE_CHARS` of `text`, taken from both ends. A change in
 * the middle of a file larger than that is invisible to the comparison, so
 * this can still report two such files as similar — it cannot make two files
 * that differ at either end look alike.
 */
export function sampleForCompare(text: string): string {
  if (text.length <= MAX_COMPARE_CHARS) return text;
  return `${text.slice(0, SAMPLE_HALF)}\n${text.slice(text.length - SAMPLE_HALF)}`;
}

function str(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

/**
 * The part of a call that carries its meaning, per tool. Reads well-known
 * property names with typeof checks and falls back to the whole input, so an
 * unknown tool or a missing field is still comparable rather than skipped.
 *
 * Returned whole; `sampleForCompare` bounds what is tokenized. The full
 * length is what the growth exemption measures, which a bounded sample can
 * no longer report.
 */
export function extractComparableText(call: ToolCall): string {
  const input = call.input;
  const obj = input && typeof input === "object" ? (input as Record<string, unknown>) : null;
  let text: string | null = null;
  if (obj) {
    const tool = call.name.toLowerCase();
    const path = str(obj.path) ?? str(obj.file_path);
    if (tool === "bash" || tool === "shellsession") {
      text = str(obj.command);
    } else if (tool === "write") {
      const content = str(obj.content);
      if (content !== null) text = `${path ?? ""}\n${content}`;
    } else if (tool === "edit" || tool === "multiedit") {
      if (obj.edits !== undefined) text = `${path ?? ""}\n${stableStringify(obj.edits)}`;
      else if (obj.old_string !== undefined || obj.new_string !== undefined) {
        text = `${path ?? ""}\n${stableStringify(obj.old_string)}\n${stableStringify(obj.new_string)}`;
      }
    }
  }
  return text ?? stableStringify(input);
}

/**
 * A call's identity, for excluding verbatim repeats from the clustering --
 * must agree with `sameCall`'s notion of "the same call", or a repeat one
 * counts as verbatim and the other doesn't evades both detectors: skipped
 * here as "the loop-breaker's job", but not actually caught by the
 * loop-breaker. So this hashes the same order-sensitive JSON `sameCall`
 * compares, not `stableStringify` (key-order-insensitive, correct for
 * `extractComparableText`'s fuzzy-similarity text, wrong for identity).
 *
 * Hashed rather than kept whole: a window entry outlives its call by up to
 * `window` turns, and the serialized input of a multi-megabyte Write would
 * otherwise be retained per entry -- the same unbounded retention
 * sampleForCompare exists to prevent.
 */
export function identityKey(call: ToolCall): string {
  return createHash("sha1").update(`${call.name} ${JSON.stringify(call.input)}`).digest("hex");
}

export interface FuzzyOptions {
  threshold: number;
  minTokens: number;
  window: number;
  streak: number;
  /** Minimum size ratio for the growth exemption; ≤ 1 disables it. */
  growthRatio: number;
}

// Named here, beside the only code that reads them, so the tests that have
// to neutralize them cannot drift from the set that exists.
export const FUZZY_ENV = {
  threshold: "LITTLE_CODER_FUZZY_LOOP_THRESHOLD",
  minTokens: "LITTLE_CODER_FUZZY_LOOP_MIN_TOKENS",
  window: "LITTLE_CODER_FUZZY_LOOP_WINDOW",
  streak: "LITTLE_CODER_FUZZY_LOOP_STREAK",
  growthRatio: "LITTLE_CODER_FUZZY_LOOP_GROWTH_RATIO",
} as const;

export function fuzzyOptionsFromEnv(overrides: Partial<FuzzyOptions> = {}): FuzzyOptions {
  return {
    threshold: envNumber(FUZZY_ENV.threshold, 0.85),
    minTokens: envNumber(FUZZY_ENV.minTokens, 20),
    window: envNumber(FUZZY_ENV.window, 8),
    streak: envNumber(FUZZY_ENV.streak, 3),
    growthRatio: envNumber(FUZZY_ENV.growthRatio, 1.02),
    ...overrides,
  };
}

// A later version containing nearly all of an earlier one AND being larger is
// a file being built up, not an attempt being retried. Containment is measured
// over shingle sets, so it needs no notion of what changed; size is measured
// in characters of the whole text, because past MAX_COMPARE_CHARS a sampled
// shingle count stops growing with the file. The ratio is small because the
// two cases are already close: appending ~8% of a file scores ~0.92 against
// its predecessor, while retrying it with cosmetic edits keeps the size
// within ~1%.
const GROWTH_CONTAINMENT = 0.95;

export interface NearDuplicateDetection {
  reason: "near_duplicate_loop";
  clusterId: number;
  count: number;
  toolName: string;
  /** Second and final message for this cluster. */
  escalated: boolean;
}

/** What one call contributes to the comparison, already shingled. */
interface Shingled {
  shingles: Set<string>;
  /**
   * Bigrams of the leading half of the sample alone. Identical to `shingles`
   * unless the text was sampled: appending to a file slides the tail sample
   * but cannot move its head, so the growth exemption reads this instead.
   */
  head: Set<string>;
  /** Characters of comparable text before sampling — the growth measure. */
  length: number;
}

interface WindowEntry extends Shingled {
  tool: string;
  /** name + stable input, for excluding verbatim repeats. */
  key: string;
  turn: number;
  cluster: number;
}

/**
 * Clusters near-identical calls across a rolling turn window and reports a
 * cluster once it holds `streak` distinct turns' calls.
 *
 * Steer-only by contract: a similarity match can never prove the next attempt
 * is wrong, so nothing here feeds the hard block.
 */
export class FuzzyLoopTracker {
  private readonly opts: FuzzyOptions;
  private entries: WindowEntry[] = [];
  private clusterIds = new Map<string, number>();
  private notified = new Map<number, number>();
  private nextCluster = 1;

  constructor(overrides: Partial<FuzzyOptions> = {}) {
    this.opts = fuzzyOptionsFromEnv(overrides);
  }

  /**
   * Fold one assessed turn's calls into the window.
   *
   * Returns the detection that is due a message, if any — computed fresh each
   * turn rather than queued, so a detection suppressed by a higher-priority
   * intervention re-surfaces only while the cluster is still growing.
   */
  recordTurn(turn: number, calls: { call: ToolCall; id?: string }[]): NearDuplicateDetection | null {
    if (this.opts.streak <= 0) return null;
    this.prune(turn);

    const added: WindowEntry[] = [];
    for (const { call, id } of calls) {
      const text = extractComparableText(call);
      const sample = sampleForCompare(text);
      const tokens = tokenize(sample);
      if (tokens.length < this.opts.minTokens) continue;
      const shingles = bigrams(tokens);
      const now: Shingled = {
        shingles,
        head: sample === text ? shingles : bigrams(tokenize(sample.slice(0, SAMPLE_HALF))),
        length: text.length,
      };
      const tool = call.name.toLowerCase();
      const key = identityKey(call);

      const matched: number[] = [];
      for (const entry of this.entries) {
        if (entry.tool !== tool) continue;
        // Verbatim repeats belong to the existing loop-breaker; clustering
        // them here would double-fire with it.
        if (entry.key === key) continue;
        if (this.isGrowth(entry, now)) continue;
        if (jaccard(entry.shingles, shingles) >= this.opts.threshold) matched.push(entry.cluster);
      }

      let cluster: number;
      if (matched.length === 0) {
        cluster = this.nextCluster++;
      } else {
        cluster = Math.min(...matched);
        // Transitive: a call matching two clusters proves they were one.
        for (const entry of this.entries) if (matched.includes(entry.cluster)) entry.cluster = cluster;
        for (const [id, c] of this.clusterIds) if (matched.includes(c)) this.clusterIds.set(id, cluster);
        // The merged cluster inherits the largest message budget already
        // spent, or attempts that have been steered about once would earn a
        // first message again under the surviving id.
        const sent = Math.max(...matched.map((c) => this.notified.get(c) ?? 0));
        for (const c of matched) this.notified.delete(c);
        if (sent > 0) this.notified.set(cluster, sent);
      }
      const entry: WindowEntry = { ...now, tool, key, turn, cluster };
      this.entries.push(entry);
      added.push(entry);
      if (id !== undefined) this.clusterIds.set(id, cluster);
    }

    // The entries, not the cluster ids read off them: a later call in this
    // same turn can merge clusters, which renumbers `entry.cluster` in place.
    return this.due(added);
  }

  clusterIdForToolCallId(id: string | undefined): number | undefined {
    return id === undefined ? undefined : this.clusterIds.get(id);
  }

  /** Distinct turns the cluster spans — its member count, one per turn. */
  clusterSize(cluster: number | undefined): number {
    if (cluster === undefined) return 0;
    const turns = new Set<number>();
    for (const entry of this.entries) if (entry.cluster === cluster) turns.add(entry.turn);
    return turns.size;
  }

  /** Consume this cluster's message slot; also used to cross-suppress. */
  markNotified(cluster: number): void {
    this.notified.set(cluster, (this.notified.get(cluster) ?? 0) + 1);
  }

  private isGrowth(earlier: Shingled, later: Shingled): boolean {
    if (this.opts.growthRatio <= 1) return false;
    if (later.length < earlier.length * this.opts.growthRatio) return false;
    return containment(earlier.head, later.head) >= GROWTH_CONTAINMENT;
  }

  // Only clusters this turn added a call to. A cluster that merely survives
  // in the window is not news: without this, a detection a higher-priority
  // intervention suppressed would re-fire on every later turn until its last
  // member aged out, long after the loop it described had stopped.
  private due(added: WindowEntry[]): NearDuplicateDetection | null {
    const seen = new Set<number>();
    let best: NearDuplicateDetection | null = null;
    for (const entry of added) {
      if (seen.has(entry.cluster)) continue;
      seen.add(entry.cluster);
      const count = this.clusterSize(entry.cluster);
      const sent = this.notified.get(entry.cluster) ?? 0;
      const dueNow =
        (sent === 0 && count >= this.opts.streak) || (sent === 1 && count >= this.opts.streak + 2);
      if (!dueNow) continue;
      if (!best || count > best.count) {
        best = {
          reason: "near_duplicate_loop",
          clusterId: entry.cluster,
          count,
          toolName: entry.tool,
          escalated: sent === 1,
        };
      }
    }
    return best;
  }

  // A cluster dies with its last window entry: a loop slow enough for every
  // member to age out is the failure-signature watchdog's job, not this one's.
  private prune(turn: number): void {
    const oldest = turn - this.opts.window + 1;
    this.entries = this.entries.filter((e) => e.turn >= oldest);
    const live = new Set(this.entries.map((e) => e.cluster));
    for (const cluster of [...this.notified.keys()]) if (!live.has(cluster)) this.notified.delete(cluster);
    for (const [id, cluster] of [...this.clusterIds]) if (!live.has(cluster)) this.clusterIds.delete(id);
  }
}
