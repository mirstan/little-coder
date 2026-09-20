import { describe, it, expect, afterEach, beforeEach } from "vitest";
import {
  bigrams,
  containment,
  extractComparableText,
  FUZZY_ENV,
  FuzzyLoopTracker,
  identityKey,
  jaccard,
  sampleForCompare,
  stableStringify,
  tokenize,
} from "./similarity.ts";
import { pinEnv } from "../_shared/env-pin.ts";

// Every `new FuzzyLoopTracker()` below asserts against the built-in
// defaults, which fuzzyOptionsFromEnv reads out of the environment. A shell
// (or a CI matrix) exporting any of them would silently retune the detector
// under the assertions.
const pinned = pinEnv(Object.values(FUZZY_ENV));
beforeEach(() => pinned.clear());
afterEach(() => pinned.restore());

const sim = (a: string, b: string) => jaccard(bigrams(tokenize(a)), bigrams(tokenize(b)));

// A ~100-line script with all-distinct lines, the shape the detector exists
// for: long content whose next version differs in one place.
const VERBS = ["read", "parse", "scan", "fold", "merge", "emit", "flush", "count", "index", "hash", "pack", "trim"];
function script(limit = 512, buf = "buf"): string {
  const lines = ["#!/usr/bin/perl", "use strict;", "use warnings;", `my $${buf} = "";`, `my $limit = ${limit};`];
  for (let i = 0; i < 95; i++) {
    const v = VERBS[i % VERBS.length];
    lines.push(
      i % 8 === 0
        ? `  my $${buf}_${i} = ${v}_stage($fh, ${i});`
        : `  $out[${i}] = ${v}_${i}($state, "${v}-${i}", ${i * 3 + 1});`,
    );
  }
  return lines.join("\n");
}

describe("tokenize/bigrams/jaccard", () => {
  it("splits on anything outside [a-z0-9_] and lowercases", () => {
    expect(tokenize("Foo-Bar_baz(9);")).toEqual(["foo", "bar_baz", "9"]);
  });
  it("returns no tokens for text with no word characters", () => {
    expect(tokenize("… ✂︎ 。")).toEqual([]);
  });
  it("is deterministic", () => {
    expect(tokenize("a b a b")).toEqual(tokenize("a b a b"));
  });
  it("pairs adjacent tokens", () => {
    expect([...bigrams(tokenize("a b c"))]).toEqual(["a b", "b c"]);
  });
  it("cannot collide on its join character", () => {
    // Tokens never contain a space, so "a b"+"c" and "a"+"b c" are
    // unreachable as distinct token lists.
    expect(tokenize("a b c").every((t) => !t.includes(" "))).toBe(true);
  });
  it("scores two empty sets as identical and one empty set as disjoint", () => {
    expect(jaccard(new Set(), new Set())).toBe(1);
    expect(jaccard(new Set(["a b"]), new Set())).toBe(0);
  });
  it("scores identical sets as 1", () => {
    expect(sim("the same text here", "the same text here")).toBe(1);
  });
  it("measures containment against the first set", () => {
    expect(containment(new Set(["a", "b"]), new Set(["a", "b", "c"]))).toBe(1);
    expect(containment(new Set(["a", "b"]), new Set(["a"]))).toBe(0.5);
  });
});

// These pin the tuning: a threshold change shows up here first.
describe("similarity calibration", () => {
  it("scores a 100-line script with one constant changed ≥ 0.95", () => {
    expect(sim(script(512), script(1024))).toBeGreaterThanOrEqual(0.95);
  });
  it("scores the same script with a variable renamed throughout ≥ 0.85", () => {
    // The variation class a number-normalizing scrubber could not see.
    expect(sim(script(), script(512, "scratch"))).toBeGreaterThanOrEqual(0.85);
  });
  it("scores a rewritten 20-line region of 100 below 0.85", () => {
    const rewritten = script()
      .split("\n")
      .map((l, i) => (i >= 40 && i < 60 ? `  $acc = reduce_window($acc, ${i}, $limit) or last;` : l))
      .join("\n");
    expect(sim(script(), rewritten)).toBeLessThan(0.85);
  });
  it("scores two unrelated scripts sharing keywords below 0.5", () => {
    const other = [
      "#!/usr/bin/perl",
      "use strict;",
      "use warnings;",
      ...Array.from({ length: 90 }, (_, i) => `  print $fh sprintf("%s=%d\\n", $keys[${i}], $vals[${i}]);`),
    ].join("\n");
    expect(sim(script(), other)).toBeLessThan(0.5);
  });
  it("keeps two punctuation-dense one-liners apart despite shared sigils", () => {
    const a = `perl -ne 'chomp; next if /^#/; my @f = split /,/; print join("|", @f[0,2,4]), "\\n" if $f[1] > 10 and $f[3] ne "skip";'`;
    const b = `perl -ne 'chomp; next unless /ERROR/; my @g = split /\\s+/; warn join(":", @g[1,3]), "\\n" if $g[5] eq "fatal" or $g[6];'`;
    expect(tokenize(a).length).toBeGreaterThanOrEqual(20);
    expect(tokenize(b).length).toBeGreaterThanOrEqual(20);
    expect(sim(a, b)).toBeLessThan(0.85);
  });
  it("leaves short commands below the min-token floor entirely", () => {
    expect(tokenize("ls -la").length).toBeLessThan(20);
    expect(tokenize("cat /tmp/out.txt").length).toBeLessThan(20);
  });
  // Over n distinct tokens there are n-1 bigrams. Changing one INTERIOR
  // token rewrites the two bigrams it sits in, leaving n-3 shared out of a
  // union of n+1: (n-3)/(n+1), which reaches 0.85 at n = 26. Changing the
  // FIRST or LAST token rewrites only one bigram: (n-2)/n, which reaches
  // 0.85 at n = 14 — well inside the eligible range, since the min-token
  // floor is 20. So similarity alone does NOT keep a short command with one
  // varying constant quiet; where the constant usually sits (trailing, as in
  // `--window 512`) it scores ≥ 0.90 as soon as the command is long enough
  // to be looked at, and the min-token floor is the whole of what excludes
  // shorter ones. Firing on a long command that sweeps a trailing constant
  // is this detector's headline case, not a false positive.
  it("takes 26 tokens for one changed interior token to reach 0.85", () => {
    const seq = (n: number, at: number) =>
      Array.from({ length: n }, (_, i) => (i === at ? "changed" : `tok_${i}`)).join(" ");
    const interior = (n: number) => sim(seq(n, -1), seq(n, Math.floor(n / 2)));
    expect(interior(25)).toBeCloseTo(22 / 26, 6);
    expect(interior(25)).toBeLessThan(0.85);
    expect(interior(26)).toBeCloseTo(23 / 27, 6);
    expect(interior(26)).toBeGreaterThanOrEqual(0.85);
  });
  it("takes 14 tokens for one changed trailing token to reach 0.85", () => {
    const seq = (n: number, at: number) =>
      Array.from({ length: n }, (_, i) => (i === at ? "changed" : `tok_${i}`)).join(" ");
    const trailing = (n: number) => sim(seq(n, -1), seq(n, n - 1));
    expect(trailing(13)).toBeCloseTo(11 / 13, 6);
    expect(trailing(13)).toBeLessThan(0.85);
    expect(trailing(14)).toBeCloseTo(12 / 14, 6);
    expect(trailing(14)).toBeGreaterThanOrEqual(0.85);
  });
});

describe("sampleForCompare", () => {
  const big = (fill: string) => fill.repeat(Math.ceil(40000 / fill.length)).slice(0, 40000);
  it("returns text at or under the cap unchanged", () => {
    expect(sampleForCompare("make test")).toBe("make test");
  });
  it("bounds a large text to the cap plus its own join", () => {
    expect(sampleForCompare(big("abcdefgh")).length).toBe(16 * 1024 + 1);
  });
  it("keeps both ends, so a shared prefix alone is not the whole comparison", () => {
    const text = `${"head ".repeat(4000)}TAILMARKER`;
    const sampled = sampleForCompare(text);
    expect(sampled.startsWith("head head")).toBe(true);
    expect(sampled.endsWith("TAILMARKER")).toBe(true);
  });
});

describe("extractComparableText", () => {
  it("reads the command for shell tools", () => {
    expect(extractComparableText({ name: "Bash", input: { command: "make test" } })).toBe("make test");
    expect(extractComparableText({ name: "ShellSession", input: { command: "make test" } })).toBe("make test");
  });
  it("reads path and content for a write", () => {
    const text = extractComparableText({ name: "Write", input: { path: "/a.pl", content: "print 1;" } });
    expect(text).toBe("/a.pl\nprint 1;");
  });
  it("serializes an edit's edits array", () => {
    const text = extractComparableText({
      name: "Edit",
      input: { path: "/a.pl", edits: [{ oldText: "1", newText: "2" }] },
    });
    expect(text).toContain("/a.pl");
    expect(text).toContain("oldText");
  });
  it("falls back to the whole input when the known fields are missing", () => {
    expect(extractComparableText({ name: "Bash", input: { cmd: "make" } })).toBe('{"cmd":"make"}');
  });
  it("is insensitive to object key order", () => {
    const a = extractComparableText({ name: "Grep", input: { a: 1, b: 2 } });
    const b = extractComparableText({ name: "Grep", input: { b: 2, a: 1 } });
    expect(a).toBe(b);
  });
  it("returns the whole text — bounding it is sampleForCompare's job", () => {
    const text = extractComparableText({ name: "Write", input: { path: "/a", content: "x".repeat(40000) } });
    expect(text.length).toBe(40003);
  });
});

describe("identityKey", () => {
  it("is the same length whatever the call carries", () => {
    const small = identityKey({ name: "Write", input: { path: "/a.pl", content: "print 1;" } });
    const large = identityKey({ name: "Write", input: { path: "/a.pl", content: "x".repeat(2_000_000) } });
    expect(small).toHaveLength(40);
    expect(large).toHaveLength(40);
  });
  it("separates calls that differ and matches calls that do not", () => {
    const call = (content: string) => ({ name: "Write", input: { path: "/a.pl", content } });
    expect(identityKey(call("a"))).toBe(identityKey(call("a")));
    expect(identityKey(call("a"))).not.toBe(identityKey(call("b")));
    expect(identityKey({ name: "Write", input: 1 })).not.toBe(identityKey({ name: "Edit", input: 1 }));
  });
  it("ignores object key order, as the comparison text does", () => {
    expect(identityKey({ name: "Grep", input: { a: 1, b: 2 } })).toBe(
      identityKey({ name: "Grep", input: { b: 2, a: 1 } }),
    );
  });
});

describe("stableStringify", () => {
  it("sorts keys at every depth", () => {
    expect(stableStringify({ b: { d: 1, c: 2 }, a: 3 })).toBe('{"a":3,"b":{"c":2,"d":1}}');
  });
  it("preserves array order", () => {
    expect(stableStringify([3, 1, 2])).toBe("[3,1,2]");
  });
});

describe("FuzzyLoopTracker", () => {
  const scriptLines = () => script().split("\n");
  const call = (name: string, input: unknown) => ({ call: { name, input } });
  const write = (content: string) => call("Write", { path: "/a.pl", content });

  it("reports a cluster once it spans `streak` turns", () => {
    const t = new FuzzyLoopTracker();
    expect(t.recordTurn(1, [write(script(1))])).toBeNull();
    expect(t.recordTurn(2, [write(script(2))])).toBeNull();
    const d = t.recordTurn(3, [write(script(3))]);
    expect(d).toMatchObject({ reason: "near_duplicate_loop", count: 3 });
  });

  it("counts a turn once however many near-identical calls it makes", () => {
    const t = new FuzzyLoopTracker();
    expect(t.recordTurn(1, [write(script(1)), write(script(2)), write(script(3))])).toBeNull();
    expect(t.recordTurn(2, [write(script(4)), write(script(5))])).toBeNull();
    expect(t.recordTurn(3, [write(script(6))])).toMatchObject({ count: 3 });
  });

  it("joins transitively matched calls into one cluster", () => {
    const t = new FuzzyLoopTracker();
    t.recordTurn(1, [{ call: { name: "Write", input: { path: "/a.pl", content: script(1) } }, id: "a" }]);
    t.recordTurn(2, [{ call: { name: "Write", input: { path: "/a.pl", content: script(2) } }, id: "b" }]);
    t.recordTurn(3, [{ call: { name: "Write", input: { path: "/a.pl", content: script(3) } }, id: "c" }]);
    expect(t.clusterIdForToolCallId("a")).toBe(t.clusterIdForToolCallId("c"));
    expect(t.clusterSize(t.clusterIdForToolCallId("a"))).toBe(3);
  });

  it("does not cluster calls made by different tools", () => {
    const t = new FuzzyLoopTracker();
    t.recordTurn(1, [write(script(1))]);
    t.recordTurn(2, [call("ShellSession", { command: script(2) })]);
    expect(t.recordTurn(3, [write(script(3))])).toBeNull();
  });

  it("excludes verbatim repeats, which the loop-breaker already owns", () => {
    const t = new FuzzyLoopTracker();
    const same = script(1);
    t.recordTurn(1, [write(same)]);
    t.recordTurn(2, [write(same)]);
    expect(t.recordTurn(3, [write(same)])).toBeNull();
  });

  it("drops entries that age out of the window", () => {
    const t = new FuzzyLoopTracker({ window: 3 });
    t.recordTurn(1, [write(script(1))]);
    t.recordTurn(2, [write(script(2))]);
    // Turn 4's window starts at turn 2, so turn 1's entry is gone and the
    // cluster is two turns wide, not three.
    expect(t.recordTurn(4, [write(script(3))])).toBeNull();
  });

  it("exempts a file being built up from a near-duplicate match", () => {
    // Incremental authoring: each version contains the previous one and is
    // meaningfully larger. Pairwise similarity is above threshold, so only
    // the growth exemption keeps this quiet.
    const lines = script().split("\n");
    const t = new FuzzyLoopTracker();
    expect(sim(lines.slice(0, 60).join("\n"), lines.slice(0, 65).join("\n"))).toBeGreaterThan(0.85);
    expect(t.recordTurn(1, [write(lines.slice(0, 60).join("\n"))])).toBeNull();
    expect(t.recordTurn(2, [write(lines.slice(0, 65).join("\n"))])).toBeNull();
    expect(t.recordTurn(3, [write(lines.slice(0, 70).join("\n"))])).toBeNull();
    expect(t.recordTurn(4, [write(lines.slice(0, 75).join("\n"))])).toBeNull();
  });

  it("still reports rewrites that only change cosmetic details", () => {
    // The same exemption must not swallow the case it sits next to: these
    // versions are the same size, so nothing is being built up.
    const lines = script().split("\n");
    const v = (i: number, note: string) =>
      write(lines.map((l, j) => (j === i ? `${l}  # ${note}` : l)).join("\n"));
    const t = new FuzzyLoopTracker();
    t.recordTurn(1, [v(7, "first")]);
    t.recordTurn(2, [v(20, "second")]);
    expect(t.recordTurn(3, [v(31, "third")])).toMatchObject({ count: 3 });
  });

  it("carries a merged cluster's spent messages onto the surviving one", () => {
    // Two clusters kept apart by a rewritten region, then one call similar
    // enough to both to prove they were one. The notified cluster's id is not
    // the survivor, so only carrying its budget over keeps the merged cluster
    // from repeating a message it already sent.
    const patch = (from: number, tag: string, note = "") =>
      scriptLines()
        .map((l, i) =>
          i >= from && i < from + 5
            ? `  $accumulator_${tag}_${i} = reduce_${tag}($acc, ${i}, $limit, $state, "${tag}-${i}") or last_${tag}();${note}`
            : l,
        )
        .join("\n");
    // Both directions of this fixture have to hold at once, and the assertions
    // below say nothing if either lands within noise of the threshold. Five
    // patched lines keeps every margin at ~5%: each patch scores ~0.895
    // against the unpatched script (merge) and ~0.804 against the other patch
    // (separation). Six lines put the merge at ~0.876, a 3% margin that a
    // changed verb or line length could flip.
    expect(sim(patch(10, "v"), scriptLines().join("\n"))).toBeGreaterThan(0.89);
    expect(sim(patch(60, "w"), scriptLines().join("\n"))).toBeGreaterThan(0.89);
    expect(sim(patch(10, "v"), patch(60, "w"))).toBeLessThan(0.81);
    const t = new FuzzyLoopTracker({ window: 20 });
    t.recordTurn(1, [write(patch(10, "v"))]);
    t.recordTurn(2, [write(patch(10, "v", " # retry"))]);
    t.recordTurn(3, [write(patch(60, "w"))]);
    t.recordTurn(4, [write(patch(60, "w", " # retry"))]);
    const first = t.recordTurn(5, [write(patch(60, "w", " # again"))]);
    expect(first).toMatchObject({ count: 3, escalated: false });
    t.markNotified(first!.clusterId);
    const merged = t.recordTurn(6, [write(scriptLines().join("\n"))]);
    expect(merged).toMatchObject({ escalated: true });
  });

  it("leaves short commands alone even when they score above the threshold", () => {
    // The min-token floor, not the similarity math, is what keeps short
    // commands out: this pair is 15 tokens and scores 0.867, because a
    // changed LAST token moves only one bigram. Without the floor these three
    // turns would cluster and steer.
    const cmd = (n: number) =>
      `./compressor --input /data/corpus.bin --output /tmp/out.bin --threads 4 --level 9 --window ${n}`;
    expect(tokenize(cmd(512)).length).toBe(15);
    expect(sim(cmd(512), cmd(1024))).toBeGreaterThan(0.85);
    const t = new FuzzyLoopTracker();
    t.recordTurn(1, [call("ShellSession", { command: cmd(512) })]);
    t.recordTurn(2, [call("ShellSession", { command: cmd(1024) })]);
    expect(t.recordTurn(3, [call("ShellSession", { command: cmd(2048) })])).toBeNull();
  });

  it("does not match two large files on a shared prefix alone", () => {
    // Past the sampling cap a head-only comparison sees nothing but the
    // shared preamble and scores 1.0 on unrelated bodies.
    const body = (tag: string) =>
      Array.from({ length: 200 }, (_, i) => `  $out_${tag}[${i}] = ${tag}_${i}($state_${tag}, ${i * 3});`).join("\n");
    const preamble = [body("hdr_a"), body("hdr_b")].join("\n");
    expect(preamble.length).toBeGreaterThan(16 * 1024);
    const t = new FuzzyLoopTracker();
    t.recordTurn(1, [write(`${preamble}\n${body("alpha")}`)]);
    t.recordTurn(2, [write(`${preamble}\n${body("beta")}`)]);
    expect(t.recordTurn(3, [write(`${preamble}\n${body("gamma")}`)])).toBeNull();
  });

  it("exempts a file built up past the sampling cap", () => {
    // The growth exemption has to survive sampling: the shingle count of a
    // sampled file stops growing with it, and its tail sample slides as the
    // file is appended to, so neither can be what growth is read from.
    const upTo = (n: number) =>
      Array.from({ length: n }, (_, i) => `  $out[${i}] = stage_${i}($state, "row-${i}", ${i * 3 + 1});`).join("\n");
    expect(upTo(400).length).toBeGreaterThan(16 * 1024);
    const t = new FuzzyLoopTracker();
    t.recordTurn(1, [write(upTo(400))]);
    t.recordTurn(2, [write(upTo(420))]);
    expect(t.recordTurn(3, [write(upTo(440))])).toBeNull();
  });

  it("does not re-report a cluster on a turn that did not grow it", () => {
    // A detection a higher-priority verdict suppressed is not queued; it is
    // re-derived. It must therefore stop being due once the loop stops,
    // rather than repeating for every turn its members stay in the window.
    const t = new FuzzyLoopTracker({ window: 20 });
    t.recordTurn(1, [write(script(1))]);
    t.recordTurn(2, [write(script(2))]);
    expect(t.recordTurn(3, [write(script(3))])).toMatchObject({ count: 3 });
    // Nothing marked notified: the harness spoke about something else this
    // turn. The model then moves on to unrelated work.
    expect(t.recordTurn(4, [call("ShellSession", { command: script(4) })])).toBeNull();
    expect(t.recordTurn(5, [])).toBeNull();
  });

  it("is disabled by a streak of 0", () => {
    const t = new FuzzyLoopTracker({ streak: 0 });
    t.recordTurn(1, [write(script(1))]);
    t.recordTurn(2, [write(script(2))]);
    expect(t.recordTurn(3, [write(script(3))])).toBeNull();
  });

  it("speaks once, then once more at streak + 2, then not again", () => {
    const t = new FuzzyLoopTracker({ window: 20 });
    const fire = (turn: number) => {
      const d = t.recordTurn(turn, [write(script(turn))]);
      if (d) t.markNotified(d.clusterId);
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
});
