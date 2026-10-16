import { describe, it, expect } from "vitest";
import { readFileSync, existsSync, readdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { parseSkillFile } from "../skill-inject/frontmatter.ts";
import { scoreEntry, MIN_SCORE_THRESHOLD, type KnowledgeEntry } from "./index.ts";

// Exercise the REAL scoreEntry (imported above), not a hand-copied duplicate —
// a regression in the production scorer must fail this test. Only `keywords`
// affects the score, so the other KnowledgeEntry fields are inert here.
function entry(keywords: string[]): KnowledgeEntry {
  return { topic: "t", body: "b", tokenCost: 0, keywords, requiresTools: [] };
}

describe("knowledge entry scoring", () => {
  it("scores single word matches at 1.0 each", () => {
    expect(scoreEntry("find the bucket", entry(["bucket"]))).toBe(1.0);
    expect(scoreEntry("find the bucket and pour", entry(["bucket", "pour"]))).toBe(2.0);
  });

  it("scores bigram/phrase matches at 2.0 each", () => {
    expect(scoreEntry("minimum moves to solve", entry(["minimum moves"]))).toBe(2.0);
    expect(scoreEntry("state space search", entry(["state space"]))).toBe(2.0);
  });

  it("combines word + bigram scores", () => {
    const kw = ["bucket", "minimum moves", "pour"];
    // "bucket" word (1.0) + "minimum moves" phrase (2.0) + "pour" word (1.0) = 4.0
    expect(scoreEntry("bucket pouring problem with minimum moves and pour", entry(kw))).toBe(4.0);
  });

  it("does not match partial words", () => {
    // 'bucket' shouldn't match 'buckets' because the scorer tokenizes on whitespace
    expect(scoreEntry("many buckets here", entry(["bucket"]))).toBe(0);
  });

  it("scores 0 for an entry with no keywords", () => {
    expect(scoreEntry("anything at all", entry([]))).toBe(0);
  });

  it("threshold at 2.0 requires at least two signals", () => {
    // The extension's MIN_SCORE_THRESHOLD = 2.0 means one word isn't enough
    expect(MIN_SCORE_THRESHOLD).toBe(2.0);
    expect(scoreEntry("find bucket", entry(["bucket", "pour"]))).toBeLessThan(MIN_SCORE_THRESHOLD);
    expect(scoreEntry("bucket pour together", entry(["bucket", "pour"]))).toBeGreaterThanOrEqual(MIN_SCORE_THRESHOLD);
  });
});

describe("knowledge directory loads from repo", () => {
  const here = dirname(fileURLToPath(import.meta.url));
  const kDir = join(here, "..", "..", "..", "skills", "knowledge");
  const pDir = join(here, "..", "..", "..", "skills", "protocols");

  it("knowledge dir has 14 files", () => {
    expect(existsSync(kDir)).toBe(true);
    expect(readdirSync(kDir).filter((f) => f.endsWith(".md")).length).toBe(14);
  });

  it("protocols dir has 3 files", () => {
    expect(existsSync(pDir)).toBe(true);
    expect(readdirSync(pDir).filter((f) => f.endsWith(".md")).length).toBe(3);
  });

  it("every knowledge entry has topic + keywords in frontmatter", () => {
    const files = readdirSync(kDir).filter((f) => f.endsWith(".md"));
    for (const file of files) {
      const parsed = parseSkillFile(readFileSync(join(kDir, file), "utf-8"));
      expect(parsed, `${file} should parse`).not.toBeNull();
      expect(typeof parsed!.frontmatter.topic).toBe("string");
      expect(Array.isArray(parsed!.frontmatter.keywords), `${file} keywords`).toBe(true);
    }
  });

  it("workspace_docs declares requires_tools", () => {
    const parsed = parseSkillFile(readFileSync(join(kDir, "workspace_docs.md"), "utf-8"));
    expect(parsed!.frontmatter.requires_tools).toEqual(["read", "glob"]);
  });
});

// A run scoring 0.617 against a 0.62 threshold sat inside its own ~0.019 spread.
describe("stochastic-training-variance entry", () => {
  const here = dirname(fileURLToPath(import.meta.url));
  const kDir = join(here, "..", "..", "..", "skills", "knowledge");
  const parsed = parseSkillFile(
    readFileSync(join(kDir, "stochastic_training_variance.md"), "utf-8"),
  );
  const keywords = parsed!.frontmatter.keywords as string[];

  it("fires on the real train-fasttext instruction text", () => {
    const prompt =
      "Please train a fasttext model on the yelp data in the data/ folder. " +
      "The final model size needs to be less than 150MB but get at least " +
      "0.62 accuracy on a private test set that comes from the same yelp " +
      "review distribution. The model should be saved as /app/model.bin";
    expect(scoreEntry(prompt, entry(keywords))).toBeGreaterThanOrEqual(MIN_SCORE_THRESHOLD);
  });

  it("fires on a different train-to-threshold prompt that never mentions fastText", () => {
    const prompt = "train an XGBoost classifier to reach at least 0.85 F1 on the validation set";
    expect(scoreEntry(prompt, entry(keywords))).toBeGreaterThanOrEqual(MIN_SCORE_THRESHOLD);
  });

  it("does not fire on unrelated coding prompts with no training/threshold shape", () => {
    const gpt2Codegolf =
      "I have downloaded the gpt-2 weights stored as a TF .ckpt. Write me a " +
      "dependency-free C file that samples from the model with arg-max " +
      "sampling. Call your program /app/gpt2.c, I will compile with gcc -O3 " +
      "-lm. It should read the .ckpt and the .bpe file. Your c program must " +
      'be <5000 bytes. I will run it /app/a.out gpt2-124M.ckpt vocab.bpe ' +
      '"[input string here]" and you should continue the output under ' +
      "whatever GPT-2 would print for the next 20 tokens.";
    expect(scoreEntry(gpt2Codegolf, entry(keywords))).toBeLessThan(MIN_SCORE_THRESHOLD);
  });

  // the test above proves only zero-overlap silence;
  // it cannot detect the false-positive boundary the old 13-bare-word keyword
  // list actually crossed. These three prompts describe no training run and
  // no metric threshold, but scored >= MIN_SCORE_THRESHOLD under the old list
  // purely from incidental bare-word pairs (verified by hand against the real
  // scorer: "deterministic"+"reproducible", "validation"+"threshold",
  // "classifier"+"training"). Unlike the gpt2 test, this one can actually
  // fail if a future keyword edit reopens the boundary.
  //
  // Four more prompts below pin the phrase-keyword boundary the
  // iteration-1 fix itself reopened -- `test set` and
  // `at least 0` were still in the keyword list and, being phrase keywords,
  // fired the entry alone (2.0 >= MIN_SCORE_THRESHOLD) on ANY prompt
  // containing that substring, ML or not (verified by hand against the real
  // scorer before the fix: all four scored 2.0). Both keywords are now
  // removed from the frontmatter; these four pin the boundary against a
  // future edit reintroducing either one.
  it("does not fire on prompts that share old keywords but describe no training run or metric threshold", () => {
    const falsePositives = [
      "make the pipeline deterministic and reproducible",
      "add validation and a threshold check",
      "Refactor the spam classifier module for readability; do not change training behaviour.",
      "Run the test setup script in /app and make sure it exits cleanly.",
      "Restore the sqlite db and verify the integrity check passes on the test set.",
      "Write a compressor. The compression ratio must be at least 0.8 on the provided corpus.",
      "I need a build script; the coverage gate is at least 0.9 line coverage.",
    ];
    for (const prompt of falsePositives) {
      expect(scoreEntry(prompt, entry(keywords)), prompt).toBeLessThan(MIN_SCORE_THRESHOLD);
    }
  });

  // iterations 1-2 fixed 2 specific
  // phrase keywords ("test set", "at least 0") caught by name, but left the
  // defect CLASS open -- any remaining phrase keyword still fires the entry
  // alone (2.0 >= MIN_SCORE_THRESHOLD) regardless of ML content, since a
  // phrase match alone equals the whole threshold. Independently verified
  // against the real scorer: "accuracy threshold", "training run", and
  // "random seed" all fired alone on these three unrelated prompts before
  // this fix. The systemic fix removes every remaining phrase keyword from
  // this entry's frontmatter (with "random seed" replaced by the plain word
  // "seed", which requires a second co-occurring signal to reach threshold)
  // rather than patching individual caught phrases -- these three pin the
  // whole class, not just the caught instances.
  it("does not fire on prompts containing a former phrase keyword's constituent words but no training/threshold content", () => {
    const falsePositives = [
      "use a random seed for the maze generator",
      "schedule the nightly training run for the CI pipeline image build",
      "what's the accuracy threshold for this linter's confidence score",
    ];
    for (const prompt of falsePositives) {
      expect(scoreEntry(prompt, entry(keywords)), prompt).toBeLessThan(MIN_SCORE_THRESHOLD);
    }
  });

  it("still fires when seed/reproducibility and another entry keyword co-occur", () => {
    const prompt = "I need to set a seed for my training script but accuracy still varies each run";
    expect(scoreEntry(prompt, entry(keywords))).toBeGreaterThanOrEqual(MIN_SCORE_THRESHOLD);
  });
});

// Re-derives index.ts:117-131's score-sort + greedy-budget-fit against the
// real entry files. Two deliberate gaps: index.ts:119's requires_tools filter
// is skipped (all entries loaded with requiresTools: []), and BUDGET mirrors
// the 200 default, which lc.knowledgeTokenBudget can override. PER_ENTRY_CAP
// (index.ts:31) is not exported, so it is a literal here.
describe("knowledge-inject real selection/budget path", () => {
  const here = dirname(fileURLToPath(import.meta.url));
  const kDir = join(here, "..", "..", "..", "skills", "knowledge");
  const pDir = join(here, "..", "..", "..", "skills", "protocols");
  const PER_ENTRY_CAP = 150;
  const BUDGET = 200;

  function loadRealEntries(dir: string): KnowledgeEntry[] {
    const out: KnowledgeEntry[] = [];
    for (const file of readdirSync(dir)) {
      if (!file.endsWith(".md")) continue;
      const parsed = parseSkillFile(readFileSync(join(dir, file), "utf-8"));
      if (!parsed) continue;
      const fm = parsed.frontmatter as Record<string, unknown>;
      const topic =
        (typeof fm.topic === "string" ? fm.topic : "") ||
        (typeof fm.name === "string" ? fm.name : "");
      if (!topic || !parsed.body) continue;
      let cost = typeof fm.token_cost === "number" ? fm.token_cost : 150;
      if (cost > PER_ENTRY_CAP) cost = PER_ENTRY_CAP;
      const keywords = Array.isArray(fm.keywords)
        ? (fm.keywords as string[]).map((k) => k.toLowerCase())
        : [];
      out.push({ topic, body: parsed.body, tokenCost: cost, keywords, requiresTools: [] });
    }
    return out;
  }

  it("fits the corrected stochastic-training-variance entry under budget on the real train-fasttext prompt", () => {
    const prompt =
      "Please train a fasttext model on the yelp data in the data/ folder. " +
      "The final model size needs to be less than 150MB but get at least " +
      "0.62 accuracy on a private test set that comes from the same yelp " +
      "review distribution. The model should be saved as /app/model.bin";

    const entries = [...loadRealEntries(kDir), ...loadRealEntries(pDir)];
    const stochastic = entries.find(
      (e) => e.topic === "Stochastic Training Variance Near a Threshold",
    );
    expect(stochastic).toBeDefined();
    // truth 1: token_cost was 90, the cheapest tier, for the largest body in
    // skills/knowledge. It now declares PER_ENTRY_CAP (150), the most the loader
    // honours.
    expect(stochastic!.tokenCost).toBe(150);

    // Mirror index.ts:117-124's scoring/filter/sort.
    const scored: Array<{ score: number; entry: KnowledgeEntry }> = [];
    for (const e of entries) {
      const s = scoreEntry(prompt, e);
      if (s >= MIN_SCORE_THRESHOLD) scored.push({ score: s, entry: e });
    }
    scored.sort((a, b) => b.score - a.score);

    // Mirror index.ts:126-131's greedy budget-fit loop exactly.
    const selected: string[] = [];
    let used = 0;
    for (const { entry } of scored) {
      if (used + entry.tokenCost > BUDGET) continue;
      selected.push(entry.topic);
      used += entry.tokenCost;
    }

    // Computed, not assumed: on this prompt the corrected entry's real score
    // (train + fasttext + accuracy = 3.0, post the removal of
    // the `test set` and `at least 0` phrase keywords that used to inflate
    // this same score to 7.0) is still >= MIN_SCORE_THRESHOLD and high
    // enough in the sort that it consumes enough of the 200-token budget to
    // keep Workspace Documentation out, even at the honest cost of 150 --
    // the original bug (a false-cheap declared cost) is fixed, but the
    // greedy budget-fit ALGORITHM is unchanged and out of scope for this fix.
    // This pins the corrected-cost outcome so a future
    // edit to either file re-computes it rather than silently reintroducing
    // (or silently "fixing") the eviction.
    expect(selected).toContain("Stochastic Training Variance Near a Threshold");
    expect(selected).not.toContain("Workspace Documentation");
    expect(used).toBe(150);
  });
});
