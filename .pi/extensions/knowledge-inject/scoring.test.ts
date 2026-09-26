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

// Motivating trial: train-fasttext__yewN65G (benchmarks/harbor_runs/
// 2026-09-24__21-02-23/) scored reward 0.0 at 0.617 accuracy vs a 0.62
// threshold, inside the model's own measured ~0.019 CV/holdout spread.
// This entry is scoped to the general train-to-threshold shape, not
// fastText specifically -- these tests exercise the REAL frontmatter
// keywords against the REAL scorer, not a hand-copied keyword list.
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
    // Proves this is a general train-to-threshold entry, not a fastText-only
    // trigger with extra words attached.
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
});
