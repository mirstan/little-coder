import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { describe, expect, it } from "vitest";
import {
  computeFileLists,
  formatFileOperations,
  mergeSplitTurn,
  PI_NO_PRIOR_HISTORY,
  PI_SUMMARIZATION_PROMPT,
  PI_SUMMARIZATION_SYSTEM_PROMPT,
  PI_TURN_PREFIX_SUMMARIZATION_PROMPT,
  PI_UPDATE_SUMMARIZATION_PROMPT,
} from "./pi-compat.ts";

// The copies are only as good as their match with the pi actually installed.
const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const compactionDir = join(repoRoot, "node_modules", "@earendil-works", "pi-coding-agent", "dist", "core", "compaction");
const compactionJs = readFileSync(join(compactionDir, "compaction.js"), "utf-8");
const utilsJs = readFileSync(join(compactionDir, "utils.js"), "utf-8");

describe("pi-compat copies match the installed pi", () => {
  it.each([
    ["SUMMARIZATION_PROMPT", PI_SUMMARIZATION_PROMPT, compactionJs],
    ["UPDATE_SUMMARIZATION_PROMPT", PI_UPDATE_SUMMARIZATION_PROMPT, compactionJs],
    ["TURN_PREFIX_SUMMARIZATION_PROMPT", PI_TURN_PREFIX_SUMMARIZATION_PROMPT, compactionJs],
    ["SUMMARIZATION_SYSTEM_PROMPT", PI_SUMMARIZATION_SYSTEM_PROMPT, utilsJs],
  ])("%s", (name, copy, source) => {
    expect(source).toContain(`const ${name} = \`${copy}\`;`);
  });

  it("split-turn merge and the empty-history text", () => {
    expect(compactionJs).toContain("`${historyText}\\n\\n---\\n\\n**Turn Context (split turn):**\\n\\n${turnPrefixResult.text}`");
    expect(compactionJs).toContain(`let historyText = "${PI_NO_PRIOR_HISTORY}";`);
    expect(mergeSplitTurn("H", "P")).toBe("H\n\n---\n\n**Turn Context (split turn):**\n\nP");
  });

  it("file-list helpers behave like pi's", async () => {
    const pi = await import(pathToFileURL(join(compactionDir, "utils.js")).href);
    const ops = () => ({
      read: new Set(["/b", "/a", "/edited"]),
      written: new Set(["/w"]),
      edited: new Set(["/edited"]),
    });
    expect(computeFileLists(ops())).toEqual(pi.computeFileLists(ops()));
    const { readFiles, modifiedFiles } = computeFileLists(ops());
    expect(formatFileOperations(readFiles, modifiedFiles)).toBe(pi.formatFileOperations(readFiles, modifiedFiles));
    expect(formatFileOperations([], [])).toBe(pi.formatFileOperations([], []));
  });
});
