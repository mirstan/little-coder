import { readdirSync, readFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

// pi adopts only the payload a before_provider_request handler RETURNS, and
// chains handlers in extension load order, which bin/little-coder.mjs sets to
// readdirSync().sort() over .pi/extensions. The capture is the exact request
// only if every bundled extension that can rewrite the payload loads first.
const here = dirname(fileURLToPath(import.meta.url));
const extRoot = join(here, "..");
const self = "cache-reuse-compaction";

describe("load order", () => {
  it("loads after every bundled extension with a before_provider_request handler", () => {
    const rewriters = readdirSync(extRoot)
      .sort()
      .filter((name) => name !== self && existsSync(join(extRoot, name, "index.ts")))
      .filter((name) => /pi\.on\(\s*["']before_provider_request["']/.test(readFileSync(join(extRoot, name, "index.ts"), "utf-8")));
    expect(rewriters).toContain("benchmark-profiles");
    for (const name of rewriters) expect([name, self].sort()[0]).toBe(name);
  });

  it("the launcher still loads bundled extensions in sorted directory order", () => {
    const launcher = readFileSync(join(extRoot, "..", "..", "bin", "little-coder.mjs"), "utf-8");
    expect(launcher).toContain("for (const name of readdirSync(extDir).sort())");
  });
});
