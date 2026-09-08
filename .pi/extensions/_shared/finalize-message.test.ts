import { describe, it, expect } from "vitest";
import { resolveFinalizeMessage } from "./finalize-message.ts";

describe("resolveFinalizeMessage", () => {
  it("returns the GAIA-style message for benchmark=gaia", () => {
    const msg = resolveFinalizeMessage("gaia");
    expect(msg).toMatch(/Answer: <value>/);
    expect(msg).toMatch(/EvidenceList/);
    expect(msg).not.toMatch(/ShellSession/);
  });

  it("returns the Terminal-Bench-style message for benchmark=terminal_bench", () => {
    const msg = resolveFinalizeMessage("terminal_bench");
    expect(msg).toMatch(/ShellSession/);
    expect(msg).not.toMatch(/Answer:/);
    expect(msg).not.toMatch(/EvidenceList/);
  });

  it("returns a generic fallback for undefined or any other benchmark", () => {
    for (const benchmark of [undefined, "some-future-benchmark"]) {
      const msg = resolveFinalizeMessage(benchmark);
      expect(msg).not.toMatch(/Answer:/);
      expect(msg).not.toMatch(/EvidenceList/);
      expect(msg).not.toMatch(/ShellSession/);
      expect(msg.length).toBeGreaterThan(0);
    }
  });
});
