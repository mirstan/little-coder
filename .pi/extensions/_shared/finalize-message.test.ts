import { describe, it, expect } from "vitest";
import { resolveFinalizeMessage } from "./finalize-message.ts";

describe("resolveFinalizeMessage", () => {
  it("returns the GAIA-style message for benchmark=gaia", () => {
    const msg = resolveFinalizeMessage("gaia");
    expect(msg).toMatch(/Answer: <value>/);
    expect(msg).toMatch(/EvidenceList/);
    expect(msg).not.toMatch(/ShellSession/);
    // Both container-file recovery pointers are meaningless for GAIA's
    // chat-reply scoring, so neither may leak here.
    expect(msg).not.toMatch(/\/tmp\/\.lc-snapshot/);
    expect(msg).not.toMatch(/\/tmp\/\.lc-initial/);
  });

  it("returns the Terminal-Bench-style message for benchmark=terminal_bench", () => {
    const msg = resolveFinalizeMessage("terminal_bench");
    expect(msg).toMatch(/ShellSession/);
    expect(msg).not.toMatch(/Answer:/);
    expect(msg).not.toMatch(/EvidenceList/);
    // Directive about saving immediately instead of continuing to verify --
    // targets the observed failure mode of having a working answer ready but
    // never persisting it before the deadline hit (e.g. raman-fitting).
    expect(msg).toMatch(/do not run further|stop investigating/i);
    // Pointer to the best-effort deadline snapshot little_coder_agent.py
    // stages under /tmp/.lc-snapshot, which recovers a working state that a
    // late mistake overwrites with nothing left to fall back on. TB-specific
    // only -- the snapshot is a container-file mechanism, meaningless for
    // GAIA's chat-reply scoring.
    expect(msg).toMatch(/\/tmp\/\.lc-snapshot/);
    // The start-of-trial copy of the task's ORIGINAL files, staged by
    // little_coder_agent.py before the model's first turn. Hedged as "may
    // exist" like the line above: this string is static and cannot see
    // whether a given trial's copy actually got staged.
    expect(msg).toMatch(/\/tmp\/\.lc-initial\/app\//);
    // Restraint clause, asserted verbatim rather than via a shared constant:
    // the failure it guards is a model near its deadline restoring pristine
    // originals over its own finished solution, and a test that imported the
    // wording could not notice that wording being softened.
    expect(msg).toContain(
      "restore from there only a file you believe you corrupted — never over your own completed solution",
    );
  });

  it("returns a generic fallback for undefined or any other benchmark", () => {
    for (const benchmark of [undefined, "some-future-benchmark"]) {
      const msg = resolveFinalizeMessage(benchmark);
      expect(msg).not.toMatch(/Answer:/);
      expect(msg).not.toMatch(/EvidenceList/);
      expect(msg).not.toMatch(/ShellSession/);
      expect(msg).not.toMatch(/\/tmp\/\.lc-snapshot/);
      expect(msg).not.toMatch(/\/tmp\/\.lc-initial/);
      expect(msg.length).toBeGreaterThan(0);
    }
  });
});
