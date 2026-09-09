import { describe, it, expect } from "vitest";
import { resolveFinalizeMessage } from "./finalize-message.ts";

describe("resolveFinalizeMessage", () => {
  it("returns the GAIA-style message for benchmark=gaia", () => {
    const msg = resolveFinalizeMessage("gaia");
    expect(msg).toMatch(/Answer: <value>/);
    expect(msg).toMatch(/EvidenceList/);
    expect(msg).not.toMatch(/ShellSession/);
    // The /tmp/.lc-snapshot pointer is a container-file recovery mechanism;
    // meaningless for GAIA's chat-reply scoring, so it must not leak here.
    expect(msg).not.toMatch(/\/tmp\/\.lc-snapshot/);
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
    // Plan 4b (shell-proxy-snapshot): pointer to the best-effort deadline
    // snapshot little_coder_agent.py stages under /tmp/.lc-snapshot, for the
    // overfull-hbox failure mode (a late mistake destroying earlier good
    // state with nothing to recover it). TB-specific only -- the snapshot is
    // a container-file mechanism, meaningless for GAIA's chat-reply scoring.
    expect(msg).toMatch(/\/tmp\/\.lc-snapshot/);
  });

  it("returns a generic fallback for undefined or any other benchmark", () => {
    for (const benchmark of [undefined, "some-future-benchmark"]) {
      const msg = resolveFinalizeMessage(benchmark);
      expect(msg).not.toMatch(/Answer:/);
      expect(msg).not.toMatch(/EvidenceList/);
      expect(msg).not.toMatch(/ShellSession/);
      expect(msg).not.toMatch(/\/tmp\/\.lc-snapshot/);
      expect(msg.length).toBeGreaterThan(0);
    }
  });
});
