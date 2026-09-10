import { describe, it, expect, afterEach } from "vitest";
import { resolveTurnCap } from "./turn-cap.ts";

describe("resolveTurnCap", () => {
  afterEach(() => {
    delete process.env.LITTLE_CODER_MAX_TURNS;
  });

  it("returns 0 when neither systemPromptOptions nor the env var set a cap", () => {
    expect(resolveTurnCap({})).toBe(0);
    expect(resolveTurnCap(undefined)).toBe(0);
  });

  it("uses the env var when set to a positive integer", () => {
    process.env.LITTLE_CODER_MAX_TURNS = "40";
    expect(resolveTurnCap({})).toBe(40);
  });

  it("ignores a non-positive or non-numeric env var", () => {
    process.env.LITTLE_CODER_MAX_TURNS = "0";
    expect(resolveTurnCap({})).toBe(0);
    process.env.LITTLE_CODER_MAX_TURNS = "-5";
    expect(resolveTurnCap({})).toBe(0);
    process.env.LITTLE_CODER_MAX_TURNS = "not-a-number";
    expect(resolveTurnCap({})).toBe(0);
  });

  it("an explicitly-set env var wins over systemPromptOptions.littleCoder.maxTurns", () => {
    process.env.LITTLE_CODER_MAX_TURNS = "40";
    const event = { systemPromptOptions: { littleCoder: { maxTurns: 10 } } };
    expect(resolveTurnCap(event)).toBe(40);
  });

  it("an explicit env var of \"0\" wins over a positive profile maxTurns", () => {
    // Plan 5 / Codex finding [high]: little_coder_agent.py passes max_turns=0
    // (no cap) but a model's benchmark_overrides.terminal_bench.max_turns
    // profile still published 40 -- the explicit "0" must win so the harness
    // kwarg is the deliberate per-run authority, not the profile.
    process.env.LITTLE_CODER_MAX_TURNS = "0";
    const event = { systemPromptOptions: { littleCoder: { maxTurns: 40 } } };
    expect(resolveTurnCap(event)).toBe(0);
  });

  it("falls back to systemPromptOptions.littleCoder.maxTurns when the env var is absent", () => {
    delete process.env.LITTLE_CODER_MAX_TURNS;
    expect(resolveTurnCap({ systemPromptOptions: { littleCoder: { maxTurns: 40 } } })).toBe(40);
    expect(resolveTurnCap({ systemPromptOptions: { littleCoder: { maxTurns: 0 } } })).toBe(0);
    expect(resolveTurnCap({ systemPromptOptions: { littleCoder: {} } })).toBe(0);
    expect(resolveTurnCap({ systemPromptOptions: {} })).toBe(0);
  });

  // An empty-but-set env var (`export LITTLE_CODER_MAX_TURNS=` in a wrapper
  // script, a CI matrix that exports unset variables, `env VAR= cmd`) must
  // be treated as UNSET, not as an authoritative "0" -- Number("") is 0,
  // and treating "" as set would silently invert the precedence this fix
  // exists to enforce. Each case is crossed with "profile override present
  // / absent" since the bug is specifically about which one wins.
  describe("empty-but-set env var (Priority 3 fix)", () => {
    it("\"\" with a profile override present falls back to the PROFILE value, not 0", () => {
      process.env.LITTLE_CODER_MAX_TURNS = "";
      const event = { systemPromptOptions: { littleCoder: { maxTurns: 40 } } };
      // This is the exact inversion being fixed: "" must NOT win as an
      // authoritative 0 over a positive profile override.
      expect(resolveTurnCap(event)).toBe(40);
    });

    it("\"\" with no profile override falls back to 0", () => {
      process.env.LITTLE_CODER_MAX_TURNS = "";
      expect(resolveTurnCap({})).toBe(0);
      expect(resolveTurnCap({ systemPromptOptions: {} })).toBe(0);
    });

    it("whitespace-only (\"  \") with a profile override present falls back to the PROFILE value", () => {
      process.env.LITTLE_CODER_MAX_TURNS = "  ";
      const event = { systemPromptOptions: { littleCoder: { maxTurns: 40 } } };
      expect(resolveTurnCap(event)).toBe(40);
    });

    it("whitespace-only (\"  \") with no profile override falls back to 0", () => {
      process.env.LITTLE_CODER_MAX_TURNS = "  ";
      expect(resolveTurnCap({})).toBe(0);
    });

    it("a literal \"0\" stays authoritative regardless of a profile override", () => {
      process.env.LITTLE_CODER_MAX_TURNS = "0";
      expect(resolveTurnCap({ systemPromptOptions: { littleCoder: { maxTurns: 40 } } })).toBe(0);
      expect(resolveTurnCap({})).toBe(0);
    });

    it("undefined (unset) with a profile override present falls back to the profile value", () => {
      delete process.env.LITTLE_CODER_MAX_TURNS;
      const event = { systemPromptOptions: { littleCoder: { maxTurns: 40 } } };
      expect(resolveTurnCap(event)).toBe(40);
    });

    it("a valid positive value stays authoritative regardless of a profile override", () => {
      process.env.LITTLE_CODER_MAX_TURNS = "40";
      expect(resolveTurnCap({ systemPromptOptions: { littleCoder: { maxTurns: 10 } } })).toBe(40);
      expect(resolveTurnCap({})).toBe(40);
    });
  });
});
