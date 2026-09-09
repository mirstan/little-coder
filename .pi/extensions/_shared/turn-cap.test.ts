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
});
