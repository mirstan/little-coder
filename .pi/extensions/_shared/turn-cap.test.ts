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

  it("systemPromptOptions.littleCoder.maxTurns wins over the env var", () => {
    process.env.LITTLE_CODER_MAX_TURNS = "40";
    const event = { systemPromptOptions: { littleCoder: { maxTurns: 10 } } };
    expect(resolveTurnCap(event)).toBe(10);
  });

  it("falls back to the env var when maxTurns is not a positive number", () => {
    process.env.LITTLE_CODER_MAX_TURNS = "40";
    expect(resolveTurnCap({ systemPromptOptions: { littleCoder: { maxTurns: 0 } } })).toBe(40);
    expect(resolveTurnCap({ systemPromptOptions: { littleCoder: {} } })).toBe(40);
    expect(resolveTurnCap({ systemPromptOptions: {} })).toBe(40);
  });
});
