import { describe, it, expect, afterEach } from "vitest";
import { resolveDeadlineEpochMs } from "./deadline.ts";

describe("resolveDeadlineEpochMs", () => {
  afterEach(() => {
    delete process.env.LITTLE_CODER_DEADLINE_EPOCH_MS;
  });

  it("returns 0 when neither systemPromptOptions nor the env var set a deadline", () => {
    expect(resolveDeadlineEpochMs({})).toBe(0);
    expect(resolveDeadlineEpochMs(undefined)).toBe(0);
  });

  it("uses the env var when set to a positive integer", () => {
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = "1700000000000";
    expect(resolveDeadlineEpochMs({})).toBe(1700000000000);
  });

  it("ignores a non-positive or non-numeric env var", () => {
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = "0";
    expect(resolveDeadlineEpochMs({})).toBe(0);
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = "-5";
    expect(resolveDeadlineEpochMs({})).toBe(0);
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = "not-a-number";
    expect(resolveDeadlineEpochMs({})).toBe(0);
  });

  it("systemPromptOptions.littleCoder.deadlineEpochMs wins over the env var", () => {
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = "1700000000000";
    const event = { systemPromptOptions: { littleCoder: { deadlineEpochMs: 1234567890 } } };
    expect(resolveDeadlineEpochMs(event)).toBe(1234567890);
  });

  it("falls back to the env var when deadlineEpochMs is not a positive number", () => {
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = "1700000000000";
    expect(
      resolveDeadlineEpochMs({ systemPromptOptions: { littleCoder: { deadlineEpochMs: 0 } } }),
    ).toBe(1700000000000);
    expect(resolveDeadlineEpochMs({ systemPromptOptions: { littleCoder: {} } })).toBe(
      1700000000000,
    );
    expect(resolveDeadlineEpochMs({ systemPromptOptions: {} })).toBe(1700000000000);
  });
});
