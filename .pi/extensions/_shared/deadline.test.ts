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

  it("an explicitly-set env var wins over systemPromptOptions.littleCoder.deadlineEpochMs", () => {
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = "1700000000000";
    const event = { systemPromptOptions: { littleCoder: { deadlineEpochMs: 1234567890 } } };
    expect(resolveDeadlineEpochMs(event)).toBe(1700000000000);
  });

  it("falls back to systemPromptOptions.littleCoder.deadlineEpochMs when the env var is absent", () => {
    delete process.env.LITTLE_CODER_DEADLINE_EPOCH_MS;
    expect(
      resolveDeadlineEpochMs({ systemPromptOptions: { littleCoder: { deadlineEpochMs: 1234567890 } } }),
    ).toBe(1234567890);
    expect(resolveDeadlineEpochMs({ systemPromptOptions: { littleCoder: { deadlineEpochMs: 0 } } })).toBe(0);
    expect(resolveDeadlineEpochMs({ systemPromptOptions: { littleCoder: {} } })).toBe(0);
    expect(resolveDeadlineEpochMs({ systemPromptOptions: {} })).toBe(0);
  });

  // An empty-but-set env var (`export LITTLE_CODER_DEADLINE_EPOCH_MS=` in a
  // wrapper script, a CI matrix that exports unset variables, `env VAR=
  // cmd`) must be treated as UNSET, not as an authoritative "0" --
  // Number("") is 0, which would otherwise silently invert the precedence.
  describe("empty-but-set env var", () => {
    it("\"\" with an event override present falls back to the EVENT value, not 0", () => {
      process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = "";
      const event = { systemPromptOptions: { littleCoder: { deadlineEpochMs: 1234567890 } } };
      expect(resolveDeadlineEpochMs(event)).toBe(1234567890);
    });

    it("\"\" with no event override falls back to 0", () => {
      process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = "";
      expect(resolveDeadlineEpochMs({})).toBe(0);
      expect(resolveDeadlineEpochMs({ systemPromptOptions: {} })).toBe(0);
    });

    it("whitespace-only (\"  \") with an event override present falls back to the EVENT value", () => {
      process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = "  ";
      const event = { systemPromptOptions: { littleCoder: { deadlineEpochMs: 1234567890 } } };
      expect(resolveDeadlineEpochMs(event)).toBe(1234567890);
    });

    it("whitespace-only (\"  \") with no event override falls back to 0", () => {
      process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = "  ";
      expect(resolveDeadlineEpochMs({})).toBe(0);
    });

    it("a literal \"0\" stays authoritative regardless of an event override", () => {
      process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = "0";
      expect(
        resolveDeadlineEpochMs({ systemPromptOptions: { littleCoder: { deadlineEpochMs: 1234567890 } } }),
      ).toBe(0);
      expect(resolveDeadlineEpochMs({})).toBe(0);
    });

    it("undefined (unset) with an event override present falls back to the event value", () => {
      delete process.env.LITTLE_CODER_DEADLINE_EPOCH_MS;
      const event = { systemPromptOptions: { littleCoder: { deadlineEpochMs: 1234567890 } } };
      expect(resolveDeadlineEpochMs(event)).toBe(1234567890);
    });

    it("a valid positive value stays authoritative regardless of an event override", () => {
      process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = "1700000000000";
      expect(
        resolveDeadlineEpochMs({ systemPromptOptions: { littleCoder: { deadlineEpochMs: 1234567890 } } }),
      ).toBe(1700000000000);
      expect(resolveDeadlineEpochMs({})).toBe(1700000000000);
    });
  });
});
