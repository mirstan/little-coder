import { afterEach, describe, expect, it } from "vitest";
import {
  PROGRESS_MILESTONES,
  budgetFractionUsed,
  resolveBudgetStartEpochMs,
} from "./budget-progress.ts";

const NOW = 1_700_000_000_000;
const now = () => NOW;
const HOUR = 60 * 60 * 1000;

afterEach(() => {
  delete process.env.LITTLE_CODER_BUDGET_START_EPOCH_MS;
});

describe("PROGRESS_MILESTONES", () => {
  it("is ascending, so a scan for the largest crossed one ends on it", () => {
    const sorted = [...PROGRESS_MILESTONES].sort((a, b) => a - b);
    expect([...PROGRESS_MILESTONES]).toEqual(sorted);
  });

  it("stays strictly inside the trial", () => {
    for (const m of PROGRESS_MILESTONES) {
      expect(m).toBeGreaterThan(0);
      expect(m).toBeLessThan(1);
    }
  });
});

describe("resolveBudgetStartEpochMs", () => {
  it("reads a set env var", () => {
    process.env.LITTLE_CODER_BUDGET_START_EPOCH_MS = String(NOW);
    expect(resolveBudgetStartEpochMs({})).toBe(NOW);
  });

  it("is 0 when nothing published a start instant", () => {
    expect(resolveBudgetStartEpochMs({})).toBe(0);
    expect(resolveBudgetStartEpochMs(undefined)).toBe(0);
  });

  it("treats an empty-but-exported var as unset, like deadline.ts", () => {
    process.env.LITTLE_CODER_BUDGET_START_EPOCH_MS = "";
    expect(
      resolveBudgetStartEpochMs({ systemPromptOptions: { littleCoder: { budgetStartEpochMs: NOW } } }),
    ).toBe(NOW);
    process.env.LITTLE_CODER_BUDGET_START_EPOCH_MS = "   ";
    expect(resolveBudgetStartEpochMs({})).toBe(0);
  });

  it("rejects a malformed value outright rather than taking its numeric prefix", () => {
    for (const raw of ["1700000000000garbage", "garbage", "1.5", "-1", "0"]) {
      process.env.LITTLE_CODER_BUDGET_START_EPOCH_MS = raw;
      expect(resolveBudgetStartEpochMs({})).toBe(0);
    }
  });

  it("lets a set env var win over the event override", () => {
    process.env.LITTLE_CODER_BUDGET_START_EPOCH_MS = String(NOW);
    expect(
      resolveBudgetStartEpochMs({
        systemPromptOptions: { littleCoder: { budgetStartEpochMs: NOW + HOUR } },
      }),
    ).toBe(NOW);
  });

  it("falls back to the event override only when the env var is absent", () => {
    expect(
      resolveBudgetStartEpochMs({
        systemPromptOptions: { littleCoder: { budgetStartEpochMs: NOW } },
      }),
    ).toBe(NOW);
    expect(
      resolveBudgetStartEpochMs({ systemPromptOptions: { littleCoder: { budgetStartEpochMs: -5 } } }),
    ).toBe(0);
  });
});

describe("budgetFractionUsed", () => {
  it("is the elapsed share of [start, deadline]", () => {
    expect(
      budgetFractionUsed({ startForRun: NOW - HOUR, deadlineForRun: NOW + HOUR, now }),
    ).toBe(0.5);
    expect(
      budgetFractionUsed({ startForRun: NOW - 3 * HOUR, deadlineForRun: NOW + HOUR, now }),
    ).toBe(0.75);
  });

  it("is 0 at the start instant and 1 at the deadline", () => {
    expect(budgetFractionUsed({ startForRun: NOW, deadlineForRun: NOW + HOUR, now })).toBe(0);
    expect(budgetFractionUsed({ startForRun: NOW - HOUR, deadlineForRun: NOW, now })).toBe(1);
  });

  it("is not clamped past the deadline", () => {
    expect(
      budgetFractionUsed({ startForRun: NOW - 2 * HOUR, deadlineForRun: NOW - HOUR, now }),
    ).toBe(2);
  });

  it("is undefined when either end is unknown", () => {
    expect(budgetFractionUsed({ startForRun: 0, deadlineForRun: NOW + HOUR, now })).toBeUndefined();
    expect(budgetFractionUsed({ startForRun: NOW, deadlineForRun: 0, now })).toBeUndefined();
    expect(budgetFractionUsed({ startForRun: 0, deadlineForRun: 0, now })).toBeUndefined();
  });

  it("is undefined for a non-positive interval rather than dividing by it", () => {
    expect(budgetFractionUsed({ startForRun: NOW, deadlineForRun: NOW, now })).toBeUndefined();
    expect(
      budgetFractionUsed({ startForRun: NOW + HOUR, deadlineForRun: NOW, now }),
    ).toBeUndefined();
  });

  it("defaults its clock to Date.now", () => {
    const real = Date.now;
    Date.now = () => NOW;
    try {
      expect(budgetFractionUsed({ startForRun: NOW - HOUR, deadlineForRun: NOW + HOUR })).toBe(0.5);
    } finally {
      Date.now = real;
    }
  });
});
