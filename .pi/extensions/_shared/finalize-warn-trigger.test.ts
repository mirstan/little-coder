import { describe, expect, it } from "vitest";
import {
  WARN_REMAINING,
  WARN_REMAINING_MS,
  finalizeWarnWouldFire,
} from "./finalize-warn-trigger.ts";

const NOW = 1_700_000_000_000;
const now = () => NOW;

describe("finalizeWarnWouldFire — turn trigger", () => {
  it("fires exactly at capForRun - WARN_REMAINING + 1", () => {
    // capForRun=40, WARN_REMAINING=5 -> fires at turn 36.
    expect(
      finalizeWarnWouldFire({ turnsThisRun: 36, capForRun: 40, deadlineForRun: 0, now }),
    ).toBe(true);
  });

  it("does not fire one turn before or after the trigger point", () => {
    expect(
      finalizeWarnWouldFire({ turnsThisRun: 35, capForRun: 40, deadlineForRun: 0, now }),
    ).toBe(false);
    expect(
      finalizeWarnWouldFire({ turnsThisRun: 37, capForRun: 40, deadlineForRun: 0, now }),
    ).toBe(false);
  });

  it("capForRun === 0 means no cap -> no turn trigger, ever", () => {
    for (const turnsThisRun of [1, 5, 100, WARN_REMAINING + 1]) {
      expect(
        finalizeWarnWouldFire({ turnsThisRun, capForRun: 0, deadlineForRun: 0, now }),
      ).toBe(false);
    }
  });

  it("a cap too small to give real headroom (cap <= WARN_REMAINING) never turn-triggers", () => {
    expect(
      finalizeWarnWouldFire({ turnsThisRun: 1, capForRun: WARN_REMAINING, deadlineForRun: 0, now }),
    ).toBe(false);
  });
});

describe("finalizeWarnWouldFire — wall-clock trigger", () => {
  it("fires just inside the WARN_REMAINING_MS threshold", () => {
    expect(
      finalizeWarnWouldFire({
        turnsThisRun: 1,
        capForRun: 0,
        deadlineForRun: NOW + WARN_REMAINING_MS - 1,
        now,
      }),
    ).toBe(true);
  });

  it("fires exactly at the threshold", () => {
    expect(
      finalizeWarnWouldFire({
        turnsThisRun: 1,
        capForRun: 0,
        deadlineForRun: NOW + WARN_REMAINING_MS,
        now,
      }),
    ).toBe(true);
  });

  it("does not fire just outside the threshold", () => {
    expect(
      finalizeWarnWouldFire({
        turnsThisRun: 1,
        capForRun: 0,
        deadlineForRun: NOW + WARN_REMAINING_MS + 1,
        now,
      }),
    ).toBe(false);
  });

  it("deadlineForRun === 0 means no deadline known -> no time trigger, ever", () => {
    expect(
      finalizeWarnWouldFire({ turnsThisRun: 1, capForRun: 0, deadlineForRun: 0, now }),
    ).toBe(false);
  });

  it("defaults `now` to Date.now when not injected", () => {
    const soon = Date.now() + 1_000; // well inside WARN_REMAINING_MS
    expect(finalizeWarnWouldFire({ turnsThisRun: 1, capForRun: 0, deadlineForRun: soon })).toBe(
      true,
    );
  });
});

describe("finalizeWarnWouldFire — either trigger is enough", () => {
  it("fires when both the turn and wall-clock triggers are true on the same turn", () => {
    expect(
      finalizeWarnWouldFire({
        turnsThisRun: 36,
        capForRun: 40,
        deadlineForRun: NOW + 1_000,
        now,
      }),
    ).toBe(true);
  });
});
