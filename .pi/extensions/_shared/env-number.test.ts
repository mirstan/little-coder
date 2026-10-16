import { describe, it, expect, afterEach } from "vitest";
import { envNumber } from "./env-number.ts";

const VAR = "LITTLE_CODER_TEST_ENV_NUMBER";

describe("envNumber", () => {
  afterEach(() => {
    delete process.env[VAR];
  });

  it("falls back when the var is absent", () => {
    expect(envNumber(VAR, 0.25)).toBe(0.25);
  });

  it.each(["", "   ", "\t"])("falls back for empty/whitespace %j", (raw) => {
    process.env[VAR] = raw;
    expect(envNumber(VAR, 0.25)).toBe(0.25);
  });

  it.each(["abc", "0.25abc", "NaN", "Infinity", "-Infinity"])(
    "falls back for non-finite %j",
    (raw) => {
      process.env[VAR] = raw;
      expect(envNumber(VAR, 0.25)).toBe(0.25);
    },
  );

  // deadline.ts / turn-cap.ts's Number.isInteger convention would reject this
  // outright; a guard fraction has to survive it.
  it("accepts a fractional value", () => {
    process.env[VAR] = "0.25";
    expect(envNumber(VAR, 4)).toBe(0.25);
  });

  // Returned, not swallowed into the fallback: call sites test `<= 0`
  // themselves to mean "disabled".
  it.each([
    ["0", 0],
    ["-1", -1],
    ["-0.5", -0.5],
  ])("returns %j as-is rather than falling back", (raw, expected) => {
    process.env[VAR] = raw;
    expect(envNumber(VAR, 4)).toBe(expected);
  });

  it("accepts a plain positive integer", () => {
    process.env[VAR] = "900000";
    expect(envNumber(VAR, 4)).toBe(900000);
  });
});
