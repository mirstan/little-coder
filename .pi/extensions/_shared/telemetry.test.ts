import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { emitTelemetry, telemetryEnabled } from "./telemetry.ts";

const ENABLE = "LITTLE_CODER_TELEMETRY";

function recorder() {
  const calls: Array<[string, unknown]> = [];
  const pi = {
    appendEntry(customType: string, data: unknown) {
      calls.push([customType, data]);
    },
  };
  return { calls, pi };
}

describe("emitTelemetry", () => {
  let saved: string | undefined;
  beforeEach(() => {
    saved = process.env[ENABLE];
    process.env[ENABLE] = "1";
  });
  afterEach(() => {
    if (saved === undefined) delete process.env[ENABLE];
    else process.env[ENABLE] = saved;
  });

  it("appends one lc-telemetry custom entry carrying v, kind and the data", () => {
    const { calls, pi } = recorder();
    expect(emitTelemetry(pi, "guard_abort", { trigger: "toolcall_cap", argChars: 32001 })).toBe(true);
    expect(calls).toEqual([
      ["lc-telemetry", { v: 1, kind: "guard_abort", trigger: "toolcall_cap", argChars: 32001 }],
    ]);
  });

  it("never lets the data overwrite v or kind", () => {
    const { calls, pi } = recorder();
    emitTelemetry(pi, "echo_block", { v: 99, kind: "forged", source: "length_stub" });
    expect(calls).toEqual([["lc-telemetry", { v: 1, kind: "echo_block", source: "length_stub" }]]);
  });

  it("is a silent no-op when pi has no appendEntry (unit-test harnesses)", () => {
    expect(emitTelemetry({}, "echo_block", {})).toBe(false);
    expect(emitTelemetry(undefined, "echo_block", {})).toBe(false);
    expect(emitTelemetry(null, "echo_block", {})).toBe(false);
  });

  it("swallows a throwing appendEntry, such as pi's stale-runtime error", () => {
    const pi = {
      appendEntry() {
        throw new Error("This extension ctx is stale after session replacement or reload.");
      },
    };
    expect(emitTelemetry(pi, "guard_abort", { trigger: "wall_clock" })).toBe(false);
  });

  it("is opt-in: emits only when LITTLE_CODER_TELEMETRY is exactly 1", () => {
    const { calls, pi } = recorder();
    delete process.env[ENABLE];
    expect(telemetryEnabled()).toBe(false);
    expect(emitTelemetry(pi, "echo_block", {})).toBe(false);
    process.env[ENABLE] = "true";
    expect(telemetryEnabled()).toBe(false);
    expect(calls).toEqual([]);
    process.env[ENABLE] = "1";
    expect(telemetryEnabled()).toBe(true);
    expect(emitTelemetry(pi, "echo_block", {})).toBe(true);
    expect(calls).toHaveLength(1);
  });
});
