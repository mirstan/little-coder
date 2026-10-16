import { describe, it, expect, afterEach } from "vitest";
import { pinEnv } from "./env-pin.ts";

const A = "LITTLE_CODER_ENV_PIN_TEST_A";
const B = "LITTLE_CODER_ENV_PIN_TEST_B";
afterEach(() => {
  delete process.env[A];
  delete process.env[B];
});

describe("pinEnv", () => {
  it("removes the named variables", () => {
    process.env[A] = "7";
    const pin = pinEnv([A, B]);
    pin.clear();
    expect(process.env[A]).toBeUndefined();
    expect(process.env[B]).toBeUndefined();
  });
  it("restores a value that was set", () => {
    process.env[A] = "7";
    const pin = pinEnv([A]);
    pin.clear();
    pin.restore();
    expect(process.env[A]).toBe("7");
  });
  it("restores absence, rather than leaving an empty string behind", () => {
    const pin = pinEnv([A]);
    pin.clear();
    process.env[A] = "set by the code under test";
    pin.restore();
    expect(A in process.env).toBe(false);
  });
  it("re-reads on every clear, so a later value is not lost", () => {
    const pin = pinEnv([A]);
    pin.clear();
    pin.restore();
    process.env[A] = "9";
    pin.clear();
    pin.restore();
    expect(process.env[A]).toBe("9");
  });
});
