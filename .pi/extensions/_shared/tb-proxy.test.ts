import { describe, it, expect, afterEach } from "vitest";
import { inTbMode, tbProxyRun, tbSessionId, TB_MODE_ENV, TB_PROXY_PREFIX } from "./tb-proxy.ts";

function ctxOf(input: (title: string, initial?: string) => Promise<unknown>) {
  return { ui: { input } };
}

afterEach(() => {
  delete process.env[TB_MODE_ENV];
  delete process.env.LITTLE_CODER_SESSION_ID;
});

describe("inTbMode", () => {
  it("is true only for the exact string '1'", () => {
    expect(inTbMode()).toBe(false);
    process.env[TB_MODE_ENV] = "1";
    expect(inTbMode()).toBe(true);
    process.env[TB_MODE_ENV] = "true";
    expect(inTbMode()).toBe(false);
  });
});

describe("tbSessionId", () => {
  it("falls back to 'default' when unset", () => {
    expect(tbSessionId()).toBe("default");
  });

  it("reads LITTLE_CODER_SESSION_ID when set", () => {
    process.env.LITTLE_CODER_SESSION_ID = "abc123";
    expect(tbSessionId()).toBe("abc123");
  });
});

describe("tbProxyRun", () => {
  it("returns the adapter's response verbatim", async () => {
    const ctx = ctxOf(async () => "[exit=0 cwd=/app timed_out=false]");
    const out = await tbProxyRun(ctx, "echo hi", 10, "s1");
    expect(out).toBe("[exit=0 cwd=/app timed_out=false]");
  });

  it("sends the documented op:run payload on the __LC_TB_SHELL__ channel", async () => {
    let seenTitle = "";
    const ctx = ctxOf(async (title) => {
      seenTitle = title;
      return "ok";
    });
    await tbProxyRun(ctx, "echo hi", 5, "s1");
    expect(seenTitle.startsWith(TB_PROXY_PREFIX)).toBe(true);
    const payload = JSON.parse(seenTitle.slice(TB_PROXY_PREFIX.length));
    expect(payload).toEqual({ op: "run", session_id: "s1", command: "echo hi", timeout: 5 });
  });

  it("treats a non-string response as no usable response", async () => {
    const ctx = ctxOf(async () => undefined);
    expect(await tbProxyRun(ctx, "echo hi", 10, "s1")).toBeNull();
  });

  // An adapter-formatted result always carries a footer, so an empty string
  // can only mean nothing came back -- not a real zero-byte result.
  it("treats an empty string as no usable response", async () => {
    const ctx = ctxOf(async () => "");
    expect(await tbProxyRun(ctx, "echo hi", 10, "s1")).toBeNull();
  });

  it("treats a rejected ui.input call as no usable response, not a thrown error", async () => {
    const ctx = ctxOf(async () => {
      throw new Error("channel closed");
    });
    await expect(tbProxyRun(ctx, "echo hi", 10, "s1")).resolves.toBeNull();
  });
});
