import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { loadArm } from "./arm-shim.ts";

const here = dirname(fileURLToPath(import.meta.url));
const extRoot = join(here, "..");

function fakePi(throwOnAppend: boolean) {
  const handlers = new Map<string, Function[]>();
  const appended: unknown[] = [];
  return {
    handlers,
    appended,
    on(name: string, fn: Function) {
      handlers.set(name, [...(handlers.get(name) ?? []), fn]);
    },
    appendEntry(type: string, data: unknown) {
      if (throwOnAppend) throw new Error("Extension runtime not initialized");
      appended.push({ type, data });
    },
  };
}

describe("arm shim", () => {
  it("never evaluates the vendored module when the env names another arm", async () => {
    const saved = process.env.LC_COMPACTION_ARM;
    process.env.LC_COMPACTION_ARM = "prefix-cache";
    let loaded = false;
    const pi = fakePi(true);
    const on = await loadArm(pi as any, "blackhole", async () => {
      loaded = true;
      throw new Error("must not load");
    });
    expect(on).toBe(false);
    expect(loaded).toBe(false);
    expect(pi.handlers.size).toBe(0);
    if (saved === undefined) delete process.env.LC_COMPACTION_ARM;
    else process.env.LC_COMPACTION_ARM = saved;
  });

  it("does not call appendEntry while loading; records registration at session_start", async () => {
    const saved = { ...process.env };
    process.env.LC_COMPACTION_ARM = "blackhole";
    process.env.LITTLE_CODER_TELEMETRY = "1";
    const pi = fakePi(true); // appendEntry throws during load, as in pi
    let registered = false;
    await loadArm(pi as any, "blackhole", async () => ({ default: () => { registered = true; } }));
    expect(registered).toBe(true);
    const start = pi.handlers.get("session_start") ?? [];
    expect(start.length).toBe(1);
    const live = fakePi(false);
    (pi as any).appendEntry = live.appendEntry.bind(live);
    await start[0]();
    expect(live.appended).toEqual([{ type: "lc-telemetry", data: { arm: "blackhole", v: 1, kind: "arm_registered" } }]);
    process.env = saved;
  });

  it("every zz-arm-* capturer sorts after every payload rewriter", () => {
    const names = readdirSync(extRoot).sort();
    const rewriters = names.filter((n) => {
      try {
        return /pi\.on\(\s*["']before_provider_request["']/.test(readFileSync(join(extRoot, n, "index.ts"), "utf-8"))
          && n !== "zzz-ab-observer" && n !== "cache-reuse-compaction";
      } catch {
        return false;
      }
    });
    for (const arm of names.filter((n) => n.startsWith("zz-arm-"))) {
      for (const r of rewriters) expect([r, arm].sort()[0]).toBe(r);
    }
  });
});
