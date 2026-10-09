import { mkdtempSync, readFileSync, readdirSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import observer, { createObserver } from "./index.ts";

const here = dirname(fileURLToPath(import.meta.url));
const extRoot = join(here, "..");

function rows(dir: string) {
  return readFileSync(join(dir, "ab-observer.jsonl"), "utf-8")
    .trim()
    .split("\n")
    .map((l) => JSON.parse(l));
}

function fakePi() {
  const handlers = new Map<string, Function[]>();
  return {
    handlers,
    on(name: string, fn: Function) {
      handlers.set(name, [...(handlers.get(name) ?? []), fn]);
    },
    getCommands: () => [{ name: "blackhole" }],
    getAllTools: () => [{ name: "read" }],
  };
}

describe("zzz-ab-observer", () => {
  it("is inert without both env vars", () => {
    const pi = fakePi();
    const saved = { ...process.env };
    delete process.env.LITTLE_CODER_AB_OBSERVER;
    process.env.LITTLE_CODER_PI_SESSION_DIR = "/nonexistent";
    observer(pi as any);
    expect(pi.handlers.size).toBe(0);
    process.env.LITTLE_CODER_AB_OBSERVER = "1";
    delete process.env.LITTLE_CODER_PI_SESSION_DIR;
    observer(pi as any);
    expect(pi.handlers.size).toBe(0);
    process.env = saved;
  });

  it("never returns a value and swallows its own errors", async () => {
    const dir = mkdtempSync(join(tmpdir(), "abobs-"));
    const saved = { ...process.env };
    process.env.LITTLE_CODER_AB_OBSERVER = "1";
    process.env.LITTLE_CODER_PI_SESSION_DIR = dir;
    const pi = fakePi();
    observer(pi as any);
    process.env = saved;
    for (const [, fns] of pi.handlers) {
      for (const fn of fns) {
        expect(await fn({ payload: { messages: [{ role: "system", content: "s" }] }, message: null })).toBeUndefined();
        expect(await fn(undefined)).toBeUndefined();
      }
    }
  });

  it("dumps the last and the newest completed payload at compaction, then the first post-compaction request", () => {
    const dir = mkdtempSync(join(tmpdir(), "abobs-"));
    const obs = createObserver(dir, () => 1000);
    const p1 = { messages: [{ role: "system", content: "S" }, { role: "user", content: "a" }], tools: [1] };
    const p2 = { messages: [...p1.messages, { role: "assistant", content: "b" }], tools: [1] };
    obs.onSessionStart(["blackhole"], ["read"]);
    obs.onRequest(p1);
    obs.onMessageEnd({ role: "assistant", stopReason: "stop" });
    obs.onRequest(p2); // aborted by the compaction: never completed
    obs.onBeforeCompact({ reason: "manual", willRetry: false, preparation: { tokensBefore: 9, firstKeptEntryId: "e1" } });
    obs.onCompact({ fromExtension: true, compactionEntry: { summary: "xyz", details: { compactor: "blackhole" }, firstKeptEntryId: "e1" } });
    const post = { messages: [{ role: "system", content: "S" }, { role: "user", content: "summary" }] };
    obs.onRequest(post);
    expect(JSON.parse(readFileSync(join(dir, "compaction-1-payload-last.json"), "utf-8"))).toEqual(p2);
    expect(JSON.parse(readFileSync(join(dir, "compaction-1-payload-completed.json"), "utf-8"))).toEqual(p1);
    expect(JSON.parse(readFileSync(join(dir, "compaction-1-post.json"), "utf-8"))).toEqual(post);
    const r = rows(dir);
    expect(r.map((x) => x.kind)).toEqual(["arm_env", "request", "request", "before_compact", "compact", "request"]);
    expect(r[0].commands).toEqual(["blackhole"]);
    expect(r[3]).toMatchObject({ n: 1, reason: "manual", last_completed: false, dumped_last: true, dumped_completed: true });
    expect(r[4]).toMatchObject({ n: 1, from_extension: true, compactor: "blackhole", summary_chars: 3 });
    expect(r[1].system_sha1).toBe(r[2].system_sha1);
  });

  it("sorts after every extension, so it sees the final payload", () => {
    const names = readdirSync(extRoot)
      .filter((n) => existsSync(join(extRoot, n, "index.ts")))
      .sort();
    expect(names[names.length - 1]).toBe("zzz-ab-observer");
  });
});
