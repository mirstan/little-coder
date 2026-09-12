import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync, writeFileSync, existsSync, readdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import setupCheckpoint, { checkpointPath, tracked } from "./index.ts";

describe("checkpointPath", () => {
  it("reads the `path` key (current pi write/edit)", () => {
    expect(checkpointPath({ path: "/a/b.ts" })).toBe("/a/b.ts");
  });
  it("falls back to the legacy `file_path` key", () => {
    expect(checkpointPath({ file_path: "/a/c.ts" })).toBe("/a/c.ts");
  });
  it("prefers `path` when both are present", () => {
    expect(checkpointPath({ path: "/p", file_path: "/f" })).toBe("/p");
  });
  it("returns undefined when neither is a string", () => {
    expect(checkpointPath({})).toBeUndefined();
    expect(checkpointPath({ path: 5 as unknown as string })).toBeUndefined();
  });
});

function setup() {
  const handlers: Record<string, (...args: any[]) => any> = {};
  const pi = {
    on(name: string, h: (...args: any[]) => any) {
      handlers[name] = h;
    },
  };
  setupCheckpoint(pi as any);
  return handlers;
}

// homedir() honors $HOME on POSIX, so pointing it at a tmpdir isolates the
// ~/.little-coder/checkpoints/ writes the backup net makes.
describe("checkpoint pre-edit backup net", () => {
  let home: string;
  let origHome: string | undefined;
  beforeEach(() => {
    origHome = process.env.HOME;
    home = mkdtempSync(join(tmpdir(), "ckpt-"));
    process.env.HOME = home;
    tracked.clear();
  });
  afterEach(() => {
    if (origHome === undefined) delete process.env.HOME;
    else process.env.HOME = origHome;
    rmSync(home, { recursive: true, force: true });
  });

  it("snapshots a file modified via the `path` key (regression: previously never fired)", async () => {
    const h = setup();
    const target = join(home, "src.txt");
    writeFileSync(target, "original");
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-123.json" } });
    await h.tool_call({ toolName: "write", input: { path: target, content: "new" } });

    const dir = join(home, ".little-coder", "checkpoints", "sess-123.json");
    expect(existsSync(dir)).toBe(true);
    expect(readdirSync(dir).length).toBe(1);
  });

  it("also snapshots via the legacy `file_path` key", async () => {
    const h = setup();
    const target = join(home, "src2.txt");
    writeFileSync(target, "orig");
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-9.json" } });
    await h.tool_call({ toolName: "Edit", input: { file_path: target, edits: [] } });

    expect(existsSync(join(home, ".little-coder", "checkpoints", "sess-9.json"))).toBe(true);
  });

  it("ignores tool calls that carry no path", async () => {
    const h = setup();
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-0.json" } });
    await h.tool_call({ toolName: "write", input: { content: "no path here" } });
    expect(existsSync(join(home, ".little-coder", "checkpoints", "sess-0.json"))).toBe(false);
  });
});

describe("checkpoint pre-edit backup net — shell writes", () => {
  let home: string;
  let origHome: string | undefined;
  beforeEach(() => {
    origHome = process.env.HOME;
    home = mkdtempSync(join(tmpdir(), "ckpt-shell-"));
    process.env.HOME = home;
    tracked.clear();
  });
  afterEach(() => {
    if (origHome === undefined) delete process.env.HOME;
    else process.env.HOME = origHome;
    rmSync(home, { recursive: true, force: true });
  });

  function ckptDir(sessionFile: string): string {
    return join(home, ".little-coder", "checkpoints", sessionFile.split("/").pop()!);
  }

  it("backs up a file redirected to via `>` in a bash call", async () => {
    const h = setup();
    const target = join(home, "task.txt");
    writeFileSync(target, "original content");
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-sh1.json" } });
    await h.tool_call({ toolName: "bash", input: { command: `echo garbage > ${target}` } });

    const dir = ckptDir("sess-sh1.json");
    expect(existsSync(dir)).toBe(true);
    expect(readdirSync(dir).length).toBe(1);
  });

  it("backs up a file redirected to via `>>` in a ShellSession call", async () => {
    const h = setup();
    const target = join(home, "task2.txt");
    writeFileSync(target, "original content");
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-sh2.json" } });
    await h.tool_call({ toolName: "ShellSession", input: { command: `echo more >> ${target}` } });

    expect(existsSync(ckptDir("sess-sh2.json"))).toBe(true);
  });

  it("backs up every file targeted by `tee`", async () => {
    const h = setup();
    const target = join(home, "task3.txt");
    writeFileSync(target, "original content");
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-sh3.json" } });
    await h.tool_call({ toolName: "bash", input: { command: `echo hi | tee ${target}` } });

    expect(existsSync(ckptDir("sess-sh3.json"))).toBe(true);
  });

  it("backs up the `of=` target of `dd`", async () => {
    const h = setup();
    const target = join(home, "task4.txt");
    writeFileSync(target, "original content");
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-sh4.json" } });
    await h.tool_call({
      toolName: "bash",
      input: { command: `dd if=/dev/zero of=${target} bs=1 count=1` },
    });

    expect(existsSync(ckptDir("sess-sh4.json"))).toBe(true);
  });

  it("is a no-op for a read-only shell command", async () => {
    const h = setup();
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-sh5.json" } });
    await h.tool_call({ toolName: "bash", input: { command: `cat somefile.txt | grep foo` } });

    expect(existsSync(ckptDir("sess-sh5.json"))).toBe(false);
  });

  it("first-write-wins across a Write call followed by a shell redirect to the same path", async () => {
    const h = setup();
    const target = join(home, "shared.txt");
    writeFileSync(target, "PRE-EXISTING");
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-sh6.json" } });

    await h.tool_call({ toolName: "Write", input: { path: target, content: "intermediate" } });
    // Simulate the Write actually landing, then a later shell redirect to the
    // same path in the same session.
    writeFileSync(target, "intermediate");
    await h.tool_call({ toolName: "bash", input: { command: `echo clobber > ${target}` } });

    const dir = ckptDir("sess-sh6.json");
    const files = readdirSync(dir);
    expect(files.length).toBe(1);
    const backedUp = readFileSync(join(dir, files[0]), "utf8");
    expect(backedUp).toBe("PRE-EXISTING");
  });

  it("attempts a best-effort resolution for a relative path without crashing", async () => {
    const h = setup();
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-sh7.json" } });

    await expect(
      h.tool_call({ toolName: "bash", input: { command: `echo hi > relative-out.txt` } }),
    ).resolves.not.toThrow();

    // Best-effort: resolves against ctx.cwd (defaulting to process.cwd() when
    // no ctx is supplied), and the `.absent` sentinel covers the common case
    // where the path doesn't exist there.
    const dir = ckptDir("sess-sh7.json");
    expect(existsSync(dir)).toBe(true);
  });
});
