import { describe, it, expect, beforeEach, afterEach } from "vitest";
import {
  mkdtempSync,
  mkdirSync,
  rmSync,
  writeFileSync,
  existsSync,
  readdirSync,
  readFileSync,
  chmodSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import setupCheckpoint, { checkpointPath, tracked, MAX_BACKUP_BYTES } from "./index.ts";

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

describe("checkpoint hardening — keying, oversize/unreadable/dir safety, ShellSend, dir resilience", () => {
  let home: string;
  let origHome: string | undefined;
  beforeEach(() => {
    origHome = process.env.HOME;
    home = mkdtempSync(join(tmpdir(), "ckpt-hard-"));
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

  it("unifies the key across a relative Write and an absolute shell redirect to the same physical file", async () => {
    const h = setup();
    const abs = join(home, "same.txt");
    writeFileSync(abs, "ORIGINAL-BYTES");
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-u1.json" } });

    await h.tool_call({ toolName: "write", input: { path: "same.txt" } }, { cwd: home });
    await h.tool_call({ toolName: "bash", input: { command: `echo x > ${abs}` } }, { cwd: home });

    const dir = ckptDir("sess-u1.json");
    const files = readdirSync(dir);
    expect(files.length).toBe(1);
    expect(readFileSync(join(dir, files[0]), "utf8")).toBe("ORIGINAL-BYTES");
  });

  it("unifies a `~/` shell target with its absolute equivalent", async () => {
    const h = setup();
    const abs = join(home, "notes.md");
    writeFileSync(abs, "TILDE-ORIGINAL");
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-u2.json" } });

    await h.tool_call({ toolName: "bash", input: { command: `echo a > ~/notes.md` } }, { cwd: home });
    await h.tool_call({ toolName: "bash", input: { command: `echo b > ${abs}` } }, { cwd: home });

    const dir = ckptDir("sess-u2.json");
    const files = readdirSync(dir);
    expect(files.length).toBe(1);
    expect(readFileSync(join(dir, files[0]), "utf8")).toBe("TILDE-ORIGINAL");
  });

  it("keys a bare-absolute shell-redirect target on the literal path, not cwd-joined (unlike write-guard's rule)", async () => {
    const h = setup();
    const literal = "/nonexistent-lc-test-should-not-exist-abc123.md";
    const cwd = join(home, "some", "subdir");
    mkdirSync(cwd, { recursive: true });
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-u3.json" } });

    await h.tool_call({ toolName: "bash", input: { command: `echo x > ${literal}` } }, { cwd });

    const dir = ckptDir("sess-u3.json");
    const files = readdirSync(dir);
    // The literal absolute path doesn't exist, so it's tracked via the
    // `.absent` sentinel — its name encodes the actual key used. If the
    // shell branch wrongly cwd-joined a bare-absolute path (write-guard's
    // rule), the sentinel would instead encode `<cwd>/nonexistent-...md`.
    const expectedSentinel = literal.replace(/[^A-Za-z0-9._-]/g, "_").slice(-200) + ".absent";
    expect(files).toEqual([expectedSentinel]);
  });

  it("backs up a `cp` destination's pre-existing bytes, not the source's", async () => {
    const h = setup();
    const src = join(home, "src4.txt");
    const dst = join(home, "dst4.txt");
    writeFileSync(src, "SRC-CONTENT");
    writeFileSync(dst, "DST-ORIGINAL");
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-u4.json" } });

    await h.tool_call({ toolName: "bash", input: { command: `cp ${src} ${dst}` } }, { cwd: home });

    const dir = ckptDir("sess-u4.json");
    const files = readdirSync(dir);
    expect(files.length).toBe(1);
    expect(readFileSync(join(dir, files[0]), "utf8")).toBe("DST-ORIGINAL");
  });

  it("backs up a GNU `sed -i` target's pre-state", async () => {
    const h = setup();
    const file = join(home, "sed5.txt");
    writeFileSync(file, "aaa");
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-u5.json" } });

    await h.tool_call({ toolName: "bash", input: { command: `sed -i 's/a/b/' ${file}` } }, { cwd: home });

    const dir = ckptDir("sess-u5.json");
    const files = readdirSync(dir);
    expect(files.length).toBe(1);
    expect(readFileSync(join(dir, files[0]), "utf8")).toBe("aaa");
  });

  it("backs up an `mv` destination's pre-state, not the source's", async () => {
    const h = setup();
    const src = join(home, "movesrc.txt");
    const dst = join(home, "movedst.txt");
    writeFileSync(src, "MOVE-SRC");
    writeFileSync(dst, "MOVE-DST-ORIGINAL");
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-u6.json" } });

    await h.tool_call({ toolName: "bash", input: { command: `mv ${src} ${dst}` } }, { cwd: home });

    const dir = ckptDir("sess-u6.json");
    const files = readdirSync(dir);
    expect(files.length).toBe(1);
    expect(readFileSync(join(dir, files[0]), "utf8")).toBe("MOVE-DST-ORIGINAL");
  });

  it("skips a target that still contains unexpanded shell syntax", async () => {
    const h = setup();
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-u7.json" } });

    await h.tool_call({ toolName: "bash", input: { command: `echo hi > $OUT` } }, { cwd: home });
    await h.tool_call({ toolName: "bash", input: { command: `cp file.txt "$DEST/b.txt"` } }, { cwd: home });

    expect(existsSync(ckptDir("sess-u7.json"))).toBe(false);
  });

  it("writes a `.toolarge` sentinel (no full copy) for an oversize file, and never retries it even after it shrinks", async () => {
    const h = setup();
    const file = join(home, "big.bin");
    writeFileSync(file, Buffer.alloc(MAX_BACKUP_BYTES + 1, 1));
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-u8.json" } });

    await h.tool_call({ toolName: "bash", input: { command: `echo x > ${file}` } }, { cwd: home });

    const dir = ckptDir("sess-u8.json");
    const expectedSentinel = file.replace(/[^A-Za-z0-9._-]/g, "_").slice(-200) + ".toolarge";
    let files = readdirSync(dir);
    expect(files).toEqual([expectedSentinel]);
    expect(readFileSync(join(dir, expectedSentinel), "utf8")).toBe(String(MAX_BACKUP_BYTES + 1));

    // Shrink the file and trigger the same target again — first-write-wins
    // tracks it forever, so this must stay a no-op.
    writeFileSync(file, "now tiny");
    await h.tool_call({ toolName: "bash", input: { command: `echo y > ${file}` } }, { cwd: home });
    files = readdirSync(dir);
    expect(files).toEqual([expectedSentinel]);
  });

  it("skips (and does not track) a directory hit; a later real file at the same path backs up correctly", async () => {
    const h = setup();
    const target = join(home, "was-a-dir");
    mkdirSync(target);
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-u9.json" } });

    await h.tool_call({ toolName: "bash", input: { command: `echo x > ${target}` } }, { cwd: home });
    const dir = ckptDir("sess-u9.json");
    expect(existsSync(dir)).toBe(false);

    rmSync(target, { recursive: true, force: true });
    writeFileSync(target, "REAL FILE CONTENT");
    await h.tool_call({ toolName: "bash", input: { command: `echo y > ${target}` } }, { cwd: home });

    const files = readdirSync(dir);
    expect(files.length).toBe(1);
    expect(readFileSync(join(dir, files[0]), "utf8")).toBe("REAL FILE CONTENT");
  });

  const isRoot = typeof process.getuid === "function" && process.getuid() === 0;
  (isRoot ? it.skip : it)(
    "skips (and does not track) an unreadable file; it backs up correctly once made readable again",
    async () => {
      const h = setup();
      const file = join(home, "unreadable.txt");
      writeFileSync(file, "SECRET-ORIGINAL");
      chmodSync(file, 0o000);
      try {
        await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-u10.json" } });
        await h.tool_call({ toolName: "bash", input: { command: `echo x > ${file}` } }, { cwd: home });

        const dir = ckptDir("sess-u10.json");
        const expectedName = file.replace(/[^A-Za-z0-9._-]/g, "_").slice(-200);
        expect(existsSync(join(dir, expectedName))).toBe(false);

        chmodSync(file, 0o644);
        await h.tool_call({ toolName: "bash", input: { command: `echo y > ${file}` } }, { cwd: home });
        expect(readFileSync(join(dir, expectedName), "utf8")).toBe("SECRET-ORIGINAL");
      } finally {
        chmodSync(file, 0o644);
      }
    },
  );

  it("backs up a ShellSend text payload that redirects to a file", async () => {
    const h = setup();
    const file = join(home, "send1.txt");
    writeFileSync(file, "ORIG-SEND");
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-u11.json" } });

    await h.tool_call({ toolName: "ShellSend", input: { text: `echo hi > ${file}` } }, { cwd: home });

    const dir = ckptDir("sess-u11.json");
    const files = readdirSync(dir);
    expect(files.length).toBe(1);
    expect(readFileSync(join(dir, files[0]), "utf8")).toBe("ORIG-SEND");
  });

  it("is a no-op for a ShellSend payload with no write in it", async () => {
    const h = setup();
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-u12.json" } });

    await h.tool_call({ toolName: "ShellSend", input: { text: "just some repl input\n" } }, { cwd: home });

    expect(existsSync(ckptDir("sess-u12.json"))).toBe(false);
  });

  it("recreates the checkpoint dir if it was deleted mid-session, so a later backup still succeeds", async () => {
    const h = setup();
    const file1 = join(home, "first.txt");
    const file2 = join(home, "second.txt");
    writeFileSync(file1, "FIRST-ORIGINAL");
    writeFileSync(file2, "SECOND-ORIGINAL");
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-u13.json" } });

    await h.tool_call({ toolName: "write", input: { path: file1 } }, { cwd: home });
    const dir = ckptDir("sess-u13.json");
    expect(existsSync(dir)).toBe(true);
    rmSync(dir, { recursive: true, force: true });

    await h.tool_call({ toolName: "write", input: { path: file2 } }, { cwd: home });
    expect(existsSync(dir)).toBe(true);
    const files = readdirSync(dir);
    expect(files.length).toBe(1);
    expect(readFileSync(join(dir, files[0]), "utf8")).toBe("SECOND-ORIGINAL");
  });
});
