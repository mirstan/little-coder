import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  mkdtempSync,
  mkdirSync,
  rmSync,
  writeFileSync,
  existsSync,
  readdirSync,
  readFileSync,
  chmodSync,
  symlinkSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import setupCheckpoint, {
  checkpointPath,
  tracked,
  MAX_BACKUP_BYTES,
  snapshotWorkingDirectory,
} from "./index.ts";

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
    // An ambient benchmark/session-id var on a benchmark box must not flake
    // these tests, which assume interactive (no-env-var) behavior.
    delete process.env.LITTLE_CODER_BENCHMARK;
    delete process.env.LITTLE_CODER_SESSION_ID;
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
    delete process.env.LITTLE_CODER_BENCHMARK;
    delete process.env.LITTLE_CODER_SESSION_ID;
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
    delete process.env.LITTLE_CODER_BENCHMARK;
    delete process.env.LITTLE_CODER_SESSION_ID;
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

  it("normalizes an absolute path so ./ and // spellings key on the same physical file", async () => {
    const h = setup();
    const abs = join(home, "norm.txt");
    writeFileSync(abs, "NORM-ORIGINAL");
    const dotted = join(home, ".", "norm.txt");
    const doubled = home + "//norm.txt";
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-u5.json" } });

    await h.tool_call({ toolName: "bash", input: { command: `echo a > ${dotted}` } }, { cwd: home });
    writeFileSync(abs, "CLOBBERED-BY-FIRST-WRITE");
    await h.tool_call({ toolName: "bash", input: { command: `echo b > ${doubled}` } }, { cwd: home });

    const dir = ckptDir("sess-u5.json");
    const files = readdirSync(dir);
    // Unnormalized, these three spellings would be three distinct tracking
    // keys -- the second "backup" would actually capture the already-
    // clobbered bytes under a different filename, silently lying about what
    // the original content was.
    expect(files.length).toBe(1);
    expect(readFileSync(join(dir, files[0]), "utf8")).toBe("NORM-ORIGINAL");
  });

  it("backs up a bracket-named file that exists on disk (Next.js/Remix-style routes)", async () => {
    const h = setup();
    const routesDir = join(home, "routes");
    mkdirSync(routesDir, { recursive: true });
    const target = join(routesDir, "[id].tsx");
    writeFileSync(target, "ORIGINAL-ROUTE-CONTENT");
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-u6.json" } });

    await h.tool_call(
      { toolName: "bash", input: { command: `echo x > routes/[id].tsx` } },
      { cwd: home },
    );

    const dir = ckptDir("sess-u6.json");
    const files = readdirSync(dir);
    // Before the fix, the bare presence of `[`/`]` unconditionally excluded
    // this target as "unexpanded glob syntax" -- silently backing up nothing
    // for a real, existing file.
    expect(files.length).toBe(1);
    expect(readFileSync(join(dir, files[0]), "utf8")).toBe("ORIGINAL-ROUTE-CONTENT");
  });

  it("still skips a genuinely nonexistent bracket-shaped path (can't tell glob from literal)", async () => {
    const h = setup();
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-u7.json" } });

    await h.tool_call(
      { toolName: "bash", input: { command: `echo x > routes/[slug].tsx` } },
      { cwd: home },
    );

    const dir = ckptDir("sess-u7.json");
    expect(existsSync(dir) ? readdirSync(dir).length : 0).toBe(0);
  });

  it("still skips a target containing real shell substitution", async () => {
    const h = setup();
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-u8.json" } });

    await h.tool_call(
      { toolName: "bash", input: { command: `echo x > "$OUT"` } },
      { cwd: home },
    );

    const dir = ckptDir("sess-u8.json");
    expect(existsSync(dir) ? readdirSync(dir).length : 0).toBe(0);
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

  it("backs up the clobbered file inside a directory destination for `cp dir/`", async () => {
    const h = setup();
    const src = join(home, "report.md");
    const destDir = join(home, "docs");
    mkdirSync(destDir);
    const clobbered = join(destDir, "report.md");
    writeFileSync(src, "NEW-REPORT");
    writeFileSync(clobbered, "DOCS-ORIGINAL");
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-d1.json" } });

    await h.tool_call({ toolName: "bash", input: { command: `cp ${src} ${destDir}/` } }, { cwd: home });

    // The destination *directory* is not a file and must not be snapshotted;
    // the file it actually overwrites, docs/report.md, must be.
    const dir = ckptDir("sess-d1.json");
    const files = readdirSync(dir);
    expect(files.length).toBe(1);
    expect(readFileSync(join(dir, files[0]), "utf8")).toBe("DOCS-ORIGINAL");
  });

  it("backs up every clobbered file for a multi-source `mv` into a directory", async () => {
    const h = setup();
    const destDir = join(home, "dest");
    mkdirSync(destDir);
    for (const n of ["a.md", "b.md"]) {
      writeFileSync(join(home, n), `SRC-${n}`);
      writeFileSync(join(destDir, n), `DEST-ORIGINAL-${n}`);
    }
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-d2.json" } });

    await h.tool_call(
      { toolName: "bash", input: { command: `mv ${join(home, "a.md")} ${join(home, "b.md")} ${destDir}/` } },
      { cwd: home },
    );

    const dir = ckptDir("sess-d2.json");
    const contents = readdirSync(dir)
      .map((f) => readFileSync(join(dir, f), "utf8"))
      .sort();
    expect(contents).toEqual(["DEST-ORIGINAL-a.md", "DEST-ORIGINAL-b.md"]);
  });

  it("handles `cp -t DIR` (every operand is a source)", async () => {
    const h = setup();
    const destDir = join(home, "tdest");
    mkdirSync(destDir);
    writeFileSync(join(home, "c.md"), "SRC-C");
    writeFileSync(join(destDir, "c.md"), "TDEST-ORIGINAL");
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-d3.json" } });

    await h.tool_call(
      { toolName: "bash", input: { command: `cp -t ${destDir} ${join(home, "c.md")}` } },
      { cwd: home },
    );

    const dir = ckptDir("sess-d3.json");
    const files = readdirSync(dir);
    expect(files.length).toBe(1);
    expect(readFileSync(join(dir, files[0]), "utf8")).toBe("TDEST-ORIGINAL");
  });

  it("still keys on the destination itself when it is a plain file, not a directory", async () => {
    const h = setup();
    const src = join(home, "s7.md");
    const dst = join(home, "d7.md");
    writeFileSync(src, "SRC7");
    writeFileSync(dst, "DST7-ORIGINAL");
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-d4.json" } });

    await h.tool_call({ toolName: "bash", input: { command: `cp ${src} ${dst}` } }, { cwd: home });

    const dir = ckptDir("sess-d4.json");
    const files = readdirSync(dir);
    expect(files.length).toBe(1);
    expect(readFileSync(join(dir, files[0]), "utf8")).toBe("DST7-ORIGINAL");
  });

  it("matches the write/edit tool name case-insensitively, like write-guard does", async () => {
    const h = setup();
    const file = join(home, "cased.md");
    writeFileSync(file, "CASED-ORIGINAL");
    await h.session_start({}, { sessionManager: { getSessionFile: () => "/x/sess-d5.json" } });

    await h.tool_call({ toolName: "EDIT", input: { path: file } }, { cwd: home });

    const dir = ckptDir("sess-d5.json");
    const files = readdirSync(dir);
    expect(files.length).toBe(1);
    expect(readFileSync(join(dir, files[0]), "utf8")).toBe("CASED-ORIGINAL");
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

// GAIA-only session-start walk. `taskDir` stands in for gaia.py's fresh
// host-side TemporaryDirectory (gaia.py:138-172) -- a directory distinct
// from `home`, mirroring the real shape where the task's work dir and
// ~/.little-coder are never the same tree.
describe("checkpoint session-start snapshot", () => {
  let home: string;
  let taskDir: string;
  let origHome: string | undefined;
  let origBenchmark: string | undefined;
  let origSessionId: string | undefined;

  function ckptDir(sessionId: string): string {
    return join(home, ".little-coder", "checkpoints", sessionId);
  }

  beforeEach(() => {
    origHome = process.env.HOME;
    origBenchmark = process.env.LITTLE_CODER_BENCHMARK;
    origSessionId = process.env.LITTLE_CODER_SESSION_ID;
    home = mkdtempSync(join(tmpdir(), "ckpt-snap-home-"));
    taskDir = mkdtempSync(join(tmpdir(), "ckpt-snap-task-"));
    process.env.HOME = home;
    process.env.LITTLE_CODER_BENCHMARK = "gaia";
    delete process.env.LITTLE_CODER_SESSION_ID;
    delete process.env.LITTLE_CODER_CHECKPOINT_SNAPSHOT_MAX_ENTRIES;
    delete process.env.LITTLE_CODER_CHECKPOINT_SNAPSHOT_MAX_BYTES;
    delete process.env.LITTLE_CODER_CHECKPOINT_SNAPSHOT_MAX_DEPTH;
    tracked.clear();
  });

  afterEach(() => {
    if (origHome === undefined) delete process.env.HOME;
    else process.env.HOME = origHome;
    if (origBenchmark === undefined) delete process.env.LITTLE_CODER_BENCHMARK;
    else process.env.LITTLE_CODER_BENCHMARK = origBenchmark;
    if (origSessionId === undefined) delete process.env.LITTLE_CODER_SESSION_ID;
    else process.env.LITTLE_CODER_SESSION_ID = origSessionId;
    delete process.env.LITTLE_CODER_CHECKPOINT_SNAPSHOT_MAX_ENTRIES;
    delete process.env.LITTLE_CODER_CHECKPOINT_SNAPSHOT_MAX_BYTES;
    delete process.env.LITTLE_CODER_CHECKPOINT_SNAPSHOT_MAX_DEPTH;
    rmSync(home, { recursive: true, force: true });
    rmSync(taskDir, { recursive: true, force: true });
  });

  // 1. Regression shape, end to end: a subprocess-style overwrite with no
  // tool_call at all must still be recoverable, because the session-start
  // walk captured the original before the model ever acted.
  it("backs up a pre-existing task file at session start, surviving a subprocess-style overwrite with no tool_call", async () => {
    const h = setup();
    const target = join(taskDir, "input.tex");
    writeFileSync(target, "ORIGINAL");
    await h.session_start(
      {},
      { sessionManager: { getSessionFile: () => undefined }, cwd: taskDir, ui: { notify: vi.fn() } },
    );

    // No tool_call at all here -- e.g. a spawned script's own internal
    // `open(path).write(...)`, which no shell-syntax detector can see.
    writeFileSync(target, "CLOBBERED-BY-SUBPROCESS");

    const dir = ckptDir("default");
    const files = readdirSync(dir);
    expect(files.length).toBe(1);
    expect(readFileSync(join(dir, files[0]), "utf8")).toBe("ORIGINAL");
  });

  // 2. First-write-wins integration: the walk and the per-write net share
  // `tracked`, so a later write tool_call to an already-walked file must
  // not overwrite the session-start original with a second backup.
  it("first-write-wins: a post-walk write tool_call to a walked file adds no second backup", async () => {
    const h = setup();
    const target = join(taskDir, "notes.txt");
    writeFileSync(target, "ORIGINAL");
    await h.session_start(
      {},
      { sessionManager: { getSessionFile: () => undefined }, cwd: taskDir, ui: { notify: vi.fn() } },
    );

    await h.tool_call({ toolName: "write", input: { path: target, content: "model output" } }, { cwd: taskDir });

    const dir = ckptDir("default");
    const files = readdirSync(dir);
    expect(files.length).toBe(1);
    expect(readFileSync(join(dir, files[0]), "utf8")).toBe("ORIGINAL");
  });

  // 3. Gating: an allowlist, not a truthy gate.
  describe("gating", () => {
    it("does not walk when LITTLE_CODER_BENCHMARK is unset", async () => {
      delete process.env.LITTLE_CODER_BENCHMARK;
      const h = setup();
      writeFileSync(join(taskDir, "f.txt"), "x");
      await h.session_start(
        {},
        { sessionManager: { getSessionFile: () => undefined }, cwd: taskDir, ui: { notify: vi.fn() } },
      );
      expect(existsSync(ckptDir("default"))).toBe(false);
    });

    it('does not walk for "terminal_bench" -- the corrected premise: an allowlist, not truthiness', async () => {
      process.env.LITTLE_CODER_BENCHMARK = "terminal_bench";
      const h = setup();
      writeFileSync(join(taskDir, "f.txt"), "x");
      await h.session_start(
        {},
        { sessionManager: { getSessionFile: () => undefined }, cwd: taskDir, ui: { notify: vi.fn() } },
      );
      expect(existsSync(ckptDir("default"))).toBe(false);
    });

    it("does not walk when cwd is missing, even under the gaia gate", async () => {
      const h = setup();
      await h.session_start(
        {},
        { sessionManager: { getSessionFile: () => undefined }, ui: { notify: vi.fn() } }, // no cwd
      );
      expect(existsSync(ckptDir("default"))).toBe(false);
    });

    it("disables the walk entirely when SNAPSHOT_MAX_ENTRIES is <= 0", async () => {
      process.env.LITTLE_CODER_CHECKPOINT_SNAPSHOT_MAX_ENTRIES = "0";
      const h = setup();
      writeFileSync(join(taskDir, "f.txt"), "x");
      await h.session_start(
        {},
        { sessionManager: { getSessionFile: () => undefined }, cwd: taskDir, ui: { notify: vi.fn() } },
      );
      expect(existsSync(ckptDir("default"))).toBe(false);
    });
  });

  // 4. Session id: LITTLE_CODER_SESSION_ID wins over the derived session
  // file (even when both are present); unset falls back to the file, then
  // "default"; an empty string is treated as unset (the `||` vs `??` fix).
  describe("session id", () => {
    it("prefers LITTLE_CODER_SESSION_ID over the session file when both are present", async () => {
      process.env.LITTLE_CODER_SESSION_ID = "gaia-abc";
      const h = setup();
      writeFileSync(join(taskDir, "f.txt"), "ORIG");
      await h.session_start(
        {},
        { sessionManager: { getSessionFile: () => "/x/sess-999.json" }, cwd: taskDir, ui: { notify: vi.fn() } },
      );

      expect(existsSync(ckptDir("gaia-abc"))).toBe(true);
      expect(existsSync(ckptDir("sess-999.json"))).toBe(false);
    });

    it("falls back to the derived session-file id when LITTLE_CODER_SESSION_ID is unset, exactly as today", async () => {
      const h = setup();
      writeFileSync(join(taskDir, "f.txt"), "ORIG");
      await h.session_start(
        {},
        { sessionManager: { getSessionFile: () => "/x/sess-777.json" }, cwd: taskDir, ui: { notify: vi.fn() } },
      );
      expect(existsSync(ckptDir("sess-777.json"))).toBe(true);
    });

    it('falls back to "default" when both LITTLE_CODER_SESSION_ID and the session file are absent', async () => {
      const h = setup();
      writeFileSync(join(taskDir, "f.txt"), "ORIG");
      await h.session_start(
        {},
        { sessionManager: { getSessionFile: () => undefined }, cwd: taskDir, ui: { notify: vi.fn() } },
      );
      expect(existsSync(ckptDir("default"))).toBe(true);
    });

    it("treats an empty-string LITTLE_CODER_SESSION_ID as unset (`||`, not `??`)", async () => {
      process.env.LITTLE_CODER_SESSION_ID = "";
      const h = setup();
      writeFileSync(join(taskDir, "f.txt"), "ORIG");
      await h.session_start(
        {},
        { sessionManager: { getSessionFile: () => "/x/sess-555.json" }, cwd: taskDir, ui: { notify: vi.fn() } },
      );
      // An ambient empty-string var must fall through to the session file,
      // not become "" and (via backupIfNeeded's own `!sessionId` guard)
      // silently disable every backup for the session.
      expect(existsSync(ckptDir("sess-555.json"))).toBe(true);
    });
  });

  // 5. ui guard (Fix 2): a ctx lacking `ui` must never throw; when `ui` is
  // present, the partial-snapshot notice fires exactly once.
  describe("ui guard", () => {
    it("does not throw when the entry budget is hit and ctx has no ui", async () => {
      process.env.LITTLE_CODER_CHECKPOINT_SNAPSHOT_MAX_ENTRIES = "1";
      const h = setup();
      writeFileSync(join(taskDir, "a.txt"), "A");
      writeFileSync(join(taskDir, "b.txt"), "B");

      await expect(
        h.session_start({}, { sessionManager: { getSessionFile: () => undefined }, cwd: taskDir }), // no ui at all
      ).resolves.not.toThrow();
    });

    it("notifies exactly once, matching /harness intervention: .*partial/, when the entry budget is hit", async () => {
      process.env.LITTLE_CODER_CHECKPOINT_SNAPSHOT_MAX_ENTRIES = "1";
      const h = setup();
      writeFileSync(join(taskDir, "a.txt"), "A");
      writeFileSync(join(taskDir, "b.txt"), "B");
      const notify = vi.fn();

      await h.session_start(
        {},
        { sessionManager: { getSessionFile: () => undefined }, cwd: taskDir, ui: { notify } },
      );

      expect(notify).toHaveBeenCalledTimes(1);
      expect(notify.mock.calls[0][0]).toMatch(/harness intervention: .*partial/);
    });

    it("stays silent when the walk completes under budget", async () => {
      const h = setup();
      writeFileSync(join(taskDir, "a.txt"), "A");
      const notify = vi.fn();

      await h.session_start(
        {},
        { sessionManager: { getSessionFile: () => undefined }, cwd: taskDir, ui: { notify } },
      );

      expect(notify).not.toHaveBeenCalled();
    });
  });

  // 6. Symlink hazard: the confirmed cycle terminates promptly (never
  // `recursive: true`); a file symlink is backed up with the target's real
  // content; a symlinked directory is neither descended nor tracked.
  describe("symlinks", () => {
    it("terminates promptly on a symlinked directory cycle and still backs up real files inside it", async () => {
      const realDir = join(taskDir, "real");
      mkdirSync(realDir);
      writeFileSync(join(realDir, "keep.txt"), "KEEP-ME");
      // mkdir real; ln -s ../real real/loop -- a one-level cycle that
      // `readdirSync(dir, { recursive: true })` is confirmed to follow
      // 30+ levels deep before being killed.
      symlinkSync(join("..", "real"), join(realDir, "loop"), "dir");

      const h = setup();
      const start = Date.now();
      await h.session_start(
        {},
        { sessionManager: { getSessionFile: () => undefined }, cwd: taskDir, ui: { notify: vi.fn() } },
      );
      expect(Date.now() - start).toBeLessThan(5000);

      const dir = ckptDir("default");
      const contents = readdirSync(dir).map((f) => readFileSync(join(dir, f), "utf8"));
      expect(contents).toContain("KEEP-ME");
    });

    it("backs up a file symlink with the target's real content", async () => {
      const outsideTarget = join(home, "outside-target.txt");
      writeFileSync(outsideTarget, "TARGET-CONTENT");
      symlinkSync(outsideTarget, join(taskDir, "link.txt"));

      const h = setup();
      await h.session_start(
        {},
        { sessionManager: { getSessionFile: () => undefined }, cwd: taskDir, ui: { notify: vi.fn() } },
      );

      const dir = ckptDir("default");
      const files = readdirSync(dir);
      expect(files.length).toBe(1);
      expect(readFileSync(join(dir, files[0]), "utf8")).toBe("TARGET-CONTENT");
    });

    it("never descends into, nor tracks, a symlinked directory", async () => {
      // The real directory lives OUTSIDE taskDir -- reachable only through
      // the symlink below -- so this actually exercises "never descended
      // into a symlinked dir," not just "a real subdirectory got walked."
      const realDir = join(home, "elsewhere");
      mkdirSync(realDir);
      writeFileSync(join(realDir, "secret.txt"), "SECRET");
      symlinkSync(realDir, join(taskDir, "dirlink"), "dir");

      const h = setup();
      await h.session_start(
        {},
        { sessionManager: { getSessionFile: () => undefined }, cwd: taskDir, ui: { notify: vi.fn() } },
      );

      const dir = ckptDir("default");
      const contents = existsSync(dir)
        ? readdirSync(dir).map((f) => readFileSync(join(dir, f), "utf8"))
        : [];
      expect(contents).not.toContain("SECRET");
    });
  });

  // 7. Budgets: entries, bytes, and depth caps each bound the walk; shallow
  // ordering means a top-level-only entry budget still captures every
  // top-level file before any nested one.
  describe("budgets", () => {
    it("a tiny entries cap stops the walk: only entries up to the cap get processed", async () => {
      process.env.LITTLE_CODER_CHECKPOINT_SNAPSHOT_MAX_ENTRIES = "2";
      const h = setup();
      for (const n of ["a.txt", "b.txt", "c.txt", "d.txt"]) writeFileSync(join(taskDir, n), n);

      await h.session_start(
        {},
        { sessionManager: { getSessionFile: () => undefined }, cwd: taskDir, ui: { notify: vi.fn() } },
      );

      const dir = ckptDir("default");
      expect(readdirSync(dir).length).toBe(2);
    });

    it("a tiny byte cap stops copying after the file that crosses it", async () => {
      process.env.LITTLE_CODER_CHECKPOINT_SNAPSHOT_MAX_BYTES = "5";
      const h = setup();
      writeFileSync(join(taskDir, "a.txt"), "AAAAAAAAAA"); // 10 bytes: alone over the 5-byte cap
      writeFileSync(join(taskDir, "b.txt"), "BBBBBBBBBB");

      await h.session_start(
        {},
        { sessionManager: { getSessionFile: () => undefined }, cwd: taskDir, ui: { notify: vi.fn() } },
      );

      // The first file's own bytes may overshoot the cap by up to one file
      // (accepted, per design, over a second stat) -- but the walk must
      // stop before a second file is copied.
      const dir = ckptDir("default");
      expect(readdirSync(dir).length).toBe(1);
    });

    it("a depth cap skips files below it", async () => {
      process.env.LITTLE_CODER_CHECKPOINT_SNAPSHOT_MAX_DEPTH = "1";
      const h = setup();
      const nested = join(taskDir, "sub");
      mkdirSync(nested);
      writeFileSync(join(taskDir, "top.txt"), "TOP");
      writeFileSync(join(nested, "deep.txt"), "DEEP");

      await h.session_start(
        {},
        { sessionManager: { getSessionFile: () => undefined }, cwd: taskDir, ui: { notify: vi.fn() } },
      );

      const dir = ckptDir("default");
      const contents = readdirSync(dir).map((f) => readFileSync(join(dir, f), "utf8"));
      expect(contents).toContain("TOP");
      expect(contents).not.toContain("DEEP");
    });

    it("shallow-first: a top-level-only entry budget still backs up every top-level file, never a nested one", async () => {
      const nested = join(taskDir, "sub");
      mkdirSync(nested);
      writeFileSync(join(taskDir, "top1.txt"), "TOP1");
      writeFileSync(join(taskDir, "top2.txt"), "TOP2");
      writeFileSync(join(nested, "deep.txt"), "DEEP");
      // Top level has exactly 3 dirents: top1.txt, top2.txt, sub/.
      process.env.LITTLE_CODER_CHECKPOINT_SNAPSHOT_MAX_ENTRIES = "3";

      const h = setup();
      await h.session_start(
        {},
        { sessionManager: { getSessionFile: () => undefined }, cwd: taskDir, ui: { notify: vi.fn() } },
      );

      const dir = ckptDir("default");
      const contents = readdirSync(dir).map((f) => readFileSync(join(dir, f), "utf8"));
      expect(contents.sort()).toEqual(["TOP1", "TOP2"]);
    });
  });

  // 8. Skip-list and oversize passthrough.
  describe("skip-list and oversize passthrough", () => {
    it("never descends into a skip-listed directory (.git, node_modules, __pycache__, .venv, venv)", async () => {
      const h = setup();
      for (const skipped of [".git", "node_modules", "__pycache__", ".venv", "venv"]) {
        const d = join(taskDir, skipped);
        mkdirSync(d);
        writeFileSync(join(d, "f.txt"), `INSIDE-${skipped}`);
      }
      writeFileSync(join(taskDir, "real.txt"), "REAL");

      await h.session_start(
        {},
        { sessionManager: { getSessionFile: () => undefined }, cwd: taskDir, ui: { notify: vi.fn() } },
      );

      const dir = ckptDir("default");
      const contents = readdirSync(dir).map((f) => readFileSync(join(dir, f), "utf8"));
      expect(contents).toEqual(["REAL"]);
    });

    it("writes a .toolarge sentinel (no full copy) for an oversize file found by the walk", async () => {
      const h = setup();
      const big = join(taskDir, "big.bin");
      writeFileSync(big, Buffer.alloc(MAX_BACKUP_BYTES + 1, 1));

      await h.session_start(
        {},
        { sessionManager: { getSessionFile: () => undefined }, cwd: taskDir, ui: { notify: vi.fn() } },
      );

      const dir = ckptDir("default");
      const expectedSentinel = big.replace(/[^A-Za-z0-9._-]/g, "_").slice(-200) + ".toolarge";
      expect(readdirSync(dir)).toEqual([expectedSentinel]);
    });
  });

  // 9. Full existing suite as regression: covered by the describe blocks
  // above this one in the file, whose beforeEach now deletes
  // LITTLE_CODER_BENCHMARK / LITTLE_CODER_SESSION_ID so an ambient var on a
  // benchmark box can't flake them, and whose exact entry-count assertions
  // (e.g. `readdirSync(dir).length` toBe(1)) are exactly why the
  // `!== "gaia"` gate must keep the walk inert for every one of them.
});
