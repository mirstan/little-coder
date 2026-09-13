import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { isAbsolute, join } from "node:path";
import { homedir } from "node:os";
import { SHELL_TOOLS, detectDeliverableWrites } from "../_shared/shell-write.ts";
import { normalizeWritePath } from "../write-guard/index.ts";

// Port of checkpoint/hooks.py. Snapshots a file's contents before a Write
// or Edit tool modifies it. First-write-wins per session (don't re-backup
// a file already tracked this session). Backups land in
// ~/.little-coder/checkpoints/<session>/.

export const tracked = new Map<string, Set<string>>(); // sessionId -> absolute paths

// Read whichever key carries the destination path. pi's built-in `write`/`edit`
// use `path`; older little-coder builds and some prompts use `file_path`. We
// accept both so the pre-edit backup fires regardless of which write
// implementation is in play — write-guard and read-guard-edit already do this,
// and keying only on `file_path` meant the snapshot silently never ran for the
// current pi tools (whose input uses `path`).
export function checkpointPath(input: Record<string, unknown>): string | undefined {
  if (typeof input?.path === "string") return input.path;
  if (typeof input?.file_path === "string") return input.file_path;
  return undefined;
}

function checkpointDir(sessionId: string): string {
  const dir = join(homedir(), ".little-coder", "checkpoints", sessionId);
  // Unconditional and idempotent — no existsSync memoization here. Memoizing
  // "dir exists" would permanently swallow every future backup this session
  // if the directory were ever deleted mid-session (see the checkpoint-dir
  // resilience test).
  mkdirSync(dir, { recursive: true });
  return dir;
}

function safeName(filePath: string): string {
  return filePath.replace(/[^A-Za-z0-9._-]/g, "_").slice(-200);
}

// Local to checkpoint: the shell expands `~/` itself and writes bare-absolute
// paths (e.g. `> /foo.md`) verbatim -- no write-guard-style cwd rewrite happens
// for shell redirection, so fidelity to the file the shell will actually touch
// beats key-uniformity with the write branch here. A write-tool call and a
// shell redirect that both "look like" the same bare-absolute path are
// legitimately DIFFERENT physical files today (write-guard rewrites one into
// cwd; the shell does not) -- divergent backup keys for this shape are correct.
export function resolveShellTarget(p: string, cwd: string): string {
  let out = p;
  if (out === "~") out = homedir();
  else if (out.startsWith("~/")) out = join(homedir(), out.slice(2));
  return isAbsolute(out) ? out : join(cwd, out);
}

// Checkpoint-only filter (not part of _shared/shell-write.ts): a detected
// target that still contains unexpanded shell syntax means the real path is
// unknowable statically — backing up a literal `$OUT`-named key would be
// noise, not a useful checkpoint. This is deliberately NOT added inside
// detectDeliverableWrites itself: tb-finalize-guard (the function's other
// consumer) legitimately treats a `$VAR` target as evidence-of-work and must
// not lose that signal.
const DYNAMIC_TARGET = /[$`*?[{]/;

export const MAX_BACKUP_BYTES = 10 * 1024 * 1024; // exported for tests

function backupIfNeeded(sessionId: string, filePath: string): void {
  if (!sessionId || !filePath) return;
  let session = tracked.get(sessionId);
  if (!session) { session = new Set(); tracked.set(sessionId, session); }
  if (session.has(filePath)) return;
  try {
    let st;
    try {
      st = statSync(filePath);
    } catch (e) {
      if ((e as NodeJS.ErrnoException)?.code === "ENOENT") {
        writeFileSync(join(checkpointDir(sessionId), safeName(filePath) + ".absent"), "");
        session.add(filePath);
      }
      return; // any other stat failure: leave untracked so a later write can retry
    }
    if (!st.isFile()) return; // directory/fifo/socket: skip, do NOT track
    if (st.size > MAX_BACKUP_BYTES) {
      writeFileSync(join(checkpointDir(sessionId), safeName(filePath) + ".toolarge"), String(st.size));
      session.add(filePath); // tracked forever: never retry (a later smaller version is a mid-session intermediate, not the original)
      return;
    }
    writeFileSync(join(checkpointDir(sessionId), safeName(filePath)), readFileSync(filePath));
    session.add(filePath); // only after the backup actually succeeded
  } catch {
    // best-effort; deliberately NOT tracked so a failed copy doesn't suppress a retry
  }
}

export default function (pi: ExtensionAPI) {
  let currentSessionId = "default";

  pi.on("session_start", async (_event, ctx) => {
    currentSessionId = ctx.sessionManager.getSessionFile()?.split("/").pop() ?? "default";
  });

  pi.on("tool_call", async (event, ctx) => {
    const name = (event as any).toolName;
    const input: any = (event as any).input ?? (event as any).args;

    if (name === "write" || name === "Write" || name === "edit" || name === "Edit") {
      const raw = checkpointPath(input ?? {});
      if (raw) {
        const cwd = ctx?.cwd ?? process.cwd();
        backupIfNeeded(currentSessionId, normalizeWritePath(raw, cwd).path);
      }
      return;
    }

    // Issue: a model's own buggy shell script overwrote a task file in place
    // with garbage, and no backup existed anywhere — the net above only
    // watches the write/edit tools, not a shell command that mutates a file
    // (`cat > f`, `tee f`, `dd of=f`, `cp`/`mv`/`sed -i`, …). Reuse
    // detectDeliverableWrites (tb-finalize-guard's broader detector, exported
    // from _shared/shell-write.ts) so this net also catches cp/mv/install/
    // sed -i/a compiler's -o, which the narrower detectWriteTargets (used by
    // write-guard/permission-gate for actual write-permission gating) misses
    // by design. checkpoint is a non-gating, best-effort consumer, so
    // over-detection here is acceptable in a way it isn't for those two.
    //
    // ShellSend (bg-shell) is routed through the same detection here, locally
    // — it is deliberately NOT added to the shared SHELL_TOOLS set (two
    // security gates rely on that set's current membership), but its `text`
    // payload writes to a running job's stdin exactly like a command writes
    // to a shell, so it deserves the same pre-write snapshot.
    //
    // Known limitations of this net:
    // - The detector only understands shell syntax; it cannot see a
    //   program's own internal file writes (e.g. a Python/C script's
    //   `open(path).write(...)`). This closes the shell-command-shaped gap,
    //   not a general backstop against every way a subprocess can modify a
    //   file.
    // - A target containing unexpanded shell syntax (`$OUT`, `$DEST/b.txt`,
    //   a glob, …) is skipped: the real path is unknowable statically, so
    //   backing up the literal placeholder text would be noise, not a
    //   useful checkpoint.
    // - BSD `sed -i '' 's/a/b/' file` misreports the script text itself as a
    //   target (no dynamic chars, so the filter above doesn't catch it) —
    //   pre-existing shared-module behavior; produces one harmless bogus
    //   `.absent` sentinel. Not fixed here.
    // - `ShellSession`'s persistent working directory (tracked internally by
    //   that tool across calls) isn't visible from this event, so a relative
    //   path after an earlier `cd` may resolve against the wrong base here.
    //   This is a deliberate gap, not an oversight: shadow-tracking `cd` to
    //   compensate is the same rabbit hole write-guard chose to stay out of,
    //   for uncertain benefit. Resolution below is best-effort against
    //   `ctx.cwd`; if the resolved path doesn't exist, the `.absent` sentinel
    //   in `backupIfNeeded` already handles that gracefully.
    const command =
      SHELL_TOOLS.has(name) && typeof input?.command === "string" ? input.command
      : name === "ShellSend" && typeof input?.text === "string" ? input.text
      : undefined;
    if (!command) return;
    const cwd = ctx?.cwd ?? process.cwd();
    for (const write of detectDeliverableWrites(command)) {
      if (DYNAMIC_TARGET.test(write.path)) continue;
      backupIfNeeded(currentSessionId, resolveShellTarget(write.path, cwd));
    }
  });
}
