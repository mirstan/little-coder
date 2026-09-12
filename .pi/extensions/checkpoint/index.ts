import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { isAbsolute, join } from "node:path";
import { homedir } from "node:os";
import { SHELL_TOOLS, detectWriteTargets } from "../_shared/shell-write.ts";

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
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  return dir;
}

function safeName(filePath: string): string {
  return filePath.replace(/[^A-Za-z0-9._-]/g, "_").slice(-200);
}

function backupIfNeeded(sessionId: string, filePath: string): void {
  if (!sessionId || !filePath) return;
  let session = tracked.get(sessionId);
  if (!session) {
    session = new Set();
    tracked.set(sessionId, session);
  }
  if (session.has(filePath)) return;
  session.add(filePath);
  try {
    if (existsSync(filePath)) {
      const content = readFileSync(filePath);
      writeFileSync(join(checkpointDir(sessionId), safeName(filePath)), content);
    } else {
      // Sentinel: file didn't exist before modification
      writeFileSync(
        join(checkpointDir(sessionId), safeName(filePath) + ".absent"),
        "",
      );
    }
  } catch {
    // Silent — checkpointing is best-effort
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
      const filePath = checkpointPath(input ?? {});
      if (filePath) {
        backupIfNeeded(currentSessionId, filePath);
      }
      return;
    }

    // Issue: a model's own buggy shell script overwrote a task file in place
    // with garbage, and no backup existed anywhere — the net above only
    // watches the write/edit tools, not a shell command that redirects output
    // to a file (`cat > f`, `tee f`, `dd of=f`, …). Reuse the same detector
    // permission-gate/write-guard already rely on for this exact parsing job
    // (issue #70) so a shell-redirect write gets the same pre-write snapshot.
    //
    // Two known limitations of this net:
    // - `detectWriteTargets` only understands shell-redirection syntax; it
    //   cannot see a program's own internal file writes (e.g. a Python/C
    //   script's `open(path).write(...)`). This closes the shell-redirect-
    //   shaped gap, not a general backstop against every way a subprocess can
    //   modify a file.
    // - `ShellSession`'s persistent working directory (tracked internally by
    //   that tool across calls) isn't visible from this event, so a relative
    //   path after an earlier `cd` may resolve against the wrong base here.
    //   This is a deliberate gap, not an oversight: shadow-tracking `cd` to
    //   compensate is the same rabbit hole write-guard chose to stay out of,
    //   for uncertain benefit. Resolution below is best-effort against
    //   `ctx.cwd`; if the resolved path doesn't exist, the `.absent` sentinel
    //   in `backupIfNeeded` already handles that gracefully.
    if (SHELL_TOOLS.has(name)) {
      const command = typeof input?.command === "string" ? input.command : undefined;
      if (!command) return;
      const cwd = ctx?.cwd ?? process.cwd();
      for (const write of detectWriteTargets(command)) {
        const resolved = isAbsolute(write.path) ? write.path : join(cwd, write.path);
        backupIfNeeded(currentSessionId, resolved);
      }
    }
  });
}
