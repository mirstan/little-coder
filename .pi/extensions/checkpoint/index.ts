import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { Dirent } from "node:fs";
import { lstatSync, mkdirSync, opendirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { basename, join, resolve } from "node:path";
import { homedir } from "node:os";
import { SHELL_TOOLS, detectDeliverableWrites } from "../_shared/shell-write.ts";
import { normalizeWritePath } from "../write-guard/index.ts";
import { envNumber } from "../_shared/env-number.ts";
import { harnessIntervention, type InterventionCtx } from "../_shared/intervention.ts";
import { DEFAULT_HEAVY_DIRS } from "../extra-tools/glob.ts";

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

// A short hash of the full path, not just the flattened+truncated tail,
// disambiguates two different paths that happen to share their last 200
// (post-flattening) characters -- e.g. two files at the same depth under
// differently-named long parent directories. Without it the second such
// backup silently overwrites the first's bytes on disk: a real, if narrow,
// risk once a single walk can touch hundreds of files in one pass, rather
// than the one-or-two files a normal write/edit session ever backs up.
function safeName(filePath: string): string {
  const hash = createHash("sha1").update(filePath).digest("hex").slice(0, 8);
  const flat = filePath.replace(/[^A-Za-z0-9._-]/g, "_").slice(-190);
  return `${flat}.${hash}`;
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
  // `resolve` (not a bare isAbsolute/join branch) so `/a/./f`, `/a//f`, and
  // `/a/x/../f` all collapse to the same key as their canonical spelling —
  // otherwise two shell redirects to the "same" file under different
  // spellings double-track it, and first-write-wins captures whichever
  // spelling happened to run second, not the true original.
  return resolve(cwd, out);
}

// Checkpoint-only filter (not part of _shared/shell-write.ts): `$`/backtick
// mean genuine shell substitution — the real path is unknowable statically,
// so backing up a literal `$OUT`-named key would be noise, not a useful
// checkpoint. Deliberately NOT added inside detectDeliverableWrites itself:
// tb-finalize-guard (the function's other consumer) legitimately treats a
// `$VAR` target as evidence-of-work and must not lose that signal.
const SUBSTITUTION_TARGET = /[$`]/;

// `[`, `*`, `?`, `{` are unexpanded-glob shapes in MOST contexts, but they
// are also ordinary, common filename characters in web-framework routing
// conventions (`routes/[id].tsx`, `app/[...slug]/page.tsx` — Next.js, Remix,
// SvelteKit). Treating them as unconditionally dynamic silently excludes
// exactly the kind of file little-coder's small-model target audience edits
// constantly. So: only skip when the literal, resolved path does NOT exist —
// a real file on disk is never "unknowable," bracket in its name or not.
const GLOB_LIKE_TARGET = /[*?[{]/;

function isUnbackupable(rawPath: string, cwd: string): boolean {
  if (SUBSTITUTION_TARGET.test(rawPath)) return true;
  if (!GLOB_LIKE_TARGET.test(rawPath)) return false;
  // Existence alone, not specifically isFile: a glob-shaped DIRECTORY
  // destination (e.g. `output/[locale]/`) must still reach
  // shellWriteTargets' own directory-expansion logic rather than being
  // rejected here before it gets the chance.
  try {
    statSync(resolveShellTarget(rawPath, cwd));
    return false;
  } catch {
    return true; // doesn't exist (or unreadable parent) -- still unknowable
  }
}

export const MAX_BACKUP_BYTES = 10 * 1024 * 1024; // exported for tests

function isExistingDir(p: string): boolean {
  try {
    return statSync(p).isDirectory();
  } catch {
    return false;
  }
}

/**
 * Every path a detected shell write can clobber, resolved against `cwd`.
 *
 * `cp a.md docs/` names `docs` as its target, but the bytes it destroys are
 * `docs/a.md`. `backupIfNeeded` correctly refuses to snapshot a directory, so
 * keying only on the target would silently back up *nothing* for the single
 * most common copy shape a model writes. When the destination is a directory
 * on disk, each source contributes `<dir>/<basename(source)>` instead — which
 * is what `cp`/`mv`/`install` actually overwrite there.
 */
export function shellWriteTargets(
  write: { path: string; sources?: string[] },
  cwd: string,
): string[] {
  const dest = resolveShellTarget(write.path, cwd);
  if (!write.sources?.length || !isExistingDir(dest)) return [dest];
  // A directory destination is never itself the clobbered file, so it is not
  // in the returned list — only the per-source paths landing inside it.
  return write.sources
    .filter((src) => src && !isUnbackupable(src, cwd))
    .map((src) => join(dest, basename(src)));
}

// Return value (added for the session-start walk's byte budget, below):
// content bytes actually written on a real copy, `undefined` for a
// sentinel/skip/already-tracked/failure. Additive -- the two existing
// call sites (tool_call's write/edit and shell-write branches) still
// call this as a bare statement and ignore it.
function backupIfNeeded(sessionId: string, filePath: string): number | undefined {
  if (!sessionId || !filePath) return undefined;
  let session = tracked.get(sessionId);
  if (!session) { session = new Set(); tracked.set(sessionId, session); }
  if (session.has(filePath)) return undefined;
  try {
    let st;
    try {
      st = statSync(filePath);
    } catch (e) {
      if ((e as NodeJS.ErrnoException)?.code === "ENOENT") {
        writeFileSync(join(checkpointDir(sessionId), safeName(filePath) + ".absent"), "");
        session.add(filePath);
      }
      return undefined; // any other stat failure: leave untracked so a later write can retry
    }
    if (!st.isFile()) return undefined; // directory/fifo/socket: skip, do NOT track
    if (st.size > MAX_BACKUP_BYTES) {
      writeFileSync(join(checkpointDir(sessionId), safeName(filePath) + ".toolarge"), String(st.size));
      session.add(filePath); // tracked forever: never retry (a later smaller version is a mid-session intermediate, not the original)
      return undefined;
    }
    const content = readFileSync(filePath);
    writeFileSync(join(checkpointDir(sessionId), safeName(filePath)), content);
    session.add(filePath); // only after the backup actually succeeded
    return content.length;
  } catch {
    // best-effort; deliberately NOT tracked so a failed copy doesn't suppress a retry
    return undefined;
  }
}

// ---------------------------------------------------------------------------
// GAIA session-start snapshot
//
// gaia.py stages at most one attachment into a fresh host-side temp dir and
// launches pi with `cwd` pointing there (gaia.py:138-172) -- the one
// benchmark shape where this extension's own `fs` calls can see the task's
// files before the model ever acts. Harbor/TB's task files live inside a
// Docker container this host-side process has no path to; that case is
// handled adapter-side instead (see the comment on the session_start
// handler below).
//
// Budgets, re-derived for GAIA's actual task shape (0-1 files, flat at
// session start): these are guardrails expected to never bind on a real
// GAIA task -- they exist to bound the blast radius of a future adapter
// change (a new allowlist entry, a staging change that unpacks archives, a
// misconfigured cwd), not to ration a known-large tree.
const SNAPSHOT_MAX_ENTRIES_ENV = "LITTLE_CODER_CHECKPOINT_SNAPSHOT_MAX_ENTRIES";
const SNAPSHOT_MAX_BYTES_ENV = "LITTLE_CODER_CHECKPOINT_SNAPSHOT_MAX_BYTES";
const SNAPSHOT_MAX_DEPTH_ENV = "LITTLE_CODER_CHECKPOINT_SNAPSHOT_MAX_DEPTH";

const SNAPSHOT_MAX_ENTRIES = 500; // dirents processed (dirs included); <= 0 disables the walk
const SNAPSHOT_MAX_BYTES = 50 * 1024 * 1024; // cumulative content bytes copied
const SNAPSHOT_MAX_DEPTH = 4; // levels below cwd (cwd's own entries = depth 1)

// Reuses extra-tools/glob.ts's own heavy-dir list rather than growing a
// second, narrower one that only some future edit remembers to keep in
// sync -- the exact "two lists drift apart" risk this file's own
// SHELL_TOOLS comment already warns about for a different set. Everything
// on that list is regenerable or a macOS/system directory that would never
// legitimately be a task's own reference input, so treating it the same
// way here (never an irreplaceable original) is consistent with its
// existing rationale, not a new one.
const SNAPSHOT_SKIP_DIRS = DEFAULT_HEAVY_DIRS;

// readdir dirents don't follow symlinks for their `d_type`-derived checks,
// so `dirent.isDirectory()` alone already excludes symlinked directories
// (they report `isSymbolicLink()` instead). The only real ambiguity is
// DT_UNKNOWN (some filesystems/platforms never populate `d_type`), where
// every `is*()` check comes back false -- one `lstatSync` settles it.
// `lstat`, not `stat`, so a symlinked directory found this way is also
// correctly seen as NOT a real directory to descend into; it falls through
// to `backupIfNeeded` below instead, same as the `isSymbolicLink()` case.
function isRealDir(dirent: Dirent, fullPath: string): boolean {
  if (dirent.isSymbolicLink()) return false;
  if (dirent.isDirectory()) return true;
  if (dirent.isFile()) return false;
  try {
    return lstatSync(fullPath).isDirectory();
  } catch {
    return false;
  }
}

/**
 * Session-start walk of `cwd`, backing up every pre-existing file via the
 * same `backupIfNeeded` the per-write net uses -- same storage, same
 * `tracked` first-write-wins map, so a later write/edit/shell redirect to a
 * walked file is already tracked and the session-start original is what
 * survives.
 *
 * Manual iterative BFS -- one `opendirSync`/`readSync` stream per directory
 * -- never `readdirSync(dir, { recursive: true })`, which follows
 * symlinked directory loops (confirmed empirically: a one-level cycle
 * recursed 30+ levels before being killed). The queue is FIFO, so the walk
 * is shallow-first: if a budget runs out mid-walk, top-of-tree files --
 * where task inputs live -- were captured first.
 *
 * Exported (rather than kept module-local) so tests can drive it directly
 * as well as through the `session_start` handler.
 */
export function snapshotWorkingDirectory(
  sessionId: string,
  cwd: string | undefined,
  ctx: InterventionCtx | undefined,
): void {
  if (!sessionId || !cwd) return; // best-effort: ctx.cwd is non-optional per pi's types, but never trust that here
  try {
    const maxEntries = envNumber(SNAPSHOT_MAX_ENTRIES_ENV, SNAPSHOT_MAX_ENTRIES);
    if (maxEntries <= 0) return; // literal-0 (or negative) kill switch
    const maxBytes = envNumber(SNAPSHOT_MAX_BYTES_ENV, SNAPSHOT_MAX_BYTES);
    const maxDepth = envNumber(SNAPSHOT_MAX_DEPTH_ENV, SNAPSHOT_MAX_DEPTH);
    // Never descend into our own backup store, however cwd happens to be set.
    const excludeDir = resolve(join(homedir(), ".little-coder"));

    let entries = 0;
    let bytes = 0;
    let truncated: "entries" | "bytes" | undefined;
    const queue: Array<{ dir: string; depth: number }> = [{ dir: resolve(cwd), depth: 1 }];

    walk: while (queue.length > 0) {
      const { dir, depth } = queue.shift()!;
      // opendirSync + readSync, not readdirSync: readdirSync materializes
      // every dirent in the directory into one array before the loop below
      // ever gets a chance to check the entries budget, so a single
      // pathological directory (an unpacked archive, say) would still pay
      // for a full allocation + listing regardless of maxEntries. Streaming
      // one entry at a time makes the budget bound the actual read, not
      // just the processing after the read already happened.
      let handle;
      try {
        handle = opendirSync(dir);
      } catch {
        continue; // unreadable dir: best-effort, move on
      }
      try {
        let dirent: Dirent | null;
        while ((dirent = handle.readSync()) !== null) {
          if (entries >= maxEntries) { truncated = "entries"; break walk; }
          // Checked here, unconditionally, rather than only once a file
          // dirent turns up: once the byte budget is spent nothing more
          // will ever be copied, so there's no reason to keep dequeuing
          // and reading further directories just to discover more files
          // that would immediately be skipped anyway.
          if (bytes >= maxBytes) { truncated = "bytes"; break walk; }
          entries++;
          const full = join(dir, dirent.name);
          if (full === excludeDir) continue; // never snapshot our own backup store

          if (isRealDir(dirent, full)) {
            if (SNAPSHOT_SKIP_DIRS.has(dirent.name)) continue;
            if (depth < maxDepth) queue.push({ dir: full, depth: depth + 1 });
            continue;
          }

          // File, file symlink, or dir symlink: backupIfNeeded's own statSync
          // follows a symlink and only copies a real file (`isFile()`), so a
          // dir symlink is skipped untracked and a broken link throws into
          // its own catch -- no separate type logic needed here.
          const written = backupIfNeeded(sessionId, full);
          if (written) bytes += written;
        }
      } finally {
        handle.closeSync();
      }
    }

    // Budget exhaustion is visible, not silent; a complete walk stays quiet.
    // Guarded on `ctx?.ui` (Fix 2): pi's ExtensionContext declares `ui`
    // non-optional and gaia.py does forward notifications in practice, but
    // an uncaught throw here would be a session-killing failure mode for a
    // best-effort feature, and this runs under RPC mode -- the least-
    // typical ctx shape. The outer try/catch is the last line of defense
    // regardless.
    if (truncated && ctx?.ui) {
      const reason =
        truncated === "entries"
          ? `entry budget reached at ${maxEntries}`
          : `byte budget reached at ${maxBytes}`;
      harnessIntervention(
        ctx,
        `session-start snapshot of the task directory is partial (${reason}) -- files not yet walked have no pre-run backup.`,
      );
    }
  } catch {
    // best-effort, like every other checkpoint path -- never throw out of session_start
  }
}

export default function (pi: ExtensionAPI) {
  let currentSessionId = "default";

  pi.on("session_start", async (_event, ctx) => {
    // `||`, not `??`: an ambient *empty-string* LITTLE_CODER_SESSION_ID (a
    // wrapper script that exports the var without a value) must fall
    // through here, not become "" and silently disable every backup for
    // the session via backupIfNeeded's own `if (!sessionId || ...) return;`
    // guard. `||` matches the existing convention (evidence/index.ts,
    // browser/index.ts, shell-session/index.ts all key on
    // `LITTLE_CODER_SESSION_ID || "default"`) -- this conforms checkpoint
    // to a pattern three sibling extensions already follow, rather than
    // inventing a new one. rpc_client.py exports this var for every
    // benchmark caller that passes session_id (GAIA and Harbor both do),
    // so this also un-collides every benchmark's checkpoint namespace, not
    // just GAIA's -- interactive pi (no env var) keeps today's
    // derived-from-file behavior.
    // Scrubbed with the same char class safeName uses for backup filenames:
    // a value containing `/` or `..` used raw here would let checkpointDir's
    // join() resolve outside ~/.little-coder/checkpoints/ entirely. Not
    // reachable by an attacker today (every setter of this env var is the
    // harness itself), but a misconfigured wrapper or future caller
    // shouldn't be able to silently redirect or disable the whole backup
    // net -- replacing `/` leaves no real separator for a later join() to
    // walk, turning any such value into one inert, harmless segment name
    // instead of a path.
    currentSessionId = (
      process.env.LITTLE_CODER_SESSION_ID
      || ctx.sessionManager.getSessionFile()?.split("/").pop()
      || "default"
    ).replace(/[^A-Za-z0-9._-]/g, "_");

    // GAIA-only allowlist, not a truthy gate -- mirrors
    // gaia-finalize-guard's identical `!== "gaia"` early return.
    // benchmark="terminal_bench" runs with a *host-side* cwd (wherever the
    // harness was launched from, typically the little-coder repo root), so
    // a truthy gate here would walk and snapshot the repo itself. An
    // interactive session never sets this var, so a developer's real repo
    // is never walked.
    if (process.env.LITTLE_CODER_BENCHMARK !== "gaia") return;
    snapshotWorkingDirectory(currentSessionId, ctx?.cwd, ctx);
  });

  pi.on("tool_call", async (event, ctx) => {
    const name = (event as any).toolName;
    const input: any = (event as any).input ?? (event as any).args;
    // Case-folded, matching how write-guard and read-guard-edit select these
    // same two tools (`toolName.toLowerCase() !== "write"` / `"edit"`). The
    // enumerated spellings this replaces covered only `write`/`Write`/`edit`/
    // `Edit`, so any other casing was still guarded by those two but silently
    // never snapshotted. SHELL_TOOLS membership below stays case-sensitive:
    // that set is a shared security-gate list, matched verbatim by
    // permission-gate and write-guard.
    const lower = String(name ?? "").toLowerCase();

    if (lower === "write" || lower === "edit") {
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
    //   file. On GAIA, `snapshotWorkingDirectory`'s session-start walk
    //   (below) closes this specific gap by capturing the task's
    //   pre-existing files before the model ever acts, independent of
    //   shell-syntax detection entirely. On Harbor/Terminal-Bench this
    //   extension can't see the gap at all -- the task runs inside a
    //   Docker container this host-side process has no filesystem path to
    //   -- so that case is handled adapter-side instead, in
    //   benchmarks/harbor_adapter/little_coder_agent.py.
    // - Commands that destroy a file without *writing* one are out of scope:
    //   `rm`, `truncate`, `patch`, `tar -x`, `git checkout --`, `sort -o`.
    //   detectDeliverableWrites answers "did this produce a deliverable",
    //   and none of those do, so folding them in would change its meaning for
    //   tb-finalize-guard (a deletion is not evidence of work). Covering them
    //   needs a separate destructive-command detector, not a widening of this
    //   one.
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
      if (isUnbackupable(write.path, cwd)) continue;
      for (const target of shellWriteTargets(write, cwd)) {
        backupIfNeeded(currentSessionId, target);
      }
    }
  });
}
