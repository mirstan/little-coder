import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "@sinclair/typebox";
import { createHash } from "node:crypto";
import { closeSync, mkdtempSync, openSync, readSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { harnessIntervention } from "../_shared/intervention.ts";
import { envNumber } from "../_shared/env-number.ts";
import { byteLen } from "../shell-session/helpers.ts";
import {
  demoteMessages,
  findMarkerEchoIds,
  recallSlice,
  resolveOptions,
  resolveRecallOptions,
  type RetentionArchive,
} from "./retention.ts";

// Hook wiring, host archive, and the ShellRecall tool. The demotion rules live
// in retention.ts.
//
// The archive lives on the host rather than in the container the commands ran
// in: recall never round-trips through a backend, so one path serves the local
// subprocess, the TB tmux proxy, and the Harbor env proxy alike.
//
// Note for a future `context` hook added after this one in load order: the
// messages it receives already carry demoted commands and results.

const ENV_BUDGET = "LITTLE_CODER_SHELL_RETENTION_BUDGET_BYTES";
const DEFAULT_BUDGET = 256 * 1024 * 1024;

// Archived output is arbitrary command output — env dumps, credential files,
// private source — so mkdtemp's 0700 unguessable directory, not a predictable
// path in a shared /tmp. Process lifetime is one trial.
const entries = new Map<string, { path: string; sig: string }>();
let dir: string | null = null;
let dirUnavailable = false;
let bytesWritten = 0;

function ensureDir(): string | null {
  if (dir || dirUnavailable) return dir;
  try {
    dir = mkdtempSync(join(tmpdir(), "lc-retention-"));
  } catch {
    dirUnavailable = true;
  }
  return dir;
}

function signature(text: string): string {
  return createHash("sha256").update(text).digest("hex");
}

// bg-shell's session_shutdown removes its own job state the same way — a
// directory made for one session must not outlive it on a long-lived host.
function cleanupArchive(): void {
  if (dir) {
    try {
      rmSync(dir, { recursive: true, force: true });
    } catch {
      // Best-effort: an orphaned tmp dir is a disk-space nit, not a correctness
      // issue, and the process is exiting either way.
    }
  }
  entries.clear();
  dir = null;
  dirUnavailable = false;
  bytesWritten = 0;
}

const hostArchive: RetentionArchive = {
  save(id, text) {
    const sig = signature(text);
    const existing = entries.get(id);
    // A different signature under the same id means the id's hash collided
    // with a different pair's — refuse rather than let a later recall
    // silently return that other pair's content.
    if (existing) return existing.sig === sig;

    const bytes = byteLen(text);
    if (bytesWritten + bytes > envNumber(ENV_BUDGET, DEFAULT_BUDGET)) return false;
    const d = ensureDir();
    if (!d) return false;
    const path = join(d, `${id}.txt`);
    try {
      writeFileSync(path, text, { encoding: "utf-8", mode: 0o600 });
    } catch {
      return false;
    }
    bytesWritten += bytes;
    entries.set(id, { path, sig });
    return true;
  },
  size(id) {
    const path = entries.get(id)?.path;
    if (!path) return undefined;
    try {
      return statSync(path).size;
    } catch {
      return undefined;
    }
  },
  readRange(id, start, length) {
    const path = entries.get(id)?.path;
    if (!path || length <= 0) return undefined;
    let fd: number | undefined;
    try {
      fd = openSync(path, "r");
      const buf = Buffer.alloc(length);
      const read = readSync(fd, buf, 0, length, start);
      return buf.subarray(0, read);
    } catch {
      return undefined;
    } finally {
      if (fd !== undefined) {
        try {
          closeSync(fd);
        } catch {
          // Already the failure path for a read; nothing further to recover.
        }
      }
    }
  },
};

export default function (pi: ExtensionAPI) {
  if (process.env.LITTLE_CODER_NO_SHELL_RETENTION === "1") return;

  pi.on("session_shutdown", async () => {
    cleanupArchive();
  });

  // A model can echo a demoted placeholder verbatim into a NEW tool call —
  // e.g. "reconstructing" a file from the marker text left in its own
  // context — baking a corrupted line into real output instead of paging
  // the original back in with ShellRecall. Block those calls before they
  // execute; ShellRecall's own input legitimately carries a live id.
  pi.on("tool_call", async (event, ctx) => {
    if (String((event as any).toolName ?? "") === "ShellRecall") return;
    const input = (event as any).input;
    if (input == null) return;

    const liveIds = findMarkerEchoIds(input).filter((id) => hostArchive.size(id) !== undefined);
    if (liveIds.length === 0) return;

    harnessIntervention(ctx, "blocked a tool call echoing a demoted shell-retention placeholder.");
    return {
      block: true,
      reason:
        `This call quotes a shell-retention placeholder (${liveIds.join(", ")}) instead of ` +
        `real content — the marker text is not the original command/output, so copying it ` +
        `into a file or command reproduces the placeholder, not what it stands for.\n` +
        `\n` +
        `To get the original text back, call ShellRecall with that id.\n` +
        `\n` +
        `If you're instead trying to remove a placeholder already written into a file, don't ` +
        `quote the marker text in the command (that re-trips this same guard) — delete or ` +
        `replace it by line number instead, e.g. \`sed -i '42d' file\` or an edit targeted at ` +
        `that line.`,
    };
  });

  pi.on("context", async (event) => {
    const { messages, demotedCount } = demoteMessages(
      (event as any).messages || [],
      hostArchive,
      resolveOptions(),
    );
    if (demotedCount > 0) {
      return { messages };
    }
  });

  pi.registerTool({
    name: "ShellRecall",
    label: "ShellRecall",
    description:
      "Page back the full command and output of a shell observation that was demoted from " +
      "context. Use the sr-… id quoted in the demotion placeholder. Returns one slice; pass " +
      "offset to continue through a long one.",
    parameters: Type.Object({
      id: Type.String({ description: "sr-… id from a demotion placeholder" }),
      offset: Type.Optional(Type.Integer({ description: "Byte offset into the archived text (default 0)" })),
      bytes: Type.Optional(Type.Integer({ description: "How many bytes (default 8192, max 49152)" })),
    }),
    async execute(_id, params) {
      const id = String(params.id ?? "").trim();
      const out = recallSlice(
        id,
        hostArchive,
        params.offset as number | undefined,
        params.bytes as number | undefined,
        resolveRecallOptions(),
      );
      return { content: [{ type: "text", text: out.text }], details: {}, isError: out.isError };
    },
  });
}
