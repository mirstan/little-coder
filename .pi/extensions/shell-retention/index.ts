import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "@sinclair/typebox";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { envNumber } from "../_shared/env-number.ts";
import { byteLen } from "../shell-session/helpers.ts";
import {
  demoteMessages,
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
const paths = new Map<string, string>();
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

const hostArchive: RetentionArchive = {
  save(id, text) {
    if (paths.has(id)) return true;
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
    paths.set(id, path);
    return true;
  },
  get(id) {
    const path = paths.get(id);
    if (!path) return undefined;
    try {
      return readFileSync(path, "utf-8");
    } catch {
      return undefined;
    }
  },
};

export default function (pi: ExtensionAPI) {
  if (process.env.LITTLE_CODER_NO_SHELL_RETENTION === "1") return;

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
        hostArchive.get(id),
        params.offset as number | undefined,
        params.bytes as number | undefined,
        resolveRecallOptions(),
      );
      return { content: [{ type: "text", text: out.text }], details: {}, isError: out.isError };
    },
  });
}
