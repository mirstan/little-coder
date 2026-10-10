import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { execSync } from "node:child_process";
import { harnessIntervention } from "../_shared/intervention.ts";
import { inTbMode, tbProxyRun, tbSessionId, type ProxyUiCtx } from "../_shared/tb-proxy.ts";
import { checkpointPath, resolveShellTarget } from "../checkpoint/index.ts";
import {
  CHECK_TIMEOUT_SEC,
  type CheckResult,
  type CheckTarget,
  buildCheckCommand,
  checkTargets,
  checkerFor,
  classifyCheckOutput,
  footerCwd,
  formatDiagnostic,
  insertAboveFooter,
  readCheckResult,
  syntaxCheckEnabled,
} from "./helpers.ts";
import { splitFooter } from "../truncated-view/truncation.ts";

// Harness intervention: tell the model its just-written file does not parse,
// in the result of the write itself.
//
// A one-shot check over the existing shell proxy, not an LSP client: pi ships
// no LSP machinery, and under TB the files live in an ephemeral container pi
// can reach only through that proxy. The checkers are free — a container that
// receives a `.pl` already has `perl`, because the model is about to run it.
//
// Why `tool_result`: the diagnostic has to reach the model attached to the
// result it describes. A `tool_call` handler can only block, and a
// `sendUserMessage` steer detaches the diagnostic from its tool call.
//
// Cost discipline: nothing is appended on success, the check is skipped
// entirely when the same command already runs the file, and every failure mode
// (no checker, no file, proxy error, timeout) is silence.

const SHELL_TOOLS = new Set(["ShellSession", "bash", "Bash"]);
const WRITE_TOOLS = new Set(["write", "Write", "edit", "Edit"]);

// Per-process, so one missing-checker round trip is paid at most once a
// session rather than once per file.
const missingCheckers = new Set<string>();

/** Exported so tests can reach in — nothing in a real session clears this. */
export function resetMissingCheckers(): void {
  missingCheckers.clear();
}

type TextOrImage = { type: string; text?: string };

function resultText(content: TextOrImage[]): string | null {
  if (content.length === 0) return null;
  // An image result has no command output to annotate.
  if (content.some((c) => c.type !== "text")) return null;
  return content.map((c) => c.text ?? "").join("");
}

function runCheckLocal(command: string): CheckResult {
  try {
    const out = execSync(command, {
      shell: "/bin/bash",
      timeout: CHECK_TIMEOUT_SEC * 1000,
      encoding: "utf-8",
      maxBuffer: 1024 * 1024,
    });
    return classifyCheckOutput(0, String(out));
  } catch (err: any) {
    if (err.code === "ETIMEDOUT" || err.signal === "SIGTERM") return { kind: "none" };
    if (typeof err.status !== "number") return { kind: "none" };
    const output = (err.stdout?.toString?.() ?? "") + (err.stderr?.toString?.() ?? "");
    return classifyCheckOutput(err.status, output);
  }
}

async function runCheck(ctx: ProxyUiCtx, command: string): Promise<CheckResult> {
  try {
    if (!inTbMode()) return runCheckLocal(command);
    const text = await tbProxyRun(ctx, command, CHECK_TIMEOUT_SEC, tbSessionId());
    return text === null ? { kind: "none" } : readCheckResult(text);
  } catch {
    return { kind: "none" };
  }
}

/** The diagnostics for `targets`, empty when everything parses. */
async function diagnose(ctx: ProxyUiCtx, targets: CheckTarget[]): Promise<string[]> {
  const blocks: string[] = [];
  for (const target of targets) {
    if (missingCheckers.has(target.checker.key)) continue;
    const result = await runCheck(ctx, buildCheckCommand(target.path, target.checker));
    if (result.kind === "skip") {
      missingCheckers.add(target.checker.key);
      continue;
    }
    if (result.kind === "none" || result.exit === 0) continue;
    blocks.push(
      formatDiagnostic({
        label: target.checker.label,
        path: target.path,
        exit: result.exit,
        output: result.output,
      }),
    );
  }
  return blocks;
}

function report(ctx: any, text: string, blocks: string[], count: number) {
  harnessIntervention(
    ctx,
    `the model just wrote ${count === 1 ? "a file" : `${count} files`} that fail to parse — ` +
      "appended the checker's own error to the tool result.",
  );
  return { content: [{ type: "text" as const, text: insertAboveFooter(text, blocks.join("\n")) }] };
}

export default function (pi: ExtensionAPI) {
  pi.on("tool_result", async (event, ctx) => {
    const e = event as any;
    if (!syntaxCheckEnabled() || e.isError) return;
    if (!SHELL_TOOLS.has(String(e.toolName ?? ""))) return;

    const cmd = String(e.input?.command ?? "");
    if (!cmd) return;
    const text = resultText((e.content ?? []) as TextOrImage[]);
    if (text === null) return;

    // The footer's cwd is the container shell's; ctx.cwd is pi's host cwd,
    // which resolves a relative write path to a file that isn't there.
    const cwd = footerCwd(splitFooter(text).footer) ?? String(ctx.cwd ?? process.cwd());
    const targets = checkTargets(cmd, cwd);
    if (targets.length === 0) return;

    const blocks = await diagnose(ctx as ProxyUiCtx, targets);
    if (blocks.length === 0) return;
    return report(ctx, text, blocks, blocks.length);
  });

  // pi's own write/edit tools, which exist only outside TB mode (Aider
  // Polyglot and local runs) — inert under TB, where the model has no file
  // tools at all.
  pi.on("tool_result", async (event, ctx) => {
    const e = event as any;
    if (!syntaxCheckEnabled() || e.isError) return;
    if (!WRITE_TOOLS.has(String(e.toolName ?? ""))) return;

    const raw = checkpointPath((e.input ?? {}) as Record<string, unknown>);
    if (!raw) return;
    const checker = checkerFor(raw);
    if (!checker) return;
    const text = resultText((e.content ?? []) as TextOrImage[]);
    if (text === null) return;

    const path = resolveShellTarget(raw, String(ctx.cwd ?? process.cwd()));
    const blocks = await diagnose(ctx as ProxyUiCtx, [{ raw, path, checker }]);
    if (blocks.length === 0) return;
    return report(ctx, text, blocks, blocks.length);
  });
}
