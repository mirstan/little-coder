// The `__LC_TB_SHELL__` ui-proxy channel, shared by every extension that needs
// to run a command inside the Terminal-Bench container.
//
// pi runs on the HOST; the task container is reachable only through the Python
// adapter, which intercepts `extension_ui_request` titles carrying this prefix
// and answers with the formatted command output (rpc_client._handle_ui_request
// → little_coder_agent.tb_shell_handler → proxy.run → env.exec). ShellSession
// owns the model-facing tool; syntax-check issues its own checks over the same
// channel, so the wire format lives here rather than in either of them.
//
// This dir intentionally has no `index.ts` — see _shared/intervention.ts.

export const TB_MODE_ENV = "LITTLE_CODER_TB_MODE";
export const TB_PROXY_PREFIX = "__LC_TB_SHELL__:";

// Structurally typed so callers can pass pi's ctx (tool or event handler) and
// tests can pass a stub.
export interface ProxyUiCtx {
  ui: { input(title: string, initial?: string): Promise<unknown> };
}

export function inTbMode(): boolean {
  return process.env[TB_MODE_ENV] === "1";
}

/** The session key the adapter routes commands by. */
export function tbSessionId(): string {
  return process.env.LITTLE_CODER_SESSION_ID || "default";
}

/**
 * Run `command` in the container, returning the adapter's formatted output.
 *
 * `null` (rather than a formatted error string) when no usable response came
 * back: callers disagree about what that should look like — ShellSession owes
 * the model a result with a footer, syntax-check owes it silence.
 */
export async function tbProxyRun(
  ctx: ProxyUiCtx,
  command: string,
  timeoutSec: number,
  sessionId: string,
): Promise<string | null> {
  const payload = { op: "run", session_id: sessionId, command, timeout: timeoutSec };
  const response = await ctx.ui.input(TB_PROXY_PREFIX + JSON.stringify(payload), "");
  return typeof response === "string" ? response : null;
}
