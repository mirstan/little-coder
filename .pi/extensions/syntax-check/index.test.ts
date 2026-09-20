import { describe, it, expect, beforeEach, afterEach } from "vitest";
import setupSyntaxCheck, { resetMissingCheckers } from "./index.ts";
import { SKIP_SENTINEL, SYNTAX_CHECK_ENV } from "./helpers.ts";
import { formatOutput } from "../shell-session/helpers.ts";
import { TB_PROXY_PREFIX } from "../_shared/tb-proxy.ts";
import { splitFooter } from "../truncated-view/truncation.ts";

type Handler = (event: any, ctx: any) => Promise<any>;

/** The two `tool_result` handlers, in registration order (shell, then write). */
function register(): Handler[] {
  const handlers: Handler[] = [];
  setupSyntaxCheck({
    on: (event: string, handler: Handler) => {
      if (event === "tool_result") handlers.push(handler);
    },
  } as any);
  return handlers;
}

/** A ctx whose proxy answers every check with `reply`, recording the commands. */
function stubCtx(reply: string) {
  const proxied: string[] = [];
  const notices: string[] = [];
  return {
    proxied,
    notices,
    cwd: "/host/pi",
    ui: {
      notify: (m: string) => notices.push(m),
      input: async (title: string) => {
        expect(title.startsWith(TB_PROXY_PREFIX)).toBe(true);
        proxied.push(JSON.parse(title.slice(TB_PROXY_PREFIX.length)).command);
        return reply;
      },
    },
  };
}

function shellResult(command: string, text: string) {
  return { toolName: "ShellSession", isError: false, input: { command }, content: [{ type: "text", text }] };
}

const WROTE_IT = formatOutput("", 0, "/app", false, "backend=harbor-env");
const PERL_FAILURE = formatOutput(
  'syntax error at /app/v3.pl line 25, near "$ctx]"\n/app/v3.pl had compilation errors.',
  255,
  "/app",
  false,
  "backend=harbor-env",
);

beforeEach(() => {
  process.env.LITTLE_CODER_TB_MODE = "1";
  resetMissingCheckers();
});
afterEach(() => {
  delete process.env.LITTLE_CODER_TB_MODE;
  delete process.env[SYNTAX_CHECK_ENV];
});

describe("shell handler", () => {
  const WRITE = "cat > /app/v3.pl <<'EOF'\n$ctxbits{$ctx]++;\nEOF";

  it("appends the checker's error above the footer", async () => {
    const [shell] = register();
    const ctx = stubCtx(PERL_FAILURE);
    const out = await shell(shellResult(WRITE, WROTE_IT), ctx);

    const text = out.content[0].text;
    expect(ctx.proxied[0]).toContain("perl -c");
    expect(text).toContain("[syntax-check] perl -c /app/v3.pl FAILED (exit 255)");
    expect(text).toContain('near "$ctx]"');
    // truncated-view runs after this handler and needs the footer last.
    expect(splitFooter(text).footer).not.toBeNull();
    expect(ctx.notices[0]).toContain("harness intervention:");
  });

  it("costs nothing when the file parses", async () => {
    const [shell] = register();
    const ctx = stubCtx(formatOutput("/app/v3.pl syntax OK", 0, "/app", false, ""));
    expect(await shell(shellResult(WRITE, WROTE_IT), ctx)).toBeUndefined();
    expect(ctx.notices).toEqual([]);
  });

  it("issues no check at all when the same command runs the file", async () => {
    const [shell] = register();
    const ctx = stubCtx(PERL_FAILURE);
    expect(await shell(shellResult(`${WRITE}\nperl /app/v3.pl`, WROTE_IT), ctx)).toBeUndefined();
    expect(ctx.proxied).toEqual([]);
  });

  it("stays silent when the checker is missing", async () => {
    const [shell] = register();
    const ctx = stubCtx(formatOutput(SKIP_SENTINEL, 0, "/app", false, ""));
    expect(await shell(shellResult(WRITE, WROTE_IT), ctx)).toBeUndefined();
  });

  it("pays the missing-checker round trip only once a session", async () => {
    const [shell] = register();
    const ctx = stubCtx(formatOutput(SKIP_SENTINEL, 0, "/app", false, ""));
    await shell(shellResult(WRITE, WROTE_IT), ctx);
    await shell(shellResult("cat > /app/w.pl <<'EOF'\nx\nEOF", WROTE_IT), ctx);
    expect(ctx.proxied).toHaveLength(1);
  });

  it("stays silent when the proxy answers with nothing usable", async () => {
    const [shell] = register();
    const ctx = stubCtx("the proxy fell over");
    expect(await shell(shellResult(WRITE, WROTE_IT), ctx)).toBeUndefined();
  });

  it("is disabled by the kill switch before any round trip", async () => {
    process.env[SYNTAX_CHECK_ENV] = "0";
    const [shell] = register();
    const ctx = stubCtx(PERL_FAILURE);
    expect(await shell(shellResult(WRITE, WROTE_IT), ctx)).toBeUndefined();
    expect(ctx.proxied).toEqual([]);
  });

  it("resolves a relative write against the footer's cwd, not pi's host cwd", async () => {
    const [shell] = register();
    const ctx = stubCtx(PERL_FAILURE);
    await shell(shellResult("cat > v3.pl <<'EOF'\nx\nEOF", WROTE_IT), ctx);
    expect(ctx.proxied[0]).toContain("f='/app/v3.pl'");
  });

  it("ignores tools it does not own", async () => {
    const [shell] = register();
    const ctx = stubCtx(PERL_FAILURE);
    const event = { ...shellResult(WRITE, WROTE_IT), toolName: "read" };
    expect(await shell(event, ctx)).toBeUndefined();
  });
});

describe("write/edit handler", () => {
  it("checks the path key pi's write tool actually uses", async () => {
    const [, write] = register();
    const ctx = stubCtx(formatOutput("SyntaxError: invalid syntax", 1, "/host/pi", false, ""));
    const out = await write(
      { toolName: "write", isError: false, input: { path: "sol.py" }, content: [{ type: "text", text: "wrote sol.py" }] },
      ctx,
    );
    expect(ctx.proxied[0]).toContain("f='/host/pi/sol.py'");
    expect(out.content[0].text).toContain("[syntax-check] python3 ast.parse /host/pi/sol.py FAILED");
  });

  it("still fires on the legacy file_path key", async () => {
    const [, write] = register();
    const ctx = stubCtx(formatOutput("SyntaxError", 1, "/host/pi", false, ""));
    const out = await write(
      { toolName: "edit", isError: false, input: { file_path: "/x/sol.py" }, content: [{ type: "text", text: "edited" }] },
      ctx,
    );
    expect(out.content[0].text).toContain("[syntax-check]");
  });

  it("ignores a language Phase 1 has no checker for", async () => {
    const [, write] = register();
    const ctx = stubCtx(formatOutput("boom", 1, "/host/pi", false, ""));
    const event = { toolName: "write", isError: false, input: { path: "README.md" }, content: [{ type: "text", text: "wrote" }] };
    expect(await write(event, ctx)).toBeUndefined();
    expect(ctx.proxied).toEqual([]);
  });
});
