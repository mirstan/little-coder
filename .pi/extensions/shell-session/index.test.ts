import { describe, it, expect, afterAll, beforeEach, vi } from "vitest";
import { rmSync } from "node:fs";
import { dirname } from "node:path";

// execSync is stubbed so the failure modes below (a maxBuffer overflow needs
// >10MB of real output, a timeout needs to actually elapse) are reachable in
// milliseconds and deterministically.
const { execSyncMock } = vi.hoisted(() => ({ execSyncMock: vi.fn() }));
vi.mock("node:child_process", () => ({ execSync: execSyncMock }));

const { execSubprocess, default: register } = await import("./index.ts");

/** An execSync throw shaped exactly like a real maxBuffer overflow. */
function enobufsError(captured: string): Error {
  const err: any = new Error("stdout maxBuffer length exceeded");
  err.code = "ENOBUFS";
  err.signal = "SIGTERM"; // Node kills the child, same as it does on timeout
  err.status = null;
  err.stdout = captured;
  err.stderr = "";
  return err;
}

function timeoutError(): Error {
  const err: any = new Error("Command failed");
  err.code = "ETIMEDOUT";
  err.signal = "SIGTERM";
  err.status = null;
  err.stdout = "partial\n";
  err.stderr = "";
  return err;
}

const overflowDirs = new Set<string>();
function noteOverflowDir(out: string): void {
  const match = out.match(/^(?:Full|Partial) output[^:]*: (\/.*)$/m);
  if (match) overflowDirs.add(dirname(match[1]));
}
afterAll(() => {
  for (const dir of overflowDirs) rmSync(dir, { recursive: true, force: true });
});

beforeEach(() => {
  execSyncMock.mockReset();
  delete process.env.LITTLE_CODER_TB_MODE;
});

describe("execSubprocess maxBuffer overflow", () => {
  const CAPTURED = "x".repeat(1024 * 1024);

  it("does not report a maxBuffer overflow as a timeout", async () => {
    execSyncMock.mockImplementation(() => {
      throw enobufsError(CAPTURED);
    });
    const out = await execSubprocess("yes", 30);
    noteOverflowDir(out);

    const footer = out.split("\n").at(-1)!;
    expect(footer).toContain("timed_out=false");
    expect(footer).toContain("exit=-1");
  });

  it("labels the saved file partial and disowns the raw_bytes count", async () => {
    execSyncMock.mockImplementation(() => {
      throw enobufsError(CAPTURED);
    });
    const out = await execSubprocess("yes", 30);
    noteOverflowDir(out);

    expect(out).toContain("Partial output (first ~10MB only");
    expect(out).not.toContain("Full output:");
    expect(out.split("\n").at(-1)).toContain("raw_bytes_exact=false");
  });

  it("still reports a real timeout as one", async () => {
    execSyncMock.mockImplementation(() => {
      throw timeoutError();
    });
    const out = await execSubprocess("sleep 99", 1);

    const footer = out.split("\n").at(-1)!;
    expect(footer).toContain("timed_out=true");
    expect(footer).not.toContain("raw_bytes_exact=false");
  });

  it("claims an exact, complete file for output that merely hit the byte cap", async () => {
    execSyncMock.mockReturnValue("z".repeat(1024 * 1024));
    const out = await execSubprocess("cat vocab.json", 30);
    noteOverflowDir(out);

    expect(out).toContain("Full output: ");
    expect(out.split("\n").at(-1)).not.toContain("raw_bytes_exact=false");
  });
});

describe("tmux-proxy backend", () => {
  function toolsOf(): Map<string, any> {
    const tools = new Map<string, any>();
    register({ registerTool: (t: any) => tools.set(t.name, t) } as any);
    return tools;
  }

  it("never saves an overflow file, however large the proxy's output", async () => {
    // The container the command ran in cannot see a host tmp path, so a
    // "Full output:" line there would name a file the model can never read.
    process.env.LITTLE_CODER_TB_MODE = "1";
    const giant = "q".repeat(2 * 1024 * 1024);
    const ctx = { ui: { input: async () => giant } };

    const res = await toolsOf().get("ShellSession").execute("id", { command: "cat big" }, undefined, undefined, ctx);
    const text = res.content[0].text;

    expect(text).not.toContain("Full output:");
    expect(text).not.toContain("Partial output");
    expect(execSyncMock).not.toHaveBeenCalled();
  });
});
