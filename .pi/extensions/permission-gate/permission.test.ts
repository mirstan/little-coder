import { describe, it, expect } from "vitest";
import setupPermissionGate, {
  isSafeBash,
  parseExtraPrefixes,
  getSafePrefixes,
} from "./index.ts";

describe("isSafeBash", () => {
  it("allows whitelisted read-only commands", () => {
    expect(isSafeBash("ls -la")).toBe(true);
    expect(isSafeBash("cat /etc/hosts")).toBe(true);
    expect(isSafeBash("git log --oneline")).toBe(true);
    expect(isSafeBash("grep -r pattern .")).toBe(true);
    expect(isSafeBash("rg pattern src/")).toBe(true);
    expect(isSafeBash("sed -n '1,20p' file.ts")).toBe(true);
  });
  it("allows routine filesystem scaffolding (cp/mv/mkdir/touch)", () => {
    expect(isSafeBash("cp a b")).toBe(true);
    expect(isSafeBash("mv old new")).toBe(true);
    expect(isSafeBash("mkdir -p sub/dir")).toBe(true);
    expect(isSafeBash("touch foo.md")).toBe(true);
  });
  it("preserves trailing-whitespace word boundary on fs prefixes", () => {
    // Without the trailing space, "cp" would match "cpufetch". With it, these stay blocked.
    expect(isSafeBash("cpufetch")).toBe(false);
    expect(isSafeBash("mvtool")).toBe(false);
    expect(isSafeBash("mkdiroops")).toBe(false);
    expect(isSafeBash("touchscreen")).toBe(false);
  });
  it("blocks non-whitelisted commands", () => {
    expect(isSafeBash("rm -rf /")).toBe(false);
    expect(isSafeBash("npm install foo")).toBe(false);
    expect(isSafeBash("sudo anything")).toBe(false);
  });
  it("handles leading whitespace", () => {
    expect(isSafeBash("   ls")).toBe(true);
  });
  it("git subcommand gating is strict", () => {
    expect(isSafeBash("git log")).toBe(true);
    expect(isSafeBash("git push origin main")).toBe(false);
    expect(isSafeBash("git commit -m x")).toBe(false);
  });
  it("respects an explicit prefix list (LITTLE_CODER_BASH_ALLOW shape)", () => {
    const extra = ["make ", "docker compose ps"];
    expect(isSafeBash("make test", extra)).toBe(true);
    expect(isSafeBash("docker compose ps", extra)).toBe(true);
    expect(isSafeBash("docker compose down", extra)).toBe(false);
  });
});

// ── Issue #70 ──────────────────────────────────────────────────────────────
// The old check was `command.trim().startsWith(prefix)` on the raw string, so
// it only ever judged the first command and had no idea a `>` was sitting
// right after a whitelisted one.

describe("isSafeBash hardening (issue #70)", () => {
  it("refuses a whitelisted command that redirects into a file", () => {
    // rvanswieten's repro: "cat" is whitelisted, and the check was startsWith.
    expect(isSafeBash("cat > backend/main.py << 'ENDOFFILE'\nbody\nENDOFFILE")).toBe(false);
    expect(isSafeBash("echo hi > notes.txt")).toBe(false);
    expect(isSafeBash("echo hi >> notes.txt")).toBe(false);
    expect(isSafeBash("cat a | tee b.txt")).toBe(false);
  });

  it("judges every command in a chain, not just the first", () => {
    expect(isSafeBash("ls && rm -rf /")).toBe(false);
    expect(isSafeBash("ls ; npm install foo")).toBe(false);
    expect(isSafeBash("cat f | sudo tee /etc/hosts")).toBe(false);
    // …and still allows a chain where every link is whitelisted.
    expect(isSafeBash("ls && git status")).toBe(true);
    expect(isSafeBash("cat f | grep x | wc -l")).toBe(true);
  });

  it("does not trip over redirect-shaped text that writes nothing", () => {
    expect(isSafeBash("make 2>&1", ["make "])).toBe(true);
    expect(isSafeBash('grep "a > b" file.txt')).toBe(true);
    expect(isSafeBash("wc -l < input.txt")).toBe(true);
  });

  it("refuses an empty command", () => {
    expect(isSafeBash("")).toBe(false);
    expect(isSafeBash("   ")).toBe(false);
  });
});

// ── Bare `&` ───────────────────────────────────────────────────────────────
// `&` backgrounds a job and runs the next command, but it was absent from the
// chain-operator list: `ls & rm -rf /` was judged as the single segment `ls`
// and allowed, while the `&&` spelling of the same thing was refused.

describe("isSafeBash — a bare & separates commands", () => {
  it("refuses a non-whitelisted command backgrounded behind a safe one", () => {
    expect(isSafeBash("ls & rm -rf /")).toBe(false);
    expect(isSafeBash("ls & npm install foo")).toBe(false);
    expect(isSafeBash("cat f & sudo tee /etc/hosts")).toBe(false);
  });

  it("judges both sides of a bare & on their own", () => {
    expect(isSafeBash("ls & echo hi")).toBe(true);
    expect(isSafeBash("echo hi & rm -rf /")).toBe(false);
    expect(isSafeBash("sleep 1 &", ["sleep "])).toBe(true);
  });

  it("leaves && behaving exactly as before", () => {
    expect(isSafeBash("ls && rm -rf /")).toBe(false);
    expect(isSafeBash("ls && git status")).toBe(true);
  });

  it("does not read fd duplication or an &> redirect as backgrounding", () => {
    expect(isSafeBash("make 2>&1", ["make "])).toBe(true);
    expect(isSafeBash("make >&2", ["make "])).toBe(true);
    expect(isSafeBash("make 2>&-", ["make "])).toBe(true);
    expect(isSafeBash("make <&3", ["make "])).toBe(true);
    expect(isSafeBash("make &>/dev/null", ["make "])).toBe(true);
    expect(isSafeBash("make &>>/dev/null", ["make "])).toBe(true);
  });

  it("refuses an &> redirect used to smuggle a trailing command", () => {
    // Harmless under bash (`rm -rf /` are arguments to `cat`), but a real
    // second command under /bin/sh or dash — and pi's own `bash` tool falls
    // back to `sh` when no bash is on the image, so the gate must not assume.
    expect(isSafeBash("cat &>/dev/null rm -rf /")).toBe(false);
    expect(isSafeBash("ls &>>/dev/null rm -rf /")).toBe(false);
  });

  it("does not read a quoted or escaped & as an operator", () => {
    expect(isSafeBash('echo "a & b"')).toBe(true);
    expect(isSafeBash("echo a \\& b")).toBe(true);
  });
});

describe("isSafeBash — env is not a safe prefix", () => {
  it("refuses env, whose argument runs verbatim past the prefix check", () => {
    expect(isSafeBash("env rm -rf /")).toBe(false);
    expect(isSafeBash("env ls")).toBe(false);
    expect(isSafeBash("env FOO=1 npm install foo")).toBe(false);
  });

  it("keeps printenv, the read-only use env was covering", () => {
    expect(isSafeBash("printenv")).toBe(true);
    expect(isSafeBash("printenv PATH")).toBe(true);
    expect(getSafePrefixes()).not.toContain("env");
  });
});

// ── Command substitution ───────────────────────────────────────────────────
// `$(…)`, backticks and `<(…)` run a command line the prefix check never
// sees — `echo $(rm -rf /)` is judged as `echo`.

describe("isSafeBash — command substitution is refused outright", () => {
  it("refuses every substitution spelling behind a whitelisted command", () => {
    for (const cmd of [
      "echo $(touch X)",
      "echo `touch X`",
      'echo "$(touch X)"',
      "cat <(touch X)",
      "echo hi >(touch X)",
      "echo ${x:-$(touch X)}",
      "ls $(rm -rf /)",
    ]) {
      expect(isSafeBash(cmd), cmd).toBe(false);
    }
  });

  it("refuses substitution smuggled into an unquoted-delimiter heredoc body", () => {
    expect(isSafeBash("cat <<EOF\n$(touch PWNED)\nEOF")).toBe(false);
    // An apostrophe in the body leaves the quote tracker open, which used to
    // make everything after it look literal.
    expect(isSafeBash("cat <<EOF\ndon't worry\n$(touch PWNED)\nEOF")).toBe(false);
  });

  it("runs before prefix matching, so no allow-entry can lift it", () => {
    expect(isSafeBash("echo $(touch X)", ["echo ", "echo $(touch X)"])).toBe(false);
  });

  it("still allows commands that only look like substitution", () => {
    expect(isSafeBash("echo $HOME")).toBe(true);
    expect(isSafeBash("echo '$(x)'")).toBe(true);
    expect(isSafeBash('echo "$var"')).toBe(true);
    expect(isSafeBash("grep foo <<< \"$var\"")).toBe(true);
    expect(isSafeBash("cat <<'EOF'\n$(not expanded)\nEOF")).toBe(true);
  });
});

// ── Heredoc opening line ───────────────────────────────────────────────────
// Stripping a heredoc deleted the rest of its opening line along with the
// body, hiding the chain operators and redirects that sit there.

describe("isSafeBash — the heredoc opening line is not swallowed", () => {
  it("sees a command chained after the <<DELIM token", () => {
    expect(isSafeBash("cat <<EOF & rm -rf /\nbody\nEOF")).toBe(false);
  });

  it("sees a redirect placed after the <<DELIM token", () => {
    expect(isSafeBash("cat << 'EOF' > backend/app.js\nX\nEOF")).toBe(false);
  });
});

// ── `>&PATH` ───────────────────────────────────────────────────────────────

describe("isSafeBash — >&word is only plumbing when word is a fd", () => {
  it("refuses a >& redirect that truncates a real file", () => {
    for (const cmd of [
      "cat README.md >&backend/app.js",
      "ls >&/etc/passwd",
      "ls >& /etc/passwd",
      "ls 2>&/etc/passwd",
      "ls 1>&/tmp/victim.txt",
    ]) {
      expect(isSafeBash(cmd), cmd).toBe(false);
    }
  });

  it("keeps allowing genuine descriptor plumbing", () => {
    expect(isSafeBash("make 2>&1", ["make "])).toBe(true);
    expect(isSafeBash("echo err >&2")).toBe(true);
    expect(isSafeBash("make 2>&-", ["make "])).toBe(true);
    expect(isSafeBash("make >& 2", ["make "])).toBe(true);
  });
});

// ── Prefix word boundaries ─────────────────────────────────────────────────
// Most entries had no trailing space, so they matched by bare `startsWith`:
// `ls` matched `lsof`, and `git diff` matched `git difftool`, whose `-x` flag
// runs an arbitrary command (verified executing against real git).

describe("isSafeBash — safe prefixes end on a word boundary", () => {
  it("refuses a longer binary that merely starts with a whitelisted name", () => {
    for (const cmd of [
      "lsof -i",
      "catman x",
      "idle",
      "typeset -f",
      "dateutils.dadd",
      "echoevil",
      "tailscale up",
      "whoamid",
      "printenvx",
      "headers",
    ]) {
      expect(isSafeBash(cmd), cmd).toBe(false);
    }
  });

  it("refuses a longer SUBCOMMAND that starts with a whitelisted one", () => {
    // `git difftool -y -x CMD` executes CMD — this was arbitrary code
    // execution through a whitelist of read-only git subcommands.
    expect(isSafeBash("git difftool -y -x 'touch PWNED' HEAD~1")).toBe(false);
    expect(isSafeBash("git logfoo")).toBe(false);
    expect(isSafeBash("git statuses")).toBe(false);
    expect(isSafeBash("pip showoff")).toBe(false);
    expect(isSafeBash("npm listen")).toBe(false);
    expect(isSafeBash("cargo metadatax")).toBe(false);
  });

  it("still allows a bare command with no arguments", () => {
    for (const cmd of ["ls", "pwd", "printenv", "git log", "git status", "git stash list"]) {
      expect(isSafeBash(cmd), cmd).toBe(true);
    }
  });

  it("still allows the ordinary argument forms", () => {
    for (const cmd of [
      "ls -la",
      "git log --oneline",
      "git diff --stat",
      "printenv PATH",
      "cp a b",
      "ls && git status",
      "npm list",
      "pip list",
    ]) {
      expect(isSafeBash(cmd), cmd).toBe(true);
    }
  });

  it("keeps the glued-flag entries matching their flag forms", () => {
    // `top -bn` / `curl -I` are prefixes of a flag, not whole commands, so
    // they deliberately carry no trailing space.
    expect(isSafeBash("top -bn1")).toBe(true);
    expect(isSafeBash("curl -I url")).toBe(true);
    expect(isSafeBash("curl -IL url")).toBe(true);
    expect(isSafeBash("curl --head url")).toBe(true);
  });

  it("allows a bare interpreter — an intended, understood delta", () => {
    // The exact-match clause newly admits these, which today are refused only
    // because `"python3 "` cannot match a bare segment. It grants nothing new:
    // `python3 -c '…'` is already whitelisted, a separate and documented
    // interpreter hole. Pinned so it reads as a decision, not a regression.
    expect(isSafeBash("python3")).toBe(true);
    expect(isSafeBash("sed")).toBe(true);
    expect(isSafeBash("node")).toBe(true);
  });
});

describe("permission-gate tool_call interceptor", () => {
  function getHandler() {
    let handler: ((event: any, ctx: any) => any) | undefined;
    setupPermissionGate({
      on(name: string, h: (event: any, ctx: any) => any) {
        if (name === "tool_call") handler = h;
      },
    } as any);
    if (!handler) throw new Error("permission-gate registered no tool_call handler");
    return handler;
  }

  async function withMode<T>(mode: string | undefined, fn: () => T | Promise<T>): Promise<T> {
    const prev = process.env.LITTLE_CODER_PERMISSION_MODE;
    if (mode === undefined) delete process.env.LITTLE_CODER_PERMISSION_MODE;
    else process.env.LITTLE_CODER_PERMISSION_MODE = mode;
    try {
      return await fn();
    } finally {
      if (prev === undefined) delete process.env.LITTLE_CODER_PERMISSION_MODE;
      else process.env.LITTLE_CODER_PERMISSION_MODE = prev;
    }
  }

  it("gates ShellSession, which used to reach execSync with no gate at all", async () => {
    const handler = getHandler();
    await withMode("auto", async () => {
      const result = await handler(
        { toolName: "ShellSession", input: { command: "rm -rf /" } },
        {},
      );
      expect(result?.block).toBe(true);
    });
  });

  it("explains a shell-redirect refusal in terms the model can act on", async () => {
    const handler = getHandler();
    await withMode("auto", async () => {
      const result = await handler(
        { toolName: "ShellSession", input: { command: "cat > main.py << 'EOF'\nx\nEOF" } },
        {},
      );
      expect(result?.block).toBe(true);
      expect(result.reason).toContain("main.py");
      expect(result.reason).toMatch(/Write tool|Edit/);
    });
  });

  it("names the offending command in a chain, not the harmless first one", async () => {
    const handler = getHandler();
    await withMode("auto", async () => {
      const result = await handler(
        { toolName: "bash", input: { command: "ls && npm install left-pad" } },
        {},
      );
      expect(result?.block).toBe(true);
      expect(result.reason).toContain("npm");
      expect(result.reason).not.toContain('"ls"');
    });
  });

  it("explains a substitution refusal without offering an override that cannot work", async () => {
    const handler = getHandler();
    await withMode("auto", async () => {
      const result = await handler(
        { toolName: "bash", input: { command: "echo $(rm -rf /)" } },
        {},
      );
      expect(result?.block).toBe(true);
      expect(result.reason).toMatch(/substitution/);
      // Must not fall through to the generic prefix message, which would send
      // the model hunting for an allow-entry that this check runs ahead of.
      expect(result.reason).not.toContain("SAFE_PREFIXES");
      expect(result.reason).toContain("LITTLE_CODER_BASH_ALLOW cannot override this");
    });
  });

  it("still lets whitelisted commands through on every shell tool", async () => {
    const handler = getHandler();
    await withMode("auto", async () => {
      for (const toolName of ["bash", "Bash", "ShellSession"]) {
        const result = await handler({ toolName, input: { command: "git status" } }, {});
        expect(result, toolName).toBeUndefined();
      }
    });
  });

  it("leaves the no-command shell helpers ungated", async () => {
    const handler = getHandler();
    await withMode("auto", async () => {
      for (const toolName of ["ShellSessionCwd", "ShellSessionReset"]) {
        expect(await handler({ toolName, input: {} }, {}), toolName).toBeUndefined();
      }
    });
  });

  it("accept-all mode still passes everything (benchmark runs)", async () => {
    const handler = getHandler();
    await withMode("accept-all", async () => {
      const result = await handler(
        { toolName: "ShellSession", input: { command: "cat > main.py << 'EOF'\nx\nEOF" } },
        {},
      );
      expect(result).toBeUndefined();
    });
  });

  it("manual mode prompts and allows when user confirms", async () => {
    const handler = getHandler();
    await withMode("manual", async () => {
      const result = await handler(
        { toolName: "bash", input: { command: "rm -rf /" } },
        { ui: { confirm: async () => true } },
      );
      expect(result).toBeUndefined();
    });
  });

  it("manual mode prompts and blocks when user declines", async () => {
    const handler = getHandler();
    await withMode("manual", async () => {
      const result = await handler(
        { toolName: "bash", input: { command: "rm -rf /" } },
        { ui: { confirm: async () => false } },
      );
      expect(result?.block).toBe(true);
      expect(result.reason).toBe("command cancelled by user");
    });
  });

  it("manual mode prompts for whitelisted commands too", async () => {
    const handler = getHandler();
    let promptShown = false;
    await withMode("manual", async () => {
      await handler(
        { toolName: "bash", input: { command: "git status" } },
        {
          ui: {
            confirm: async () => {
              promptShown = true;
              return true;
            },
          },
        },
      );
      expect(promptShown).toBe(true);
    });
  });
});

describe("parseExtraPrefixes", () => {
  it("returns empty for undefined / empty / whitespace", () => {
    expect(parseExtraPrefixes(undefined)).toEqual([]);
    expect(parseExtraPrefixes("")).toEqual([]);
    expect(parseExtraPrefixes("   ")).toEqual([]);
  });
  it("splits on comma and trims leading whitespace, preserving trailing space as word boundary", () => {
    expect(parseExtraPrefixes("make , docker compose ps,  bun run")).toEqual([
      "make ",
      "docker compose ps",
      "bun run",
    ]);
  });
  it("drops empty / whitespace-only segments", () => {
    expect(parseExtraPrefixes("a,,b,")).toEqual(["a", "b"]);
    expect(parseExtraPrefixes("a,   ,b")).toEqual(["a", "b"]);
  });
});

describe("getSafePrefixes", () => {
  it("merges builtins with LITTLE_CODER_BASH_ALLOW from the env", () => {
    const prev = process.env.LITTLE_CODER_BASH_ALLOW;
    process.env.LITTLE_CODER_BASH_ALLOW = "make ,docker compose ps";
    try {
      const all = getSafePrefixes();
      expect(all).toContain("ls "); // builtin still present (with its word boundary)
      expect(all).toContain("make ");
      expect(all).toContain("docker compose ps");
    } finally {
      if (prev === undefined) delete process.env.LITTLE_CODER_BASH_ALLOW;
      else process.env.LITTLE_CODER_BASH_ALLOW = prev;
    }
  });
});
