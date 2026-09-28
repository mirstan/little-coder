import { describe, it, expect, afterEach } from "vitest";
import {
  CHECK_TIMEOUT_SEC,
  MAX_DIAG_CHARS,
  MAX_DIAG_LINES,
  MAX_FILES_PER_COMMAND,
  SKIP_SENTINEL,
  SYNTAX_CHECK_ENV,
  buildCheckCommand,
  checkTargets,
  checkerFor,
  commandRunsFile,
  footerCwd,
  footerExit,
  formatDiagnostic,
  insertAboveFooter,
  readCheckResult,
  syntaxCheckEnabled,
} from "./helpers.ts";
import { splitFooter } from "../truncated-view/truncation.ts";
import { splitCommandChain } from "../_shared/shell-write.ts";
import { formatOutput } from "../shell-session/helpers.ts";

// The real footer shape, from the extension that writes it — a hand-typed
// literal here would keep passing after the format moved.
const FOOTER = formatOutput("", 0, "/app", false, "backend=harbor-env").split("\n").at(-1)!;

// Snapshot/restore rather than unconditionally delete: a suite run with
// LITTLE_CODER_SYNTAX_CHECK already exported (e.g. to disable the extension
// while debugging something unrelated) must not have that value erased once
// this file's tests are done with it.
const ORIGINAL_SYNTAX_CHECK_ENV = process.env[SYNTAX_CHECK_ENV];
afterEach(() => {
  if (ORIGINAL_SYNTAX_CHECK_ENV === undefined) delete process.env[SYNTAX_CHECK_ENV];
  else process.env[SYNTAX_CHECK_ENV] = ORIGINAL_SYNTAX_CHECK_ENV;
});

describe("kill switch", () => {
  it("defaults on", () => {
    // Cleared explicitly, not just assumed absent -- this test must reflect
    // the true default regardless of what the ambient shell exported before
    // the suite started.
    delete process.env[SYNTAX_CHECK_ENV];
    expect(syntaxCheckEnabled()).toBe(true);
  });

  it("is off at exactly 0", () => {
    process.env[SYNTAX_CHECK_ENV] = "0";
    expect(syntaxCheckEnabled()).toBe(false);
  });

  it("stays on for any other value, matching the LITTLE_CODER_TB_MODE convention", () => {
    process.env[SYNTAX_CHECK_ENV] = "false";
    expect(syntaxCheckEnabled()).toBe(true);
  });
});

describe("checkerFor", () => {
  it("covers the four MVP languages", () => {
    expect(checkerFor("/tmp/a.py")?.label).toBe("python3 ast.parse");
    expect(checkerFor("/tmp/a.pl")?.label).toBe("perl -c");
    expect(checkerFor("/tmp/a.pm")?.label).toBe("perl -c");
    expect(checkerFor("/tmp/a.t")?.label).toBe("perl -c");
    expect(checkerFor("/tmp/a.c")?.label).toBe("gcc/cc -fsyntax-only");
    expect(checkerFor("/tmp/a.h")?.label).toBe("gcc/cc -fsyntax-only");
    expect(checkerFor("/tmp/a.sh")?.label).toBe("bash/sh -n");
    expect(checkerFor("/tmp/a.bash")?.label).toBe("bash/sh -n");
  });

  it("declines everything Phase 1 has no cheap checker for", () => {
    for (const p of ["/tmp/a.js", "/tmp/a.cob", "/tmp/a.tex", "/tmp/a.md", "/tmp/a.txt"]) {
      expect(checkerFor(p)).toBeNull();
    }
  });

  it("is case-insensitive on the extension", () => {
    expect(checkerFor("/tmp/A.PY")?.label).toBe("python3 ast.parse");
  });

  it("does not read a dotfile's name as an extension", () => {
    // `.py` here is the whole basename, not a Python file.
    expect(checkerFor("/tmp/.py")).toBeNull();
    expect(checkerFor("/tmp/Makefile")).toBeNull();
  });

  it("does not let a directory's dot leak into the extension", () => {
    expect(checkerFor("/tmp/v1.2/Makefile")).toBeNull();
  });
});

describe("buildCheckCommand", () => {
  it("exits silently when the write never landed", () => {
    const cmd = buildCheckCommand("/tmp/v3.pl", checkerFor("/tmp/v3.pl")!);
    expect(cmd).toContain(`[ -f "$f" ] || exit 0;`);
  });

  it("prints the skip sentinel instead of failing when the checker is absent", () => {
    for (const p of ["/tmp/a.py", "/tmp/a.pl", "/tmp/a.c", "/tmp/a.sh"]) {
      expect(buildCheckCommand(p, checkerFor(p)!)).toContain(`echo ${SKIP_SENTINEL}`);
    }
  });

  it("survives a quote in the path", () => {
    const cmd = buildCheckCommand("/tmp/it's.py", checkerFor("/tmp/a.py")!);
    expect(cmd.startsWith(`f='/tmp/it'\\''s.py';`)).toBe(true);
  });

  it("checks python syntax only, so an uninstalled import is never reported", () => {
    const cmd = buildCheckCommand("/tmp/a.py", checkerFor("/tmp/a.py")!);
    expect(cmd).toContain("ast.parse");
    expect(cmd).not.toContain("import_module");
  });
});

describe("commandRunsFile", () => {
  const HEREDOC = [
    "cat > /tmp/v3.pl <<'EOF'",
    "my %ctxbits;",
    "$ctxbits{$ctx]++;",
    "EOF",
  ].join("\n");

  /** Tests `commandRunsFile` in isolation, against every segment of `cmd` --
   *  `checkTargets` is what actually scopes this to segments after a write;
   *  see its own describe block for that ordering behavior. */
  function runs(cmd: string, rawPath: string): boolean {
    return commandRunsFile(splitCommandChain(cmd), rawPath);
  }

  it("skips the write-then-run chain the model overwhelmingly issues", () => {
    expect(runs(`${HEREDOC}\nperl /tmp/v3.pl`, "/tmp/v3.pl")).toBe(true);
  });

  it("fires on a bare write with no run", () => {
    expect(runs(HEREDOC, "/tmp/v3.pl")).toBe(false);
  });

  it("fires on a sed -i patch, the case the round trip is currently wasted on", () => {
    expect(runs("sed -i 's/foo/bar/' /tmp/v3.pl", "/tmp/v3.pl")).toBe(false);
  });

  it("fires on write-then-separate-compile of a DIFFERENT file", () => {
    expect(runs("cat > a.c <<EOF\nx\nEOF\ngcc -o b b.c", "a.c")).toBe(false);
  });

  it("matches a relative spelling against a relative write target", () => {
    expect(runs("cat > f.pl <<EOF\nx\nEOF\nperl f.pl", "f.pl")).toBe(true);
    expect(runs("cat > f.pl <<EOF\nx\nEOF\nperl ./f.pl", "f.pl")).toBe(true);
  });

  it("matches a run against a backslash-escaped operand the same way the write detector unescapes it", () => {
    // detectDeliverableWrites reports the write as "a b.pl" (its own unquote
    // strips the backslash); a run spelled `perl a\ b.pl` must resolve to the
    // same unescaped operand, not the literal `a\ b.pl`.
    expect(runs("cat > a\\ b.pl <<EOF\nx\nEOF\nperl a\\ b.pl", "a b.pl")).toBe(true);
  });

  it("does not mistake a longer filename that ends in the target's name", () => {
    expect(runs("cat > t.sh <<EOF\nx\nEOF\nbash test.sh", "t.sh")).toBe(false);
  });

  it("recognises an absolute or versioned interpreter", () => {
    expect(runs("cat > a.py <<EOF\nx\nEOF\n/usr/bin/python3 a.py", "a.py")).toBe(true);
    expect(runs("cat > a.py <<EOF\nx\nEOF\npython3.11 a.py", "a.py")).toBe(true);
  });

  it("ignores a run that only appears inside the heredoc body being written", () => {
    const cmd = ["cat > /tmp/run.sh <<'EOF'", "perl /tmp/run.sh", "EOF"].join("\n");
    expect(runs(cmd, "/tmp/run.sh")).toBe(false);
  });

  it("ignores a non-runner that merely names the file", () => {
    expect(runs("cat > a.py <<EOF\nx\nEOF\nchmod +x a.py", "a.py")).toBe(false);
  });

  it("requires an exact match for a path with a directory component -- basename alone is not enough", () => {
    // /app/f.pl and /tmp/f.pl share a basename but are different files; a run
    // of one must not suppress the check on the other.
    expect(runs("perl /tmp/f.pl", "/app/f.pl")).toBe(false);
  });

  it("still matches basename-only for a bare filename with no directory", () => {
    expect(runs("perl dir/f.pl", "f.pl")).toBe(true);
  });
});

describe("checkTargets", () => {
  it("resolves a relative write against the shell's cwd, keeping the raw spelling", () => {
    const [target] = checkTargets("cat > v3.pl <<'EOF'\nx\nEOF", "/app");
    expect(target.raw).toBe("v3.pl");
    expect(target.path).toBe("/app/v3.pl");
  });

  it("picks up a sed -i patch", () => {
    const targets = checkTargets("sed -i '25s/.*/ok/' /tmp/v3.pl", "/app");
    expect(targets.map((t) => t.path)).toEqual(["/tmp/v3.pl"]);
  });

  it("drops files no Phase 1 checker covers", () => {
    expect(checkTargets("echo hi > /tmp/notes.md", "/app")).toEqual([]);
  });

  it("drops a file the same command already runs", () => {
    expect(checkTargets("cat > a.py <<EOF\nx\nEOF\npython3 a.py", "/app")).toEqual([]);
  });

  it("caps the files checked per command", () => {
    const cmd = ["a", "b", "c", "d", "e"].map((n) => `echo x > /tmp/${n}.py`).join("\n");
    expect(checkTargets(cmd, "/app")).toHaveLength(MAX_FILES_PER_COMMAND);
  });

  it("checks a file written twice in one command only once", () => {
    const targets = checkTargets("echo a > /tmp/x.py\necho b >> /tmp/x.py", "/app");
    expect(targets).toHaveLength(1);
  });

  it("still checks a file whose only run in the command came BEFORE the write", () => {
    // perl f.pl runs the pre-write file; the write after it replaces the
    // content with something the model has not executed at all.
    const targets = checkTargets("perl f.pl; cat > f.pl <<EOF\nx\nEOF", "/app");
    expect(targets.map((t) => t.raw)).toEqual(["f.pl"]);
  });

  it("still skips when the run comes after the write, in a multi-write command", () => {
    const cmd = ["cat > a.py <<EOF", "x", "EOF", "cat > b.py <<EOF", "y", "EOF", "python3 b.py"].join("\n");
    const targets = checkTargets(cmd, "/app");
    expect(targets.map((t) => t.raw)).toEqual(["a.py"]);
  });
});

describe("footer parsing", () => {
  it("reads exit and cwd out of a real footer", () => {
    const text = formatOutput("boom", 255, "/app", false, "backend=harbor-env");
    const { footer } = splitFooter(text);
    expect(footerExit(footer)).toBe(255);
    expect(footerCwd(footer)).toBe("/app");
  });

  it("reads a cwd containing spaces", () => {
    const text = formatOutput("", 0, "/app/my dir", false, "backend=harbor-env");
    expect(footerCwd(splitFooter(text).footer)).toBe("/app/my dir");
  });

  it("has nothing to read without a footer", () => {
    expect(footerExit(null)).toBeNull();
    expect(footerCwd(null)).toBeNull();
  });
});

describe("readCheckResult", () => {
  it("takes the exit code from the footer, not from stdout", () => {
    const text = formatOutput("exit=0 is printed by the file itself\nboom", 255, "/app", false, "");
    expect(readCheckResult(text)).toEqual({
      kind: "verdict",
      exit: 255,
      output: expect.stringContaining("boom"),
    });
  });

  it("fails open when the checker is not installed", () => {
    const text = formatOutput(SKIP_SENTINEL, 0, "/app", false, "");
    expect(readCheckResult(text).kind).toBe("skip");
  });

  it("does not mistake a real error that merely quotes the sentinel for a missing checker", () => {
    // A checker's own failure output could plausibly quote this token (this
    // repo is self-hosted -- a source file testing marker strings, say);
    // only the WHOLE trimmed output being the sentinel means "skipped".
    const text = formatOutput(`syntax error near token ${SKIP_SENTINEL}`, 1, "/app", false, "");
    expect(readCheckResult(text)).toMatchObject({ kind: "verdict", exit: 1 });
  });

  it("fails open on a timed-out check", () => {
    const text = formatOutput("", -1, "/app", true, "");
    expect(readCheckResult(text).kind).toBe("none");
  });

  it("fails open when no footer came back at all", () => {
    expect(readCheckResult("garbage from a broken proxy").kind).toBe("none");
  });

  it("reports a clean parse as exit 0, so the caller appends nothing", () => {
    const text = formatOutput("/tmp/v3.pl syntax OK", 0, "/app", false, "");
    expect(readCheckResult(text)).toMatchObject({ kind: "verdict", exit: 0 });
  });
});

describe("formatDiagnostic", () => {
  const base = { label: "perl -c", path: "/tmp/v3.pl", exit: 255 };

  it("leads with the checker, the file, and what to do", () => {
    const out = formatDiagnostic({
      ...base,
      output: 'syntax error at /tmp/v3.pl line 25, near "$ctx]"\n',
    });
    expect(out).toBe(
      "[syntax-check] perl -c /tmp/v3.pl FAILED (exit 255) — fix before running:\n" +
        'syntax error at /tmp/v3.pl line 25, near "$ctx]"',
    );
  });

  it("caps at 12 lines", () => {
    const out = formatDiagnostic({
      ...base,
      output: Array.from({ length: 50 }, (_, i) => `error ${i}`).join("\n"),
    });
    const lines = out.split("\n");
    expect(lines).toHaveLength(MAX_DIAG_LINES);
    expect(lines.at(-1)).toContain("truncated");
  });

  it("caps at ~1000 chars even when the line count is legal", () => {
    const out = formatDiagnostic({ ...base, output: "x".repeat(5000) });
    expect(out.length).toBeLessThanOrEqual(MAX_DIAG_CHARS);
    expect(out).toContain("truncated");
  });

  it("does not claim truncation when nothing was cut", () => {
    const out = formatDiagnostic({ ...base, output: "one error" });
    expect(out).not.toContain("truncated");
  });
});

describe("insertAboveFooter", () => {
  const DIAG = "[syntax-check] perl -c /tmp/v3.pl FAILED (exit 255) — fix before running:\nboom";

  it("keeps the footer last, so truncated-view still finds it", () => {
    const text = formatOutput("wrote it", 0, "/app", false, "backend=harbor-env");
    const out = insertAboveFooter(text, DIAG);
    expect(splitFooter(out).footer).toBe(FOOTER);
    expect(splitFooter(out).body).toContain("[syntax-check]");
  });

  it("puts the diagnostic after the command's own output", () => {
    const text = formatOutput("wrote it", 0, "/app", false, "backend=harbor-env");
    const out = insertAboveFooter(text, DIAG);
    expect(out.indexOf("wrote it")).toBeLessThan(out.indexOf("[syntax-check]"));
  });

  it("appends when the result has no footer (local bash)", () => {
    const out = insertAboveFooter("wrote it\n", DIAG);
    expect(out).toBe("wrote it\n" + DIAG);
    expect(out).toContain("[syntax-check]");
  });
});

describe("check budget", () => {
  it("bounds a single check well under a model turn", () => {
    expect(CHECK_TIMEOUT_SEC).toBeLessThanOrEqual(10);
  });
});
