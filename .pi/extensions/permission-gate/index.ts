import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
  SHELL_TOOLS,
  detectWriteTargets,
  hasCommandSubstitution,
  splitCommandChain,
} from "../_shared/shell-write.ts";

// Port of tools.py::_SAFE_PREFIXES + agent.py::_check_permission. Shell
// commands not matching the whitelist are blocked in "auto" mode. In
// "accept-all" mode all commands pass (benchmark runs set this explicitly).
// Write/Edit confirmations are deferred to the TUI's own prompt; we simply
// add an extra guardrail on the shell here to match little-coder's behavior.
//
// Per-deployment customization (issue #15):
//   LITTLE_CODER_PERMISSION_MODE=auto|accept-all|manual
//   LITTLE_CODER_BASH_ALLOW="cmd1,cmd2 sub,..."  extra allow-prefixes,
//                                                merged with the built-in list.
//
// Issue #70: the gate used to match only `bash`/`Bash`, so a model that hit a
// refusal could re-run the same thing through the `ShellSession` tool and land
// in an execSync with no gate at all. Every shell-executing tool is listed in
// SHELL_TOOLS now, and they all go through the same whitelist.

// A trailing space is the word boundary, and every entry naming a COMPLETE
// command carries one: "cp " matches "cp a b" but not "cpufetch", "git log "
// matches "git log --oneline" but not "git logfoo". A bare command with no
// arguments still matches via the exact-match clause in `isSafeBash`.
//
// The convention applies to multi-word entries just as much as single-word
// ones, and skipping them there was not cosmetic: "git diff" without the
// space matched `git difftool`, whose `-x` flag runs an arbitrary command
// (`git difftool -y -x 'touch PWNED' HEAD~1` — confirmed executing), so a
// whitelist of read-only git subcommands granted full code execution.
//
// The only entries WITHOUT a trailing space are the ones that are prefixes of
// a glued-together flag rather than whole commands: `top -bn` has to match
// `top -bn1`, and `curl -I` has to match `curl -IL`. Adding a boundary there
// would break the flag forms these exist to allow.
const BUILTIN_SAFE_PREFIXES: readonly string[] = [
  "ls ", "cat ", "head ", "tail ", "wc ", "pwd ", "echo ", "printf ", "date ",
  // `env` removed: `env <cmd>` runs `<cmd>` verbatim while the prefix match
  // only ever sees `env`, so `env rm -rf /tmp/x` was whitelisted. Its one
  // common read-only use is already covered by `printenv` below.
  // `find`/`sed`/`python`/`node` share that run-something-verbatim shape and
  // stay — dropping them costs real interactive usability, dropping `env`
  // cost none. They remain a known, accepted hole in prefix whitelisting.
  "which ", "type ", "printenv ", "uname ", "whoami ", "id ",
  "git log ", "git status ", "git diff ", "git show ", "git branch ",
  "git remote ", "git stash list ", "git tag ",
  "find ", "grep ", "rg ", "ag ", "fd ", "sed ",
  "python ", "python3 ", "node ", "ruby ", "perl ",
  "pip show ", "pip list ", "npm list ", "cargo metadata ",
  "df ", "du ", "free ", "ps ",
  // Glued-flag prefixes — no trailing space, see the note above.
  "top -bn", "curl -I", "curl --head",
  // Routine filesystem scaffolding. rm stays off the list by design; use
  // LITTLE_CODER_BASH_ALLOW=rm if a deployment needs it.
  "cp ", "mv ", "mkdir ", "touch ",
];

// Trailing whitespace is meaningful — it acts as a word boundary in startsWith
// matching ("find " refuses "findbug"). We only strip leading whitespace so
// callers retain control over that boundary.
export function parseExtraPrefixes(raw: string | undefined): string[] {
  if (!raw) return [];
  return raw
    .split(",")
    .map((s) => s.trimStart())
    .map((s) => (s.length > 0 && s !== " ".repeat(s.length) ? s : ""))
    .filter((s) => s.length > 0);
}

export function getSafePrefixes(): string[] {
  return [...BUILTIN_SAFE_PREFIXES, ...parseExtraPrefixes(process.env.LITTLE_CODER_BASH_ALLOW)];
}

/**
 * True when EVERY command in `command` is whitelisted and none of them writes.
 *
 * Two hardenings over the original `startsWith` check, both from issue #70:
 *
 * 1. **Judge every segment.** The check ran on the raw string, so only the
 *    first command was ever inspected — `ls && rm -rf /` was "safe" because it
 *    starts with `ls`. Each segment of a `&&`/`||`/`;`/`|` chain now has to
 *    match on its own.
 * 2. **A write is never safe.** `cat` is whitelisted, so
 *    `cat > backend/main.py << 'ENDOFFILE'` passed the prefix test outright.
 *    Any command carrying a redirect/tee/dd target is refused regardless of
 *    which binary it starts with; write-guard then decides whether that
 *    particular path may be written.
 */
export function isSafeBash(command: string, prefixes: readonly string[] = getSafePrefixes()): boolean {
  // First, before any segmentation: substitution hides a whole command line
  // from the prefix check, which only ever sees the outer command. It is
  // refused outright and deliberately ahead of prefix matching, so no
  // LITTLE_CODER_BASH_ALLOW entry can reach it.
  if (hasCommandSubstitution(command)) return false;
  if (detectWriteTargets(command).length > 0) return false;
  const segments = splitCommandChain(command);
  if (segments.length === 0) return false;
  return segments.every((segment) =>
    prefixes.some(
      // The second clause is the word boundary's other half: a bare command
      // with no arguments (`ls`, `pwd`, `git log`) has nothing for the
      // trailing space of `"ls "` to match against, so it is compared
      // against the entry with that boundary removed.
      (p) => segment.startsWith(p) || (p.endsWith(" ") && segment === p.slice(0, -1)),
    ),
  );
}

// Which tools count as "hands a string to a shell" lives in _shared, so this
// gate and write-guard can never disagree about it again (issue #70).

// A plain subcommand word: `merge`, `install`, `compose`, `run-tests`, `3.11`.
// Deliberately excludes anything with a scheme/userinfo (`:`, `/`, `@`), an
// `=`, a leading `-`, or shell metacharacters — see the offenderLabel comment
// below for why that matters.
const SUBCOMMAND_RE = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

function getPermissionMode(): "auto" | "accept-all" | "manual" {
  const v = process.env.LITTLE_CODER_PERMISSION_MODE;
  if (v === "accept-all" || v === "manual") return v;
  return "auto";
}

export default function (pi: ExtensionAPI) {
  pi.on("tool_call", async (event, ctx) => {
    const mode = getPermissionMode();
    if (mode === "accept-all") return;

    const toolName = (event as any).toolName;
    const input: any = (event as any).input ?? (event as any).args;

    // Only gate shell-family tools; pi has its own confirmation flow for
    // destructive edits via the TUI.
    if (SHELL_TOOLS.has(toolName)) {
      const cmd = input?.command;
      if (typeof cmd === "string") {
        if (mode === "manual") {
          const confirmed = await ctx.ui.confirm(
            `About to execute:\n\n${cmd}`,
            "Execute this command?",
          );
          if (!confirmed) {
            return { block: true, reason: "command cancelled by user" };
          }
          return;
        }
        if (!isSafeBash(cmd)) {
          // auto: block when not whitelisted. Name the reason precisely — a
          // refusal the model can't interpret just gets retried verbatim.
          if (hasCommandSubstitution(cmd)) {
            // A dedicated reason, not the generic prefix message: that one
            // names a binary and offers LITTLE_CODER_BASH_ALLOW, and here
            // both are wrong — the refusal is about the `$(…)` rather than
            // the command it wraps, and the check runs before prefix
            // matching, so no allow-entry can lift it. Saying so is what
            // stops a model from spending turns hunting for an override.
            return {
              block: true,
              reason:
                `shell whitelist: this command runs another command through substitution ` +
                `($(…), backticks, or <(…)), which hides it from the whitelist, so it was refused.\n` +
                `LITTLE_CODER_BASH_ALLOW cannot override this — the check runs before ` +
                `prefix matching, so there is no allow-entry to add and no point retrying a ` +
                `variation.\n` +
                `Run the inner command on its own line instead, and use its output directly. ` +
                `If a file needs changing, use edit/write, which do not need a shell.`,
            };
          }
          const writes = detectWriteTargets(cmd);
          if (writes.length > 0) {
            return {
              block: true,
              reason:
                `shell whitelist: this command writes to ${writes.map((w) => `"${w.path}"`).join(", ")} ` +
                `via shell redirection. Use the Write tool for a new file, or Edit for an existing one — ` +
                `do not redirect into files.`,
            };
          }
          const offender =
            splitCommandChain(cmd).find((s) => !isSafeBash(s)) ?? cmd;
          const offenderWords = offender.trim().split(/\s+/);
          // Name the subcommand, not just the binary, when there is one.
          // A bare-binary message ("git" is not in SAFE_PREFIXES) looks
          // identical whether the call was a permanently-blocked write op
          // (`git merge`) or a borderline read op (`git reflog`) — observed
          // live, a model that saw `git log`/`git show` succeed and then hit
          // this same undifferentiated message on `git merge` concluded the
          // whitelist was "flaky" and retried the identical merge call three
          // times instead of recognizing a fixed, principled block. Two
          // words is enough to disambiguate without echoing a long/sensitive
          // command line into the reason string.
          //
          // The second word is only borrowed when it is SUBCOMMAND-SHAPED
          // (SUBCOMMAND_RE below). Everything else — flags, URLs, key=value
          // args, paths, quoted strings — falls back to the bare binary, for
          // two reasons:
          //   1. Secrets. The reason string is model-visible and ends up in
          //      transcripts and benchmark logs, and a refused command's
          //      second token is exactly where a credential tends to sit
          //      (`curl https://user:token@host`, `psql postgres://u:p@h`).
          //      A bare `[a-z]`-ish word can't carry a URL userinfo, a
          //      `KEY=secret`, or a quoted header value.
          //   2. The suggestion has to survive a copy-paste. It is
          //      interpolated into LITTLE_CODER_BASH_ALLOW="<label>", so a
          //      token containing `"`, `$`, or a backtick would either break
          //      the quoting or introduce shell expansion in the line we are
          //      telling the user to run, and a `,` would be split into two
          //      prefixes by parseExtraPrefixes.
          // `git merge`, `npm install`, `docker run` — the cases that
          // motivated this — all still get their two-word label.
          const subcommand =
            offenderWords.length > 1 && SUBCOMMAND_RE.test(offenderWords[1])
              ? offenderWords[1]
              : undefined;
          const offenderLabel = subcommand
            ? `${offenderWords[0]} ${subcommand}`
            : offenderWords[0];
          // Say what to do next, not just what was refused.
          //
          // The bare "not in SAFE_PREFIXES" line sent models hunting: observed
          // live, a refusal on `./build.sh` was followed by `bash ./build.sh`,
          // then `sh ./build.sh`, then a successful `python3 -c
          // "subprocess.run(...)"` — three wasted turns and the guard defeated
          // anyway, because interpreters are themselves whitelisted (issue #94).
          // A refusal that names the actual remedy is the only thing that stops
          // the search, and it is far closer to the decision than a line in
          // AGENTS.md thousands of tokens earlier.
          return {
            block: true,
            reason:
              `shell whitelist: "${offenderLabel}" is not in SAFE_PREFIXES, so this command was refused.\n` +
              `Do NOT try to reach the same effect another way — re-running it through ` +
              `python3 -c, node -e, env, sh, or an -exec flag defeats a limit the user set on ` +
              `purpose, and wastes your budget.\n` +
              `If a file needs changing, use edit/write, which do not need a shell. ` +
              `Otherwise tell the user this command was refused and that they can allow it with ` +
              `LITTLE_CODER_BASH_ALLOW="${offenderLabel}" (or set LITTLE_CODER_PERMISSION_MODE=accept-all), ` +
              `then continue with the rest of the task.`,
          };
        }
      }
    }
  });
}
