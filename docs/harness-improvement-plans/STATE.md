# little-coder harness-improvement: state snapshot

Last updated: 2026-09-19 23:35 EDT

## 1. The original "8 ranked harness issues" plan — COMPLETE

Source: trajectory analysis of the 2026-09-14 failing batch (5 tasks: gpt2-codegolf,
overfull-hbox, raman-fitting, reshard-c4-data, write-compressor). Full design in
`~/.claude/plans/calm-churning-wigderson.md`. Every branch is merged into `dev`:

| Fix | PR | Status |
|---|---|---|
| Step 0: deadline-triggered snapshot | #29 `feat/shell-proxy-snapshot` | MERGED |
| #1/#2/#8a: error stop_reason retry + log-preview truncation + error logging | #37 `fix/error-stop-reason-retry` | MERGED |
| #3: tb-finalize-guard Trigger C | #33 `fix/tb-guard-trigger-c` | MERGED |
| #4: adaptive thinking-budget | #36 `fix/adaptive-thinking-budget` | MERGED |
| #5: checkpoint shell-write coverage | #32 `fix/checkpoint-shell-writes` | MERGED |
| #6: shell-contract nudge | #35 `fix/shell-contract-nudge` | MERGED |
| #7: temporal-research trigger | #34 `fix/temporal-research-trigger` | MERGED |

Deferred (still not done, no evidence yet to justify the work): #8b (dynamic
max_tokens sizing — needs a vendor patch), a true vendor-level fix for #6 (bg-shell
nudge is a mitigation, not a root-cause fix).

Between that plan and now, several more rounds of fixes also landed on `dev`
(not part of the original 8, found via later review/analysis passes): PR #38/#39
(permission-gate/shell-write bypass closures), #40/#41/#42 (omlx metrics + syntax-error
tracking), #43 (thinking-budget abort-spiral fix), #45/#46 (ShellSession byte-cap +
harbor overflow capture), #47 (quality-monitor escalating loop-breaker), #48
(Trigger A correctness wording), #49 (finalize-warn steer delivery), #50/#51
(GAIA + Harbor session-start snapshots), #52 (truncated-view annotation).

**Net effect: essentially every finding from the first trajectory-analysis round has
shipped.** This is the direct cause of the validation-pass result below — one
previously-hard-failing task now passes.

## 2. #11 ObservationPack / shell-retention — PR #53, MERGED

`feat/observation-pack-context-demotion`. Demotes stale, oversized shell tool
call/result pairs out of live context, archives them host-side, recallable via a
new `ShellRecall` tool with bounded (`readRange`) paging.

Went through 3 rounds of OCR/`/code-review`/Cubic findings (tail-boundary bug,
unbounded-footer bug, unanchored marker-match bug ×2, archive-id collision,
whole-file `ShellRecall` read, archive never cleaned up on shutdown, `readRange`
failure silently swallowed) — all reproduced, fixed, and mutation-tested. A final
`/code-comments` pass over the entire PR diff found no issues. **Merged into `dev`
2026-09-20 03:33 UTC as `7a82719`**, on explicit user instruction. Remote branch
`feat/observation-pack-context-demotion` left intact (not deleted — no cleanup
instruction given).

Deep-research (this session, twice) confirmed no drop-in-compatible prior art beats
this: NVIDIA SoL-Pi's own ObservationPack is the closest precedent but not
reusable; pi.dev's `pi-sliding-context-window`/`pi-context-prune` need external DBs;
Honcho is a different problem (cross-session user memory); **`awesome-jev`
(2026-09-19 deep-research) is a dead end — "Jev" is a single proprietary hosted
decision-API product (TypeSafe AI's System One), every context-management tool
built on it ships data to a paid cloud endpoint, a hard mismatch with little-coder's
local-only, zero-dependency posture. Nothing there to adopt.**

## 3. Validation pass — COMPLETE (`benchmarks/harbor_runs/2026-09-19__13-30-47`)

Re-ran the 5 originally-failing 2026-09-14 tasks on current `dev`
(`omlx/tiel-coder-oq4e`, apples-to-apples) to measure the cumulative effect of
section 1's fixes:

| Task | 2026-09-14 | 2026-09-19 (post-fix) |
|---|---|---|
| write-compressor | 0.0 | 0.0 |
| gpt2-codegolf | 0.0 | 0.0 |
| raman-fitting | 0.0 | 0.0 |
| overfull-hbox | 0.0 | 0.0 |
| reshard-c4-data | 0.0 | **1.0** |

Mean 0.2, 1/5 recovered, 0 regressions. Finished 2026-09-19 23:03:26, confirmed via
direct manual checks (server health, reward.txt, process exit, result.json) twice.

## 4. Second trajectory-analysis round — COMPLETE, produced a NEW ranked list

4 Fable-5 subagents (general-purpose, model=fable) analyzed the 4 still-failing
2026-09-19 trajectories in full (raw `little_coder.log`, verifier output, task
files). Verdict distribution: **3 of 4 (write-compressor, raman-fitting,
gpt2-codegolf) are genuine model competence gaps** — sub-agents were explicitly
adversarial toward their own first impression and still concluded scaffolding
performed correctly in those three. **Only overfull-hbox has real shared blame**,
with a harness-caused event (30s exec-timeout SIGKILL mid-file-write) as the
proximate cause of the actual failing assertion.

See `PLAN.md` in this directory for the resulting ranked improvement list and next
steps — this supersedes `calm-churning-wigderson.md` as the active plan.

## 4a. Priorities 1-5 implementation round (2026-09-20)

Fable design → Fable adversarial critique (verdict: ready with 3 amendments,
all folded in) → 3 implementer agents in isolated worktrees:

| Branch | Worktree | Status |
|---|---|---|
| `fix/log-preview-honesty` | `little-coder-log-preview` | **Done, independently verified.** 2 commits (`fd2b4a7`, `f3cfa38`). Tests: 316 pytest + 30 vitest, typecheck clean. |
| `fix/exec-timeout-truth` | `little-coder-timeout-truth` | **Done, independently verified**, incl. a from-scratch mutation test of the anchored regex (had to de-anchor both the pattern AND the `.fullmatch()` call site together — either alone is independently sufficient, confirming real defense-in-depth). 2 commits (`f88f753`, `4c8faab`). Tests: 381 pytest + 30 vitest, typecheck clean. |
| `feat/upfront-env-limits` → `feat/advertise-initial-snapshot` (stacked) | `little-coder-env-snapshot` | **Done, independently verified**, incl. reading the restructured `_snapshot_initial_state_inner` to confirm a host-side download failure no longer nullifies a successful container-side stage, confirming the restraint wording is verbatim in both the prompt and `finalize-message.ts`, and confirming `gaia`/`default` finalize branches don't leak the new text. 2 commits (`fa22e0f`, `9777f98`), stacking confirmed via `git merge-base --is-ancestor`. Tests: 312 pytest + 155 vitest, typecheck clean. |

All 5 priority items (1, 2, 9, 10, 3) now implemented across 3 branches, pushed, and
opened as 4 PRs (2026-09-20 04:35 UTC): **PR #54** (`fix/exec-timeout-truth`→`dev`),
**PR #55** (`fix/log-preview-honesty`→`dev`), **PR #56** (`feat/upfront-env-limits`→`dev`),
**PR #57** (`feat/advertise-initial-snapshot`→`feat/upfront-env-limits`, stacked on #56).

Full review pipeline (security-review, OCR, adversarial code-review) run on all 4 via
subagents, each with authority to fix/push. Results: #54 clean (no High/Medium), #55
one comment fix pushed (`dd8c725`), #56 one comment fix pushed (`4c0c4d2`), #57 clean.
Stacked branch rebased onto #56's new tip after the fix, force-pushed clean.

**Cubic's automated pass then surfaced 15 more real findings** across #54 (6), #56 (7),
#57 (2) — a reminder that Cubic's own "pass" check status does NOT mean zero unresolved
threads (same lesson as PR #53). #55 had zero. Two fix agents dispatched for #54/#56;
#57's two findings (a hung-download timeout nullifying the already-staged snapshot
outcome — genuinely serious, mutation-tested fix in `977f72d`; and a one-word wording
drift between the prompt and finalize-message's restraint clause) fixed directly and
verified, both threads replied+resolved via GraphQL.

A dedicated `/code-comments` pass (per explicit user request) then caught 2 more real
issues the review pipeline missed: a byte-cap-safety comment duplicated within the same
file across all 3 formatters in #55 (`84895ce`), and a provenance aside ("as this used
to be ordered") in #56 (`8f0dae9`). Cubic re-scanned each new push and surfaced 2 more
genuine findings — an overclaimed "cleanup did NOT run" warning on both the TS and
Python confirmed-kill paths (a shell TERM trap can run before the process actually
dies, fixed in #54's `80f3ae7`), and test coverage my own #57 fix had dropped for the
outer `_INITIAL_SNAPSHOT_TIMEOUT_SEC` backstop (restored in `7f8c8d8`).

The stacked branch (`feat/advertise-initial-snapshot`, #57) needed rebasing onto
`feat/upfront-env-limits` (#56) **twice** as #56 picked up its own fix commits — the
first rebase hit a real conflict (both PRs touch the same `run()` prompt-composition
block) requiring a careful 3-way merge, not a trivial one; the second was clean.

**Second `/code-comments` round (2026-09-20 09:00-13:00 UTC), this time via 4 dedicated
Opus subagents** (one per PR, explicit user request) rather than folded into other
review passes — found real issues past everything above on 3 of 4 PRs:
- **#54** (2 more rounds, `776f705` then a self-initiated advisor-confirmed pass):
  a factually-wrong marker-size claim was NOT here (that was #55) — #54's finds were
  overclaimed-as-fact timeout-path assertions and comments paraphrasing the warning
  strings they sat beside; all fixed.
- **#55** (`8696dd4`): a marker-size comment claiming "~90 chars" that was actually
  78-83 (verified by hand: `len(marker)` for n=1/123/123456) — deleted rather than
  corrected to avoid drifting out of sync with `marker_reserve` again; plus two
  "unchanged from before this fix" wrong-place clauses in test docstrings.
- **#56** (`8961d5f`, then one more I caught myself in `3d04b69`): a stale docstring
  enumeration missing the newly-added toolchain probe, found and fixed by the Opus
  agent in TWO places (`run_harness`'s docstring) — but a THIRD, near-identical
  enumeration in `_wrap_command`'s own separate docstring made the exact same claim
  and was missed; I found and fixed it directly after independently verifying the
  agent's other 3 fixes.
- **#57** (`27b0673`): a self-contradicting docstring (claimed a failed download
  "degrades to no snapshot" in one paragraph, four blocks after another paragraph
  this same PR added saying the opposite), a stale "bounded at 60s" now that the
  constant is 75s (two sites), and a test docstring asserting a false claim about
  what regression it guards against — the agent verified this empirically by
  temporarily deleting the code the claim was about and confirming the test still
  passed. Real due diligence, not just re-reading the diff.

Each of #56's and #57's fixes triggered another rebase of the stacked #57 branch
(now 3 total rebases across the whole PR #53-adjacent round) — the last one hit TWO
conflicts (both expected: the shared `run()` prompt-composition block, and a
60s-vs-75s wording collision from the two branches' independent comment fixes),
resolved by hand, re-verified, force-pushed clean.

**Third `/code-comments` round (2026-09-20 12:50-14:10 UTC), 4 Fable subagents**
(explicit user request, same procedure as the Opus round) — found real issues on
3 of 4 PRs, including the single most consequential finding of any review round:

- **#54** (4th round, `cae8bce`): two more unverifiable-mechanism claims stated as
  fact — a comment asserting the parent TB adapter's tmux session "never answered
  within its own timeout" when the actual failure mode is broader (no usable
  response, cause unknown), traced through pi's RPC dialog-timeout code and the
  Python handler to confirm; and a stub-cleanup comment warning about a sibling
  test's `importorskip("terminal_bench")` that doesn't exist anywhere in the repo.
- **#55**: no findings — thorough, numerically-verified pass (recomputed every
  byte/KB claim by hand), confirming the prior 3 rounds had genuinely exhausted it.
- **#56** (`fa98afc`) — **the significant one**: the model-facing
  `_HARD_LIMITS_PARAGRAPH` itself claimed "each ShellSession call is killed at its
  timeout... a killed command does not run its cleanup" — stated as settled fact,
  and FALSE for the harbor docker backend. The agent read harbor's actual
  `DockerEnvironment.exec`/`_terminate_process` source and reproduced it live
  against a real container (both plain `docker exec` and `docker compose exec`):
  the host-side client dies at t=2s, but the in-container command keeps running to
  completion and its post-timeout statements still execute. I independently
  re-verified the mechanism claim myself by re-reading the installed harbor
  package's source directly. Rewrote the paragraph to the verified truth (call
  fails, output is discarded, but the command itself is NOT killed and may finish
  invisibly later) — concrete, actionable, and consistent with PR #54's
  already-correctly-hedged warning text (which said "connection was killed," never
  "process killed" — no cross-branch fix needed there). Plus 3 smaller fixes
  (a probe-timeout-shape claim, a 4th instance of the recurring stale-enumeration
  bug, one "silently" overstatement).
- **#57** (`7bd0d3e`): a 4th instance of the restraint-clause wording drift (this
  time "it" vs. "there"), a stale placement comment left over from before this PR
  landed, and two test-docstring accuracy fixes — one of which the agent verified
  empirically by temporarily deleting the code a docstring described and
  confirming the test still passed (proving the docstring's claim false) before
  fixing it.

This forced a 4th rebase of the stacked #57 branch onto #56 — clean this time,
no conflicts.

**MERGED (2026-09-20 15:11 UTC), on explicit user "Merge them all" instruction.**
Order: #54 (`8d364b1`) → #55 (`678c41e`) → #56 (`804e5d3`) → #57 (`1371385`).
#57 needed its base retargeted from `feat/upfront-env-limits` to `dev` first
(GitHub doesn't auto-retarget a stacked PR just because its base branch merged
and wasn't deleted) — confirmed the retargeted diff still showed only #57's own
4 files before merging. `dev`'s CI passed on all resulting merge commits. Key
fixes spot-checked present in `origin/dev` post-merge (the corrected
`_HARD_LIMITS_PARAGRAPH`, the aligned restraint-clause wording, the
download-timeout fix). Remote feature branches left intact (not deleted — no
cleanup instruction given), matching PR #53's precedent.

All 5 priority items (1, 2, 9, 10, 3) from the second trajectory-analysis round
are now live on `dev`. This closes out the priorities-1-5 implementation round
started earlier — see `PLAN.md` for what's still open (priorities 6-11).

Minor follow-up noted, not yet actioned: `.pi/extensions/shell-session/helpers.ts:2`'s
header comment still references a nonexistent predecessor file
(`local/tools/shell_session.py`) — pre-existing staleness, unrelated to this
round, correctly left out of scope by the implementer.

## 4b. Priority 6 (LSP-alternative syntax-check) implementation round (2026-09-20)

`docs/harness-improvement-plans/PLAN.md` and `STATE.md` (this file) were copied
out of the session scratchpad and committed into the repo — branch
`docs/harness-improvement-tracking`, worktree `little-coder-plan-docs`, not
pushed — so the ranked list and this summary survive past the session.

The carried-over LSP-alternative design (`syntax-check-lsp-alternative.md`, still
scratchpad-only) went through a Fable adversarial critique against post-#54-57
`dev` — verdict ready with 5 amendments, all folded into the doc (footer-parsing
reuse, a real conflict with `truncated-view` caught and fixed, drifted line
references corrected, the evidence baseline flagged for re-measurement, exit-code
parsing detail). Then implemented by an Opus subagent in worktree
`little-coder-syntax-check` (branch `feat/syntax-check`): new
`.pi/extensions/syntax-check/` (Phase 1 MVP, 4 languages), `_shared/tb-proxy.ts`
extracted from `shell-session` for reuse. The implementer proved the design's one
unverified assumption (`ctx.ui.input` from a `tool_result` hook) with a real
end-to-end smoke test before writing any checker logic, and ran 9 mutation tests
of its own.

I independently re-verified: reinstalled deps, re-ran the full suite myself,
re-mutation-tested the diagnostic-above-footer insertion (the critique's real
`truncated-view` conflict fix) by hand, and read both core source files in full
for a `/code-comments` pass (no findings). Opened as **PR #58**.

While verifying, found `.pi/extensions/skill-inject/injection.test.ts` had 3
failing tests on plain `dev` itself — confirmed pre-existing and unrelated
(reproduces without PR #58's changes). Root cause: the test file regex-extracts
harbor's prompt template to test research-directive injection against the real
boilerplate, and PR #56/#57's `_compose_prompt` refactor broke that regex.
**Not caught by CI** — `.github/workflows/ci.yml` runs only `npm ci` and
`python -m pytest benchmarks/`, no `npx vitest run` step at all, so the entire
1000+-test vitest suite has zero CI enforcement; this bug was invisible until a
manual full-suite run surfaced it. Fixed in a separate isolated worktree
(`little-coder-skill-inject-fix`, branch `fix/skill-inject-test-prompt-regex`),
mutation-tested, opened as **PR #59**.

## 4c. PRs #58, #59, #60 merged (2026-09-20)

On explicit user instruction ("merge 59, then 60, then 58, if cubic is clear"),
gated on each PR's Cubic scan being clear and zero unresolved review threads.

- Before merging, ran a further round of work per user request: 3 parallel Fable
  subagents (one per branch) applied the `/code-comments` methodology to each
  branch's own diff against `origin/dev`. #58 got 2 real fixes (an overstated
  "must never surface" claim contradicted by its own perl checker; an
  absolute "never appears" claim about resolved paths that was really "need not
  appear"). #59 got 1 real fix (a docstring overstating what it reconstructs,
  plus refactor-provenance language that belonged in the commit message, not
  the comment). #60 had no findings — its diff adds no comments, and the fix
  it landed made a pre-existing comment true rather than leaving it false.
- Those comment-fix pushes triggered fresh Cubic scans, which caught 2 more
  real findings on #58 (a new test that didn't actually exercise the
  basename-fallback branch it claimed to — confirmed via mutation test; a
  fail-open comment that named only perl's exception when gcc/cc's
  `-fsyntax-only` also resolves `#include` headers). Both fixed, mutation
  where applicable, replied+resolved.
- **Lesson learned, worth remembering**: a GitHub Actions "rerun" replays the
  *original* commit SHAs — it does not re-test against an updated base branch.
  PR #60's `vitest` job kept showing red after #59 merged even though `git
  rerun` was used, because the rerun wasn't actually testing against merged
  `dev`. Fixed by rebasing `ci/add-vitest` onto `origin/dev` and
  force-pushing, which triggers a genuine new merge-ref test. Confirmed green
  both locally (1158/1158 tests) and in the new CI run before merging.
- Merge order: #59 (`343ea47`) → #60 → #58, all via `gh pr merge --merge`
  (this repo's established convention, matching PRs #54-57's merge-commit
  style). All three fully green and thread-clean at merge time.

## 4d. Priorities 7-10 implementation round (2026-09-20)

Fable design → Fable adversarial critique → Opus implement, same pipeline as
priorities 1-6. Two grouped design tracks, chosen by shared file surface:
Group A (priority 7 + 8, both touch `tb-finalize-guard`'s trigger logic) and
Group B (priority 9 + 10, both extend `quality-monitor`'s detection state).
Before implementing, evaluated the user's suggestion of using "jev" (a hosted
TypeSafe AI "typed decision" classifier) for Group B — confirmed via direct
research it's the wrong tool (a structured yes/no/score classifier, not a
similarity/embedding engine; its own ecosystem's docs say "reranking alone did
not beat vector retrieval") and architecturally unfit regardless (an external
API call from inside an hours-long, sandboxed, unattended trial container is
exactly the dependency this harness has avoided everywhere else). Declined,
recorded in both designs' critique passes.

Both critiques came back "ready with N amendments," no redesign needed — same
shape as priority 6's round. Group A: 4 amendments (Trigger D's deliverable
wording, an endgame-suppression gap, a standing-vs-reactive baseline-diff
tension, a missing Python test fixture). Group B: 4 amendments, the headline
being that the failure-signature watchdog as designed would have been INERT
on Terminal-Bench — the exact task type it was built for — because it gated
on `isError`, which `ShellSession` (TB's shell path) never sets.

Implemented as PR #61 (priority 7), PR #63 (priority 8, stacked on #61 —
real file conflicts if built in parallel), PR #62 (priorities 9+10, one
branch). Post-implementation review was unusually extensive:

- **PR #62** went through 4 rounds of Cubic findings before landing clean:
  round 1 found 15 issues (14 real fixes + 1 resolved by independently
  re-deriving the bigram-Jaccard math from first principles, concluding no
  code change was needed but for a different reason than either side
  claimed — the min-token floor does the work, not the similarity
  threshold); round 2 found 8 more, including a genuine functional bug (a
  PASSING result could reach the model mislabeled as a failure, verified
  end-to-end); round 3 found 1 more (`identityKey` silently disagreed with
  the loop-breaker's own `sameCall` on object-key ordering, letting a
  reordered-key repeat evade both detectors — an existing test had directly
  pinned the bug). Round 4 was clean. Total: ~23 real bugs found and fixed
  across the round, on top of the critique's own 4 amendments.
- **PR #61/#63**: a manual Cubic re-scan (triggered after a mid-review
  force-push confused its incremental-review mode — had to be re-triggered
  via a PR comment, twice, once per repo's own quirk) found the same
  snapshot-var-leak bug independently in `harbor_adapter`'s `_pi_env`
  that priority 8's implementation had already fixed in `tb_adapter`'s —
  neither adapter explicitly cleared `LITTLE_CODER_INITIAL_SNAPSHOT`, so an
  omitted key could inherit a stale value from an earlier trial in the same
  worker process (`PiRpc` builds the child env as `dict(os.environ)`
  updated with the adapter's own dict — an omitted key passes an inherited
  one through untouched). Also found: Trigger D and Trigger A could queue
  two different steers on the same turn (pi delivers every queued message,
  never coalesces); Trigger C was missing the same baseline caveats and
  restore-restraint warning Trigger A needed. One fix was applied to the
  wrong branch initially (PR #63 instead of PR #61, where the finding
  actually lived) — caught, the commit was moved to #61 via cherry-pick,
  and #63 was re-rebased on top; this repeated several times as both
  branches kept independently accumulating fixes, each requiring a fresh
  rebase (once hitting a real merge conflict from priority 8's own
  `_pi_env` signature change, resolved by hand).
- A final `/code-comments` pass (3 parallel Fable agents, one per branch)
  found 9 more real stale/overstated comments total (4 on #61, 2 on #62, 3
  on #63) — all comments left behind by earlier fix rounds that no longer
  described the final code accurately. Comment-only, no logic changes.

**MERGED (2026-09-20), on explicit user "merge 61, then 63, then 62"
instruction.** Order: #61 → #63 → #62, all via `gh pr merge --merge`. All
three fully green and thread-clean at merge time.

## 6. Re-validation of the separate 8-issue plan (2026-09-20)

`~/.claude/plans/calm-churning-wigderson.md` — a Phase 1-4 design for 8 items
from an EARLIER trajectory analysis (5 Fable-5 subagents), distinct from this
`PLAN.md`'s own 2026-09-19-dated ranked list. Earlier the same day, this plan
was checked and reported as "never executed — no matching branches or PRs."
**That check was wrong.** Re-checked properly this time (traced each item to
actual current code, not just a branch-name grep) — all 8 items are done:

| # | Item | PR | Merged | Notes |
|---|---|---|---|---|
| 1 | Retry on `stopReason=="error"` | (in `rpc_client.py`'s stop_reason machinery) | pre-09-19 | `result.stop_reason="error"` + `error_message`, provider-error vs. empty-completion distinguished |
| 2 | Fix `[:400]` log-preview truncation | #55 | 2026-09-20 | Confirmed earlier same day |
| 3 | tb-finalize-guard Trigger C | (bundled into tb-finalize-guard's history) | pre-09-19 | Exists verbatim: "dead run: errored or empty final message, any time" |
| 4 | thinking-budget deadline/latency-adaptive | **#36** (`fix/adaptive-thinking-budget`) | 2026-09-13 | + 7 follow-up hardening commits through 09-14. Current code's residual-risk comment (300s HTTP idle timeout backstop) matches this plan's own critique amendment #4 almost verbatim. |
| 5 | Checkpoint shell-write snapshot coverage | (bundled into checkpoint's history) | pre-09-19 | `SHELL_TOOLS`/`ShellSend` handled, same `cd`-tracking caveat this plan flagged |
| 6 | Shell-contract-nudge (pkill/bg mitigation) | **#35** (`feat/shell-contract-nudge`) | 2026-09-13 | Literally the directory name this plan proposed, + a ReDoS fix and a rewrite since |
| 7 | skill-inject temporal-research trigger | (bundled into skill-inject's history) | pre-09-19 | `TEMPORAL_TRIGGERS` exists, matches the design |
| 8a | Error payload logging | (bundled into adapter history) | pre-09-19 | `errorMessage`/`auto_retry_start`/`auto_retry_end` handled in both adapters |
| 8b | Dynamic `max_tokens` sizing | — | not done | Still deferred, per the plan's own recommendation (needs a vendor patch, no evidence justifies it) |

All of it pre-dates this `PLAN.md`'s own 2026-09-19 draft date — meaning the
2026-09-19 trajectory analysis that produced this `PLAN.md`'s ranked list ran
without re-checking whether these 8 items were already addressed, and this
`PLAN.md` in turn carried a stale "not started" status for priority 11
(the same as item 4 here) all the way through 2026-09-20 until this
re-validation caught it. **Lesson, worth remembering**: a "no matching
branches" check is only as good as the exact names searched — `gh pr list
--search <keyword>` and reading the actual current code both found what a
`git branch` grep missed.

## 5. Open loose ends

- Priority 12 in `PLAN.md` is the only genuinely unstarted item across both
  this list and the calm-churning-wigderson.md plan (see section 6). Confirmed
  no bg-shell "stale process surfacing" feature exists.
- Item 8b from the calm-churning-wigderson.md plan (dynamic `max_tokens`
  sizing) remains explicitly deferred, per that plan's own recommendation —
  needs a vendor patch via `scripts/patch-pi.mjs`, no evidence justifies it.
- Re-run `benchmarks/syntax_error_report.py` on a fresh write-compressor trial
  to confirm the 24.5%/2026-09-19 evidence baseline for priority 6 still holds
  post-#54-63 — still outstanding, needs a live harbor trial, not a code change.
