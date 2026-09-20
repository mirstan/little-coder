# Plan: second-round harness improvements (2026-09-19 trajectory analysis)

Status: **priorities 1-6 merged (PRs #54-58, 2026-09-20); priorities 7-12 not yet
started.** See `STATE.md` for how this fits into the overall project.

## Source

4 Fable-5 subagents independently analyzed the 4 still-failing trajectories from
`benchmarks/harbor_runs/2026-09-19__13-30-47` (write-compressor, gpt2-codegolf,
raman-fitting, overfull-hbox — reshard-c4-data passed this run, excluded). Each
read the full raw log, verifier output, and task files, and was explicitly briefed
to be skeptical of scaffolding excuses. Full per-task reports are in this session's
transcript; this file merges their ranked-improvement sections.

## Verdict distribution (important framing)

3 of 4 (write-compressor, raman-fitting, gpt2-codegolf) are **genuine model
competence gaps** — the sub-agents independently concluded the harness performed
correctly and the model simply lacked the skill (couldn't invert an adaptive
arithmetic coder; confirmation-biased its way past two correctly-identified peaks
in a spectroscopy fit; shipped a transformer with no attention mechanism). Only
**overfull-hbox has real shared blame**: a harness-caused event (exec-timeout
SIGKILL mid-file-write) directly produced the failing assertion.

This means items 1-3 below are the highest-confidence, highest-leverage fixes
(they address the one trajectory where the harness demonstrably caused failure).
Items 4+ are lower-confidence generalizations from patterns seen in the
competence-gap trajectories — real gaps, but fixing them won't flip those specific
trajectories to passing (the model would still fail the task).

## Carried-over item: pre-execution syntax diagnostics (LSP alternative)

`syntax-check-lsp-alternative.md`, same directory. Origin is **separate from this
file's 2026-09-19 trajectory analysis** — it's a Fable Plan agent's earlier
LSP-integration research from mid-session, recovered from the transcript after an
explicit "locate the lsp integration plan that wasn't executed" request, and never
folded into this ranked list until now. Checked 2026-09-20: every *other* design
doc in this directory (`11-observation-pack-context-demotion.md`,
`9-truncated-view-annotation.md`, `checkpoint-session-start-snapshot.md`,
`finalize-warn-followup-to-steer.md`, `quality-monitor-escalating-loop-breaker.md`,
`shell-output-byte-cap-fix.md`, `shell-output-overflow-capture-phase2.md`,
`trigger-a-correctness-nudge.md`) maps to an already-merged PR — their own status
headers say "not yet implemented" but that's stale, written before implementation.
Spot-verified two of the less obviously-titled ones directly against code
(`checkpoint/index.ts`'s `session_start` handler, `quality-monitor/index.ts`'s
escalation state) to be sure rather than trust filenames. This is the one
genuine carryover.

**What it is**: not an actual LSP — pi has zero LSP machinery and the model has no
write/edit tools in TB mode to keep a client's document model in sync with anyway.
Instead, a one-shot in-container syntax check (`perl -c`, `python3 -c
'ast.parse(...)'`, `gcc -fsyntax-only`, `bash -n`) appended to the `ShellSession`
tool result only on failure, for whichever of 4 languages the model just wrote.

**Why it ranks where it does**: strongest evidence base of anything on this list —
not a single-trajectory anecdote but a *measured, reproducing* failure rate: 24.5%
calls-with-syntax-error on the motivating trial, and confirmed on 2026-09-19 that
the identical trial re-run on current `dev` (with everything else already merged
at that time) still accumulates syntax errors mid-run. Generalizes to Aider
Polyglot's write/edit-tool path for free (verified against `aider_polyglot.py`'s
actual `ALLOWED_TOOLS`). Mechanics are unusually thoroughly pre-verified (exact
hook, exact proxy channel, a real observed failure loop it targets directly — a
Perl syntax error the model rewrote identically twice before finally reading the
line).

**Update 2026-09-20**: sent through a Fable adversarial critique against current
`dev` (post PR #54-57) — verdict **ready with 5 amendments, all folded into
`syntax-check-lsp-alternative.md`**. No amendment was fundamental; every reuse
target (the `tool_result` hook pattern, the proxy channel, `detectDeliverableWrites`,
`checkpointPath`) checked out against the post-merge codebase, though several line
references had drifted and two real conflicts with extensions that landed *after*
this design was first written got caught and fixed (reuse `truncated-view`'s
`splitFooter` instead of hand-rolling footer parsing; insert the diagnostic above
the footer, not appended after, to stay compatible with `truncated-view`'s own
footer-last assumption). One amendment flags that the 2026-09-19 evidence number
should be re-measured before implementation, since PR #54's timeout warnings
plausibly already reduce one subset of the motivating failure — this is a
measurement task, not a design blocker. **This item is now ready to implement,
same as priorities 1-5 were after their own critique passes.**

## Priority order (impact × effort)

The list below is ordered by evidence strength (how directly each item is tied to
an observed failure). Re-ranked here by **suggested implementation priority** —
impact weighed against effort — since evidence strength and effort don't move
together. `#N` cross-references the original item number in "Ranked list" below,
where the full rationale lives.

| Priority | Item (`#N` = original rank) | Impact | Effort | Why this slot |
|---|---|---|---|---|
| 1 | `#1` Exec-timeout kill → "may be half-written" warning | **High** | **S** | Directly caused the one harness-attributable failure (overfull-hbox); one-line append to the shell-session result formatter — **merged, PR #54** |
| 2 | `#2` Fix `timed_out=false` on timed-out commands | Medium | **S** | Same event/commit as #1 — trivial metadata-correctness fix — **merged, PR #54** |
| 3 | `#9` State hard tool-contract limits up front (30s cap, interpreters present) | Medium | **S** | Static injected hint, no new state machine; cost real turns in 2 of 4 trajectories — **merged, PR #56** |
| 4 | `#10` Don't truncate small (<4KB) tool outputs | Low–Med | **S** | Threshold tweak to an existing formatter; narrow but free — **merged, PR #55** |
| 5 | `#3` Advertise the pristine initial-state snapshot to the agent | **High** | M | Would have caught overfull-hbox's corruption instantly; extends checkpoint/snapshot machinery already built (PR #29/#50/#51) — **merged, PR #57** |
| 6 | *(carried over)* Pre-execution syntax diagnostics (LSP alternative) | **High** | M | Strongest evidence of anything remaining — measured, reproducing failure rate (not a single trajectory), generalizes to Aider Polyglot for free. **Merged, PR #58** (2026-09-20), after a second implementation round fixed 8 more Cubic findings (2 declined with evidence, 6 real) and a `/code-comments` pass found 4 more real comment issues. |
| 7 | `#6` Baseline-grounded adversarial re-verification in "don't give up" nudges | Medium | S–M | Directive-text change to existing nudge extensions; pairs directly with #3/PR #57 (a real baseline now exists to verify against) |
| 8 | `#5` Earlier/more frequent deadline-progress nudges (50%/75% checkpoints) | Med–High | M | Extends tb-finalize-guard triggers; needs a "does a runnable deliverable exist yet" heuristic |
| 9 | `#4` Fuzzy/near-duplicate loop detection | Med–High | M | Extends quality-monitor's loop-breaker; similarity matching carries real false-positive risk, needs careful tuning |
| 10 | `#8` Repeated-identical-failure-signature watchdog | Medium | M | New output-hashing state tracked across turns; distinct mechanism from #4 |
| 11 | `#11` Adaptive thinking-budget continuation | Medium | **L** | Cuts both ways (this session saw both starvation and leaked-reasoning failure modes) — needs careful tuning and validation, real regression risk, continuation of PR #36 |
| 12 | `#7` Surface stale background processes touching shared files | Low | M | Only 1 of 4 trajectories; needs bg-shell job-registry + file-path-tracking integration |

`#12` in the original "Ranked list" below (validates PR #53's value) was already
resolved before this update — PR #53 is merged — and isn't part of this
prioritization.

**Reading the table**: priorities 1-6 are done (all merged 2026-09-20, PRs
#54-58). 7 is the next tier — still high-leverage, lower effort, and unblocked
(needs the baseline #3/PR #57 provides). 8-10 are real but generalized gaps,
medium effort with more design judgment required. 11-12 are the
highest-effort/highest-risk or lowest-impact items — 11 in particular should
not be rushed given the direct tension in the evidence.

**Also merged alongside priority 6, not part of this ranked list** (discovered
as drive-by fixes during the priority-6 round, not scored items): PR #59 fixed
a pre-existing `skill-inject` test-harness bug (a test's prompt-reconstruction
regex broke silently when PR #56/#57 refactored the real prompt-assembly code);
PR #60 added a `vitest` CI job — the ~1170-test suite under `.pi/extensions/`
had zero CI enforcement before this, which is exactly how PR #59's bug went
undetected. See `STATE.md` section 4b for both.

## Ranked list

1. **On an exec-timeout kill, append a "command was killed mid-execution — files it
   was writing may be half-modified, re-verify before trusting them" warning to the
   tool result.** Directly caused overfull-hbox's failure: the 30s exec cap SIGKILLed
   a solver script mid-write, the model then backed up the corrupted file as
   "original," and every later verification was self-referential against that
   corrupted baseline. Cheap fix to the shell-session output formatter.

2. **Fix `timed_out=false` being reported on a command that did time out.** Same
   event as #1 — the result footer contradicted its own stderr ("Command timed out
   after 30 seconds" alongside `timed_out=false`), hiding the one signal that would
   have told the model something went wrong. Metadata correctness bug, trivial fix.

3. **Advertise the pristine initial-state snapshot to the agent**, not just keep it
   container-side (`/tmp/.lc-initial`, confirmed present but never surfaced in
   overfull-hbox's trial). Would have let the model catch its own file corruption
   instantly instead of trusting a self-made corrupted backup. Natural extension of
   the checkpoint/snapshot machinery already built (PR #29, #50, #51).

4. **Fuzzy/near-duplicate loop detection**, not just verbatim-repeat matching.
   Verbatim loop-breaker fired correctly in both write-compressor and gpt2-codegolf,
   but missed near-duplicate loops: write-compressor's 12-probe segfault loop only
   varied a numeric constant each time; gpt2-codegolf wrote three ~100-line,
   essentially-identical perl scripts back to back. Extends the existing
   quality-monitor loop-breaker (already escalating past PR #47).

5. **Fire deadline/progress-awareness nudges earlier and more often** (e.g. at
   50%/75% budget used), demanding a concrete runnable deliverable rather than
   waiting for a near-deadline warning. gpt2-codegolf's first time-pressure signal
   arrived at ~96% of budget spent, by which point zero working code existed;
   the only real code was written in the two turns after that (too-late) nudge.

6. **Make finalize/"don't give up" nudges demand baseline-grounded, adversarial
   re-verification, not self-confirmation.** Seen in two trajectories:
   raman-fitting's nudge produced a re-check that just re-confirmed the same wrong
   peak identification; overfull-hbox's nudge-triggered "independent" verification
   diffed a file against its own backup copy (tautological, "changed positions: 0").
   Pairs with #3 — a real baseline is what makes genuine re-verification possible.

7. **Surface still-running background processes touching files the agent is
   working on.** overfull-hbox had a stale `setsid`'d process still mutating shared
   files while the model debugged, confusing its own measurements for many turns.
   Lower priority — only seen in 1 of 4 trajectories — but buildable on the
   existing bg-shell job registry.

8. **Repeated-identical-failure-signature watchdog** ("your last N attempts
   produced the identical error/output — try a different approach"). Distinct from
   #4: write-compressor's inputs varied each attempt but the outcome
   (`count=-512` / the same segfault) repeated 5+ times across ~2 hours with no
   nudge to change approach.

9. **State hard tool-contract limits up front** (the 30s exec cap; which
   interpreters are actually present) via an early environment-probe hint, instead
   of letting the model rediscover them by trial and error. Cost real turns in both
   overfull-hbox (discovering the 30s cap) and write-compressor (~35 turns in
   "Perl hell" after finding no python3).

10. **Don't truncate small (<4KB) tool outputs.** Narrow, cheap fix — write-
    compressor's turn-1 `cat decomp.c` (1262 bytes, the central task artifact) was
    truncated at ~200 chars, costing 3 extra reads.

11. *(Tension, not a clean fix — needs care)* Thinking-budget tuning cuts both
    ways: write-compressor's reasoning leaked into inefficient comment-only scratch
    files with thinking off, but an earlier trial this session showed thinking:high
    starving turns at low tok/s. Argues for finishing out the deadline/latency-
    adaptive thinking-budget work (already partly built, PR #36) rather than a
    blanket toggle change.

12. *(Validates existing work, no new action)* gpt2-codegolf's 232-turn/20.3M-token
    run with 0 compactions correlated with visible late-run repetition/degradation
    — this is exactly the problem PR #53 (shell-retention/ObservationPack) targets.
    No new item; just confirms #53 is worth merging.

## Not investigated by this round (out of scope, noted for completeness)

- `awesome-jev` deep-research (2026-09-19): no adoptable technology found — see
  `STATE.md` section 2 for the verdict. Does not feed into this ranked list.

## Next steps

1. ~~Priorities 1-5~~ — **done**: implemented via Fable design → Fable adversarial
   critique → Sonnet/Opus implement, reviewed (security-review, OCR, `/code-review
   high`, Cubic, and 3 rounds of dedicated `/code-comments` across Opus and Fable
   passes), and merged as PRs #54-57 on 2026-09-20.
2. ~~Priority 6 (LSP-alternative syntax-check)~~ — **done**, merged as PR #58
   (2026-09-20). Still outstanding: re-run `benchmarks/syntax_error_report.py`
   on a fresh write-compressor trial to confirm the 2026-09-19 24.5%
   evidence baseline still holds post-#54-58 — not done, needs a live harbor
   trial, not just a code change.
3. Priority 7 is newly unblocked (PR #57 gives it a real baseline to verify
   against) and is small — could go straight to implementation without a full
   design pass, mirroring how PR #47-#52 were handled.
4. Priorities 8-10 are candidates for the full design→critique→implement pipeline.
5. Priority 11 should be scoped as a continuation of PR #36's adaptive
   thinking-budget work, not a fresh design.
6. Priority 12 is lowest priority — only 1 of 4 trajectories, buildable on the
   existing bg-shell job registry whenever there's room for it.
