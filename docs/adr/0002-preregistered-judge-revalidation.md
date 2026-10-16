# ADR 0002: Pre-registered re-test of the compaction-summary judge

**Status:** Accepted · **Date:** 2026-10-16 · **Author:** mirstan (with Claude)

## Context

ADR 0001's offline pass on the 7 r3 compaction points did not establish judge validity. The ADR had
not said how to combine two rediscovery labellers. The consensus (both labellers agree) gave OR 2.03,
p = 0.073. The Claude labeller alone passed (OR 2.42, p = 0.023), and the GPT labeller alone did not
(OR 0.82). Picking a labeller after seeing those numbers would be a post-hoc choice, so the rule is
fixed here, before new data exists, and tested on compaction points that r3 did not produce.

## Decision

Everything below is frozen as of the commit that adds this file. The scripts in
`benchmarks/compaction_judge/` at that commit are the protocol. Any change to them or to this list
needs a new ADR, written before the data it applies to is analysed.

1. **Data.** A new A/B run with both arms (`prefix-cache`, `blackhole`), about 7 trials per arm, on
   tasks whose context passes the 200K compaction trigger. Every compaction point from those trials
   enters, including second and later compactions in a trial. r3 points are never pooled in.
   Exclusions, as in ADR 0001: a point whose pre-compaction transcript is not fully saved, and a point
   with no actions after it. The target is at least 14 usable points. If fewer exist, the test runs on
   what exists and the shortfall is reported, without adding trials after the analysis has run.
2. **Pipeline, unchanged from the r3 pass and with no human step.** `extract.py`: blinded IDs,
   thinking dropped, drafter tool results capped. Drafting: Claude Opus 5.5 and GPT (Codex CLI, its
   configured model). Then `match.py`, `tiebreak.py` (Qwen 3.8 27B on omlx), `fable.py` for quotes
   not found, and `origin.py` (Fable; drafter-origin and false facts dropped), then freeze. `judge.py`
   recall: 3 samples from each family. A "yes" counts only with a verified 24-character quote, and a
   fact is present when most of its 6 samples say so.
3. **Rediscovery rule.** Claude Opus 5.5 and GPT each run `label.py` once over the first 20
   post-compaction actions. A fact is rediscovered **only when both labellers say yes**.
4. **Test.** `validity.py` runs once, after every usable point has been judged and labelled. It pools
   all vital facts and builds a 2×2 table of judged missing against rediscovered. The judge is valid
   when OR ≥ 2 **and** two-sided Fisher p < 0.05. Per-point and per-arm tables are reported but
   decide nothing.
5. **What follows.** If the judge is valid, ADR 0001's recall/precision gate is applied to the new
   points, and its offline verdict counts. If it is not valid, LLM judging of summaries is dropped
   for this question, and the choice is between paired reruns and stopping.

## Consequences

- The run must not start before the r3 compaction-hang fix is merged, because a hung trial loses its
  compaction point (r3 trial 6).
- About 14 trials at about 3.4 h each is roughly 2 days of GPU time. omlx must not be restarted
  mid-trial, so the Qwen tie-break runs only after the A/B run finishes.
- Model and CLI versions at registration: `claude-opus-5-5`, `claude-fable-5-1`, Codex CLI 0.155.1
  serving `gpt-5.6-luna`, omlx 0.7.0 with `dirk-q38-27b-oq4e`. If one has changed when the
  analysis runs, the change is reported, and the analysis runs anyway.
- `validity.py` reproduces the r3 result (OR 2.03, p = 0.073, not valid). That reproduction is a check
  of the script, not evidence for either outcome.

## Amendments (2026-10-16, before the run starts and before any new data exists)

- **Tasks.** The run uses no task from the TB2.1 `test` split, so that split stays held out. In 266
  earlier trials, only four non-test tasks ever reached the compaction trigger:
  `schemelike-metacircular-eval` (2 of 3 trials), `extract-moves-from-video` (2 of 3), `mailman` (1 of 2)
  and `video-processing` (1 of 2). The run uses those four.
- **Trials.** Each task runs 3 times per arm: 24 trials, 12 per arm, in place of about 7 per arm. At
  the rates above that gives about 14 compaction points. Each repeat is its own round, and the arm
  order inside a round is shuffled with seed 20261016 (`make_schedule.py --seed 20261016`).
- **Code.** Branch `exp/compaction-ab-r4`. It is r3's experiment code merged with `dev` at `d2d8936`,
  which includes the compaction-hang watchdog (PR #90). The driver checks that HEAD is the frozen SHA
  before every trial.
