# ADR 0003: Stop the r4 re-test at 10 trials and extract every compaction

**Status:** Accepted · **Date:** 2026-10-18 · **Author:** mirstan (with Claude)

## Context

ADR 0002 planned 24 trials, 3 per arm on 4 tasks. After 9 trials the run had averaged about 4.6 h per
trial, against the planned 3.3 h, and the remaining 15 trials needed about 65 more hours. A timing
analysis (`scratchpad/r4-speedup.md`) found that decode is 77% of trial time and that a compaction
point arrives every 1.7 to 3.5 h, whatever per-trial cap is used. Shortening trials would save little.
Only collecting fewer points saves real time. Collecting more points also helps the test only a
little: at a true odds ratio of 2.0, the chance of passing is 0.50 at 16 points and 0.54 at 41.

The frozen `extract.py` also cannot read r4 as ADR 0002 intends. It reads the r3 ledger and only the
first compaction of each trial, so it would yield 7 of the r4 points, where ADR 0002 item 1 says every
compaction point enters.

This ADR is written before any r4 point has been extracted, judged or labelled. The stopping rule
depends only on elapsed time and the count of compaction points, never on rewards, summaries or
labels.

## Decision

1. **Data.** The r4 data is trials 0 to 9 of `~/tb21-runs/compaction-ab-r4/schedule.json`, the first
   10 rows of its `ledger.jsonl`. They give 22 compaction points, 11 per arm, and each has its
   pre-compaction payload saved and at least one action after it. Trial 10 started at 21:19 on
   2026-10-18, 13 minutes before this decision, and is excluded however it ends. Trials 11 to 23 are
   not run. The driver's STOP file halts the schedule.
2. **Extraction.** `extract.py` takes the ledger path and the number of trials as arguments, and treats
   every compaction in a trial as its own point:
   - The transcript the summariser of compaction k replaced is the previous compaction's summary,
     followed by the entries from that compaction's first kept entry up to compaction k. For the first
     compaction it is everything before it, as in r3.
   - The window after compaction k is its first 20 actions, stopping early at the next compaction.
   - A point with no action after it is dropped.
   - `key.json` records the compaction number of each point.

   With its default arguments the script gives output byte-identical to the frozen r3 version (checked
   on 2026-10-18). The run is `extract.py ~/tb21-runs/compaction-ab-r4/ledger.jsonl 10`.
3. **Unchanged from ADR 0002.** The drafting, tie-break, origin, judging and labelling pipeline, the
   both-labellers rediscovery rule, and the validity test (OR ≥ 2 and two-sided Fisher p < 0.05,
   pooled over all vital facts) are unchanged. So is the rule that the analysis runs once.
4. **Known deviations carried over.** ADR 0002 records the retry path that once stopped the harness
   from compacting again (trial 0) and `trial_check.py`'s false budget void. Neither changes which
   points enter.

## Consequences

- The run ends about 56 h earlier than planned. GPU time is about 55 h for trials 0 to 9 (2026-10-16 14:33 to 2026-10-18 21:19), plus trial
  10 until it is stopped.
- The points are unevenly spread by task: schemelike-metacircular-eval 14, extract-moves-from-video 6,
  mailman 1 and video-processing 1. Prefix-cache has no point on mailman or video-processing. The
  validity test pools all vital facts, so this does not decide it. A per-task reading of ADR 0001's
  recall/precision gate would have no prefix-cache point on those two tasks.
- Some windows are shorter than 20 actions: 5 for video-processing in trial 7, 9 for the third point
  of trial 2, and 14 for mailman in trial 5. Fewer actions means fewer chances for a rediscovery to be
  labelled.
- Points from the same trial share a task and a trajectory, so they are not independent. ADR 0002's
  pooled test already treats facts, not trials, as the unit.
