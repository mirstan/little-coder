# ADR 0001: Judge compaction summaries offline first, with paired reruns gated on the result

**Status:** Accepted · **Date:** 2026-10-15 · **Author:** mirstan (with Claude)

## Context

The compaction A/B (r1 to r3) compares the `prefix-cache` and `blackhole` arms. Pass/fail cannot
separate them, because neither arm passed the tasks that compact. r3 left 8 compaction points on 2
tasks, and each one keeps its pre-compaction transcript, its summary and the agent's actions after
compaction. The task containers are gone, so no compaction point can be resumed as it was.

The relevant research disagrees on method. Decomposed yes/no checks agree across judges far better
than holistic scores (CheckEval, EMNLP 2025). Pairwise judging is swayed by length (35% preference
flips against 9% pointwise), and our arms' summaries differ about 4x in length. LLM-extracted key
facts were refuted as a substitute for human ones. Agent-compaction papers (TRACE, CliffCompaction,
Complexity Trap) never judge summary text and use only downstream behaviour. TRACE's measure is
paired continuations from the same state.

## Decision

1. Score each summary on its own, never pairwise, with the arm hidden. The primary number is
   vital recall over a frozen vital-fact list. Precision over the summary's own claims and the
   fixed R/P/A checklist are reported beside it. Two judge families (Claude, GPT) each run 3 samples.
   Per-item agreement between the families is reported and low items are flagged, not dropped
   automatically, because alpha on 8 summaries is too noisy to act on alone.
2. Claude and GPT draft vital facts from the pre-compaction transcript only. A human resolves only
   the facts where they disagree. The post-compaction window stays hidden from drafting, so the
   validity test cannot pass circularly.
3. Validate the judge against rediscovery in the first 20 post-compaction actions, labelled per fact
   by an LLM that cites the action.
4. A compaction point is excluded when its pre-compaction transcript was not fully saved to disk.
   Trial 6 of r3 is excluded: its compaction hung, its saved messages stop 54 minutes before the
   compaction, and only 2 actions follow it. That leaves 7 points: doom 2 v 2, gcode 1
   prefix-cache v 2 blackhole.
5. The r3 trials ran independently, so the offline arm comparison is unpaired and descriptive. It
   decides only when the judge is valid (OR ≥ 2, Fisher p < 0.05) **and**, in both tasks, the winning
   arm's lowest vital recall is above the losing arm's highest, without losing more than 10 points of
   precision. Otherwise, run paired reruns. Rebuild each compaction point by replaying its recorded commands in
   a fresh container. Continue it under full history, each arm's summary, and CliffCompaction, for
   20 actions × 3 seeds, and compare burden.

## Consequences

- The offline pass needs no GPU and about 1 hour of human review. With 7 unpaired points, its arm
  verdict describes a pattern and is not statistically significant. Only the per-fact validity test
  can be. A paired sign test applies only to reruns.
- Reruns need a new harness entry point that starts pi from a supplied summary, plus a replay step
  that checks the rebuilt file state. Commands that are not deterministic, such as downloads,
  timestamps or long builds, may stop a compaction point from being rebuilt. That point is then
  dropped, not approximated.
- CliffCompaction writes no summary, so it appears only in reruns, and only burden scores it.
- A fact that only turns out to matter after compaction is missed by design. The rediscovery labels
  show how often that happens.
