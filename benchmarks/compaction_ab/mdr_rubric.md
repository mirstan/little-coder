# Compaction A/B quality rubric (plan §4.4 Q1 and Q2; the same rubric as Stage 4 / O2)

The judge is Claude, run as a fresh-context subagent for each point. The judge never sees `key.json`.

## Q1: blind fact checklist

**Stage 1 (gold).** The judge reads only `transcript.md` and writes 10–15 atomic facts that a continuing agent needs, in these categories:
1. goal and acceptance criteria;
2. hard constraints;
3. files created or modified (paths);
4. current state of the work;
5. current failure (exact error or failing test);
6. approaches tried and rejected;
7. key values (paths, versions, commands, ports);
8. the next step.

It must not open `summaries.md` until the gold sheet is written to `gold.md`.

**Stage 2 (score).** A new judge gets `gold.md` and `summaries.md` (S1..Sn, blinded). For each S it marks each fact **correct**, **wrong** or **missing**, and counts claims that contradict the transcript.

Score = (correct − wrong − 0.5 × contradictions) / facts.

Output: `q1-scores.json`, in the form `{S1: {correct, wrong, missing, contradictions, score}, ...}`.

**Calibration:** a second judge re-scores 25% of the points, and the agreement is reported.

## Q2: next-action probe

The judge gets `q2.md`: the agent's real next 5 actions (reference), then 3 sampled next actions per blinded S. Each sample gets exactly one label:
- **progresses**: consistent with the state and moves toward the goal; it need not match the reference;
- **redundant**: redoes work already done;
- **regresses**: contradicts the state, or undoes or breaks finished work;
- **off-task**.

A sample with finish=length on thinking is labelled **budget-breach** and not scored. A call to a tool outside the allowlist (e.g. `recall`) is labelled **off-task**, and the judge notes it.

Metric per S: fraction of samples labelled progresses.

Output: `q2-labels.json`.

## Decision use (plan §6 G4)

Paired by point, against A0:
- mean Q1 difference ≥ −0.10, and the lower bound of the 90% paired bootstrap CI ≥ −0.20;
- Q2 progresses rate ≥ A0's − 0.15.

With 4 nested points from one trajectory, only gross inferiority is detectable. Points from the same trial are not independent.
