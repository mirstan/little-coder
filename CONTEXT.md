# Glossary

## Compaction point
The moment in a trial when the harness replaces the session history with a summary. It has a
*pre-compaction transcript* (everything the summary replaced) and a *post-compaction window*.
Not the same as the compaction *request*, which can precede the point by a long stall.

## Arm
One compaction strategy under comparison, e.g. `prefix-cache` or `blackhole`. A trial runs exactly
one arm. Each summary is attributed to the arm that wrote it, and judges never see that attribution.

## Vital fact
A fact from the pre-compaction transcript that the agent needs to continue without re-work:
a path, a command, an error, a failed approach, a verified result. Claude and GPT each draft and
label facts independently, from the pre-compaction transcript only (never the post-compaction
window). A fact both propose with the same vital/okay label is kept. A fact only one proposes, or one
they label differently, goes to a human. The list is frozen before any summary is scored. Facts kept but not needed are *okay
facts*, which are scored separately and never count toward recall.

## Post-compaction window
The first 20 agent actions after a compaction point, or fewer if the trial ends first. Rediscovery is only counted inside it.

## Rediscovery
An action in the post-compaction window that re-establishes a vital fact the agent already had
before compaction: re-reading, re-running, rewriting or retrying it. Labelled per vital fact by an
LLM that must cite the action. A first orientation check (`ls`, reading the output file) is not
rediscovery unless it re-establishes a specific vital fact.
Distinct from an *exact repeat* (a command whose normalised text matches one run before
compaction), which is a mechanical baseline only.

## Vital recall
The share of a compaction point's vital facts that a summary preserves, as judged yes/no per fact
with a quoted span. The primary summary-quality number. Not the same as *precision*, the share of a
summary's own atomic claims that the pre-compaction transcript supports.

## Judge validity
Whether the judge's "missing" verdicts predict rediscovery. Tested per vital fact across all
compaction points: valid when facts judged missing are rediscovered at an odds ratio of at least 2
with Fisher p < 0.05.

## Paired rerun
A continuation from a rebuilt compaction point, run once per condition from the same container
state: full history, each arm's summary, and CliffCompaction's truncated history. Each condition gets
20 actions × 3 seeds. Its outcome is *burden*, the count of rediscoveries and errored actions, taken
relative to the full-history condition.
