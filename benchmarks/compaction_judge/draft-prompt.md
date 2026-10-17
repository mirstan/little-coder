You are preparing ground truth for evaluating context compaction in a coding agent.

Below is the transcript of a coding agent working on a Terminal-Bench task inside a Linux container. At the end of this transcript the agent's context was compacted: the history was replaced by a summary, and the agent continued from that summary. You do NOT see the summary. Your job is to list the facts that a good summary would need to preserve so the agent can continue WITHOUT re-work.

List 20 to 40 facts, as of the END of the transcript. Each fact is one atomic, checkable statement. Prefer:
- the task goal and acceptance criteria (exact output paths and formats)
- hard constraints from the task instructions
- verified progress (what was built, run or confirmed, and how it was confirmed)
- files created or modified, by exact path, with their purpose
- exact identifiers the agent will need again: commands, flags, paths, versions, values
- the most recent blocking error, with its key text
- approaches already tried that failed, and why, so they are not retried
- environment facts discovered (tools present or missing, network limits, working directories)
- open hypotheses that are still unverified
- the concrete next step the agent was about to take

Rules:
- State facts as they stand at the end. Do not list superseded state as current; if something changed, state the final version.
- Do not invent anything. Every fact must be supported by the transcript; give a short verbatim evidence quote (under 200 characters).
- Label each fact "vital" if losing it would likely cause the agent to re-read, re-run, re-derive or retry something, or to break a constraint. Otherwise label it "okay".
- category is one of: goal, constraint, progress, file, identifier, error, failed_approach, environment, hypothesis, next_step.

Output ONLY a JSON object, no prose, no code fence:
{"facts": [{"fact": "...", "category": "...", "label": "vital|okay", "evidence": "..."}]}

=== TRANSCRIPT START ===
