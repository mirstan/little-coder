"""judge.py <ID> <claude|gpt> <sample> <recall|precision>: one judge call, written to judgments/.

recall:    context after compaction (summary + kept messages) vs the frozen facts, plus checklist items.
precision: the summary's own claims vs the pre-compaction transcript.
Labels, arms and the drafters are never shown to the judge.
"""
import json, subprocess, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
pid, fam, sample, kind = sys.argv[1:5]

CHECKLIST = {
    "R1": "The task goal and acceptance criteria are stated, including the exact deliverable path and format.",
    "R2": "Hard constraints from the task instructions are preserved.",
    "R3": "Current progress is stated as verified facts, including how each was verified.",
    "R4": "Every file created or modified is listed by exact path, with its purpose.",
    "R5": "Exact identifiers are reproduced verbatim where needed: paths, commands, flags, versions, values.",
    "R6": "The most recent blocking error or failing check is quoted with its key text.",
    "R7": "Approaches already tried that failed are listed with why each failed.",
    "R8": "Discovered environment facts are preserved (tools present or missing, limits, working directories).",
    "R9": "Open hypotheses or unverified assumptions are marked as unverified.",
    "R10": "A concrete, executable next step is stated.",
    "A1": "It is self-contained: no references to content that is no longer available (e.g. 'as shown above', 'the earlier output').",
    "A2": "Key file contents or locations are given in enough detail that files already read need not be re-read.",
}

if kind == "recall":
    facts = json.load(open("frozen/%s.json" % pid))
    prompt = """A coding agent's context was compacted. Below is EVERYTHING the agent had in context right after compaction: a summary of its earlier work, followed by its most recent messages kept verbatim. You do not see the earlier history.

Part 1. For each fact, answer "yes" if this context states it, or states something that makes it unambiguous to the agent; otherwise "no". A fact that is only partly stated, or stated in a way that contradicts it, is "no". For every "yes", give a verbatim quote from the context (under 200 characters) that shows it. Judge each fact independently; length of the context is not evidence.

Part 2. Answer each checklist item about this context "yes" or "no", with a verbatim quote for "yes".

FACTS:
%s

CHECKLIST:
%s

Output ONLY JSON: {"facts": [{"id": "<fact id>", "present": "yes|no", "quote": "..."}], "checklist": [{"id": "R1", "answer": "yes|no", "quote": "..."}]}

=== CONTEXT AFTER COMPACTION START ===
%s
=== CONTEXT AFTER COMPACTION END ===""" % ("\n".join("%s: %s" % (f["id"], f["fact"]) for f in facts),
        "\n".join("%s: %s" % kv for kv in CHECKLIST.items()), open("points/%s/context.txt" % pid).read())
else:
    summary = open("points/%s/context.txt" % pid).read().split("\n\n## Kept messages\n\n")[0]
    prompt = """Below is a summary that replaced a coding agent's history, followed by that history (the transcript). Check the summary's accuracy against the transcript, as of the END of the transcript.

1. Split the summary into atomic factual claims (one checkable statement each; skip pure headings, and treat a block of copied task instructions as one claim). For each claim give a verdict: "supported" (the transcript supports it), "contradicted" (the transcript contradicts it), or "unsupported" (the transcript neither supports nor contradicts it). Give a short verbatim transcript quote for supported or contradicted.
2. unverified_success: list any summary claims that something works, passes or is complete when the transcript does not verify it.
3. superseded: list any summary claims that present earlier state as current when the transcript shows it changed later.

Output ONLY JSON: {"claims": [{"claim": "...", "verdict": "supported|contradicted|unsupported", "quote": "..."}], "unverified_success": ["..."], "superseded": ["..."]}

=== SUMMARY START ===
%s
=== SUMMARY END ===

=== TRANSCRIPT START ===
%s
=== TRANSCRIPT END ===""" % (summary, open("points/%s/pre.txt" % pid).read())

os.makedirs("judgments", exist_ok=True)
base = "judgments/%s.%s.%s.%s" % (pid, fam, sample, kind)
if fam == "claude":
    out = subprocess.run(["claude", "-p", "--model", "claude-opus-5-5"], input=prompt, capture_output=True, text=True, check=True).stdout
else:
    subprocess.run(["codex", "exec", "--skip-git-repo-check", "-s", "read-only", "--color", "never", "-o", base + ".raw", "-"],
                   input=prompt, capture_output=True, text=True, check=True)
    out = open(base + ".raw").read()
open(base + ".raw", "w").write(out)
res = json.loads(out[out.find("{"): out.rfind("}") + 1])
json.dump(res, open(base + ".json", "w"), indent=1)
print(base, "ok")
