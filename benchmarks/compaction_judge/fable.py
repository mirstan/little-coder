"""fable.py <ID> <item>...: Fable rules on items whose evidence quote was not found, with the full transcript."""
import json, subprocess, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
pid, refs = sys.argv[1], sys.argv[2:]
q = dict(json.load(open("queue.json")))
items = []
for r in refs:
    d = q[r]
    for fam in ("claude", "gpt"):
        if fam in d:
            items.append("ITEM %s (proposed by a drafter as %s)\nFact: %s\nClaimed evidence: %s" % (r, d[fam]["label"], d[fam]["fact"], d[fam]["evidence"]))
prompt = """Below is the transcript of a coding agent working on a task in a Linux container. At the end, its context was compacted (history replaced by a summary). Drafters proposed facts that a good summary must preserve so the agent can continue without re-work. The drafters' evidence quotes for the items below could not be found verbatim in the transcript, so check each fact against the transcript yourself.

For each item decide:
- vital: true as of the END of the transcript, and losing it would likely make the agent re-read, re-run, re-derive or retry something, or break a task constraint
- okay: true, but the agent could continue fine without it
- reject: false, unsupported, or superseded by later events

%s

Output ONLY JSON: {"rulings": [{"id": "<item id>", "label": "vital|okay|reject", "reason": "<one sentence>", "quote": "<a short verbatim quote from the transcript that supports or refutes it, or empty>"}]}

=== TRANSCRIPT START ===
%s
=== TRANSCRIPT END ===""" % ("\n\n".join(items), open("points/%s/pre.txt" % pid).read())
out = subprocess.run(["claude", "-p", "--model", "claude-fable-5-1"], input=prompt, capture_output=True, text=True, check=True).stdout
res = json.loads(out[out.find("{"): out.rfind("}") + 1])["rulings"]
os.makedirs("fable", exist_ok=True)
json.dump(res, open("fable/%s.json" % pid, "w"), indent=1)
for x in res: print(x["id"], x["label"], "|", x["reason"])
