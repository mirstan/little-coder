"""origin.py <ID>: Fable decides, per frozen-draft fact, whether the agent itself established it."""
import json, subprocess, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
pid = sys.argv[1]
facts = json.load(open("frozen/%s.json" % pid))
listing = "\n".join("%s: %s" % (f["id"], f["fact"]) for f in facts)
prompt = """Below is the transcript of a coding agent, followed by facts that drafters extracted from it. A summary of this transcript can only be expected to preserve what the AGENT itself knew. For each fact decide its origin:
- agent: the agent observed it (a command output it saw), stated it, acted on it, or it is in the task instructions the agent received
- drafter: the drafter's own inference or analysis that the agent never reached (e.g. noticing a bug the agent did not notice, or a conclusion the agent never drew)
- false: contradicted by or absent from the transcript

Output ONLY JSON: {"origins": [{"id": "<fact id>", "origin": "agent|drafter|false", "reason": "<one short sentence, only for drafter or false>"}]} covering every fact.

=== FACTS ===
%s

=== TRANSCRIPT START ===
%s
=== TRANSCRIPT END ===""" % (listing, open("points/%s/pre.txt" % pid).read())
out = subprocess.run(["claude", "-p", "--model", "claude-fable-5-1"], input=prompt, capture_output=True, text=True, check=True).stdout
res = json.loads(out[out.find("{"): out.rfind("}") + 1])["origins"]
os.makedirs("origin", exist_ok=True)
json.dump(res, open("origin/%s.json" % pid, "w"), indent=1)
bad = [x for x in res if x["origin"] != "agent"]
print(pid, "facts=%d returned=%d not_agent=%d" % (len(facts), len(res), len(bad)))
for x in bad: print("  ", x["id"], x["origin"], "|", x.get("reason", ""))
