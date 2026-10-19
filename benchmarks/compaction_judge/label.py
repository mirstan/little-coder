"""label.py <ID> <claude|gpt>: label rediscovery per frozen fact in the post-compaction window.

The labeller sees the facts (no labels, no judge verdicts) and the first 20 actions after compaction.
Writes labels/<ID>.<family>.json.
"""
import json, subprocess, sys, os, re
os.chdir(os.path.dirname(os.path.abspath(__file__)))
pid, fam = sys.argv[1:3]
facts = json.load(open("frozen/%s.json" % pid))
post = open("points/%s/post.txt" % pid).read()
n = [0]
def num(m):
    n[0] += 1
    return "[ACTION %d] %s" % (n[0], m.group(0))
post = re.sub(r"\[tool call ", num, post)

prompt = """A coding agent's context was compacted. All the facts listed below were things the agent already knew BEFORE compaction. Below them are the agent's first actions AFTER compaction, numbered.

For each fact, decide whether any of these actions REDISCOVERED it: an action whose purpose or effect is to re-establish that specific fact (re-reading a file or output to learn it again, re-running a command to see its result again, rewriting code or a file that already existed with that content, or retrying an approach the fact says already failed). Using a fact without re-establishing it is NOT rediscovery. A first generic orientation check (listing files, printing the output file) counts only if it re-establishes this specific fact.

Output ONLY JSON: {"labels": [{"id": "<fact id>", "rediscovered": "yes|no", "action": <action number or null>, "evidence": "<short verbatim excerpt of that action's command, or empty>"}]} covering every fact.

=== FACTS ===
%s

=== ACTIONS AFTER COMPACTION ===
%s""" % ("\n".join("%s: %s" % (f["id"], f["fact"]) for f in facts), post)

os.makedirs("labels", exist_ok=True)
base = "labels/%s.%s" % (pid, fam)
if fam == "claude":
    out = subprocess.run(["claude", "-p", "--model", "claude-opus-5-5"], input=prompt, capture_output=True, text=True, check=True).stdout
else:
    subprocess.run(["codex", "exec", "--skip-git-repo-check", "-s", "read-only", "--color", "never", "-o", base + ".raw", "-"],
                   input=prompt, capture_output=True, text=True, check=True)
    out = open(base + ".raw").read()
open(base + ".raw", "w").write(out)
res = json.loads(out[out.find("{"): out.rfind("}") + 1])["labels"]
json.dump(res, open(base + ".json", "w"), indent=1)
print(base, "facts=%d labelled=%d rediscovered=%d" % (len(facts), len(res), sum(x["rediscovered"] == "yes" for x in res)))
