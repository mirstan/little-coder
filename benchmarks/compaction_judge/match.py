"""match.py <ID>: align the two families' drafted facts for one compaction point.

Writes matched/<ID>.json: {"agreed": [...], "label_conflict": [...], "only_claude": [...], "only_gpt": [...]}.
Alignment is by meaning, done by one Claude call; the merge into buckets is mechanical.
"""
import json, subprocess, sys, os

def load(path):
    t = open(path).read()
    return json.loads(t[t.find("{"): t.rfind("}") + 1])["facts"]

pid = sys.argv[1]
os.chdir(os.path.dirname(os.path.abspath(__file__)))
A, B = load("drafts/%s.claude.raw" % pid), load("drafts/%s.gpt.raw" % pid)
fmt = lambda L, p: "\n".join("%s%d: %s" % (p, i, f["fact"]) for i, f in enumerate(L))
prompt = """Two lists of facts were extracted independently from the same transcript. Pair facts that state the same thing (same claim, possibly different wording or granularity). A fact may pair with at most one fact from the other list; if one fact covers two from the other list, pair it with the closer one. Do not pair facts that only share a topic.

List A:
%s

List B:
%s

Output ONLY JSON, no prose: {"pairs": [[a_index, b_index], ...]}""" % (fmt(A, "A"), fmt(B, "B"))
out = subprocess.run(["claude", "-p", "--model", "claude-opus-5-5"], input=prompt, capture_output=True, text=True, check=True).stdout
pairs = json.loads(out[out.find("{"): out.rfind("}") + 1])["pairs"]
pa, pb = {a for a, b in pairs}, {b for a, b in pairs}
res = {"agreed": [], "label_conflict": [], "only_claude": [], "only_gpt": []}
for a, b in pairs:
    item = {"claude": A[a], "gpt": B[b]}
    res["agreed" if A[a]["label"] == B[b]["label"] else "label_conflict"].append(item)
res["only_claude"] = [{"claude": f} for i, f in enumerate(A) if i not in pa]
res["only_gpt"] = [{"gpt": f} for i, f in enumerate(B) if i not in pb]
os.makedirs("matched", exist_ok=True)
json.dump(res, open("matched/%s.json" % pid, "w"), indent=1)
print(pid, {k: len(v) for k, v in res.items()})
