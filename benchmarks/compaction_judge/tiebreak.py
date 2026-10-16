"""tiebreak.py <ID>: blind vital/okay/reject labels from local Qwen 3.8 27B for one point.

Labels every queue item (disputed) and every agreed fact (calibration), shuffled together,
without showing the drafters' labels. Writes qwen/<ID>.json.
"""
import json, random, sys, time, urllib.request, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
pid = sys.argv[1]
pre = open("points/%s/pre.txt" % pid).read()
queue = [(i, d) for i, d in json.load(open("queue.json")) if d["point"] == pid]
agreed = json.load(open("agreed.json"))[pid]

import re
def norm(t):  # JSON escaping and whitespace differ between quote and transcript
    return re.sub(r"\s+", " ", re.sub(r"[\\\"'`]", "", t))
npre = norm(pre)

def locate(ev):
    for part in re.split(r"\.\.\.|…", ev):  # drafters stitch quotes with ellipses
        e = norm(part).strip()
        for off in range(0, max(1, len(e) - 20), 10):
            k = e[off: off + 24]
            if len(k) >= 16:
                at = npre.find(k)
                if at >= 0:
                    return npre[max(0, at - 600): at + 600]
    return None

facts = []
for i, d in queue:
    f = d.get("claude") or d.get("gpt")
    facts.append({"ref": i, "set": "queue", "fact": f["fact"], "evidence": f["evidence"]})
for n, x in enumerate(agreed):
    facts.append({"ref": "%s-A%02d" % (pid, n), "set": "agreed", "fact": x["claude"]["fact"], "evidence": x["claude"]["evidence"],
                  "consensus": x["claude"]["label"]})
random.Random(pid).shuffle(facts)
blocks = []
for k, f in enumerate(facts, 1):
    f["k"] = k
    ctx = locate(f["evidence"])
    f["located"] = ctx is not None
    blocks.append("FACT %d: %s\nEvidence quote: %s\nTranscript around the quote: %s" % (
        k, f["fact"], f["evidence"], ctx if ctx else "[quote not found in transcript]"))

prompt = """A coding agent worked on a task in a Linux container. Its context was then compacted (history replaced by a summary). Below are candidate facts that a good summary should preserve so the agent can continue without re-work. For each, decide:
- vital: true as of the end of the transcript, and losing it would likely make the agent re-read, re-run, re-derive or retry something, or break a task constraint
- okay: true, but the agent could continue fine without it
- reject: false, not supported by the transcript, or superseded by later events

=== END OF TRANSCRIPT (final state) ===
%s

=== CANDIDATE FACTS ===
%s

Output ONLY JSON: {"labels": [{"k": <fact number>, "label": "vital|okay|reject"}, ...]} covering every fact.""" % (pre[-8000:], "\n\n".join(blocks))

body = json.dumps({"model": "dirk-q38-27b-oq4e", "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 6000, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
t = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:8000/v1/chat/completions", body,
                                     {"Content-Type": "application/json"}), timeout=3600))
out = r["choices"][0]["message"]["content"]
labels = {x["k"]: x["label"] for x in json.loads(out[out.find("{"): out.rfind("}") + 1])["labels"]}
for f in facts:
    f["qwen"] = labels.get(f["k"])
os.makedirs("qwen", exist_ok=True)
json.dump({"usage": r.get("usage"), "seconds": round(time.time() - t), "facts": facts}, open("qwen/%s.json" % pid, "w"), indent=1)
print(pid, "prompt=%s" % r["usage"]["prompt_tokens"], "%ds" % (time.time() - t), "missing=%d" % sum(1 for f in facts if not f["qwen"]))
