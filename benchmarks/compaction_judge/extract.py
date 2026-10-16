"""Render each usable r3 compaction point to blinded text files.

points/<ID>/pre.txt   transcript the summariser replaced (drafters see only this)
points/<ID>/context.txt what the agent had after compaction: summary + kept messages
points/<ID>/post.txt  first 20 actions after compaction (hidden until labelling)
key.json              ID -> trial, arm (never shown to judges)
"""
import json, glob, os, random, sys

LEDGER = os.path.expanduser("~/tb21-runs/compaction-ab-r3/ledger.jsonl")
EXCLUDE = {"gcode-to-text__MGHczpn"}  # ADR 0001 item 4: transcript not fully saved
POST_ACTIONS = 20

def text_of(content):
    if isinstance(content, str):
        return content
    out = []
    for b in content or []:
        t = b.get("type")
        if t == "text":
            out.append(b["text"])
        elif t == "thinking":
            pass  # reasoning, not state; halves the transcript
        elif t == "toolCall":
            out.append("[tool call %s] %s" % (b.get("name"), json.dumps(b.get("arguments"), ensure_ascii=False)))
        elif t == "image":
            out.append("[image]")
    return "\n".join(out)

def render(entries, cap=True):
    out = []
    for e in entries:
        if e.get("type") != "message":
            continue
        m = e["message"]
        role = m.get("role")
        body = text_of(m.get("content"))
        if role == "toolResult":
            if cap and len(body) > 6000:  # long dumps carry few facts; keep head and the error-bearing tail
                body = body[:4000] + "\n[... %d chars cut ...]\n" % (len(body) - 6000) + body[-2000:]
            out.append("### tool result (%s)%s\n%s" % (m.get("toolName"), " ERROR" if m.get("isError") else "", body))
        else:
            out.append("### %s\n%s" % (role, body))
    return "\n\n".join(out)

trials = [json.loads(l) for l in open(LEDGER)]
trials = [t for t in trials if os.path.basename(t["trial"]) not in EXCLUDE]
rng = random.Random(20261015)
ids = ["P%d" % i for i in range(1, len(trials) + 1)]
rng.shuffle(ids)
key = {}
for pid, t in zip(ids, trials):
    sess = [f for f in glob.glob(t["trial"] + "/agent/pi-session/*.jsonl") if "observer" not in f][0]
    L = [json.loads(l) for l in open(sess)]
    ci = next(i for i, x in enumerate(L) if x.get("type") == "compaction")
    post, n = [], 0
    for e in L[ci + 1:]:
        if e.get("type") == "message" and e["message"].get("role") == "assistant":
            n += sum(1 for b in e["message"].get("content", []) if b.get("type") == "toolCall")
        post.append(e)
        if n >= POST_ACTIONS:
            break
    d = "points/" + pid
    os.makedirs(d, exist_ok=True)
    open(d + "/pre.txt", "w").write(render(L[:ci]))
    ki = next(i for i, x in enumerate(L) if x.get("id") == L[ci]["firstKeptEntryId"])
    open(d + "/context.txt", "w").write("## Summary\n\n" + L[ci]["summary"] + "\n\n## Kept messages\n\n" + render(L[ki:ci], cap=False))
    open(d + "/post.txt", "w").write(render(post))
    key[pid] = {"trial": os.path.basename(t["trial"]), "arm": t["arm"], "task": os.path.basename(t["trial"]).split("__")[0]}
json.dump(key, open("key.json", "w"), indent=1)
for pid in sorted(key):
    s = os.path.getsize("points/%s/pre.txt" % pid)
    print(pid, key[pid]["task"], "pre_chars=%d (~%dK tok)" % (s, s / 3.5 / 1000), "context=%d" % os.path.getsize("points/%s/context.txt" % pid))
