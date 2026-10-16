"""validity.py: the pre-registered judge-validity test (ADR 0002).

A vital fact is judged missing when most of its recall samples (both families) lack a verified "yes".
It counts as rediscovered only when BOTH labellers say so. The judge is valid when facts judged
missing are rediscovered at odds ratio >= 2 with two-sided Fisher p < 0.05, pooled over all points.
"""
import collections, glob, json, os, re
from scipy.stats import fisher_exact

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[\\\"'`]", "", t or "")).strip()


def quote_ok(q, ctx):
    q = norm(q)
    if len(q) < 8:
        return False
    for part in re.split(r"\.\.\.|…", q):
        part = part.strip()
        if 8 <= len(part) < 24 and part in ctx:
            return True
        if any(part[i:i + 24] in ctx for i in range(0, max(1, len(part) - 23), 4)):
            return True
    return False


key = json.load(open("key.json"))
rows = []
for p in sorted(key):
    ctx = norm(open("points/%s/context.txt" % p).read())
    facts = {f["id"]: f for f in json.load(open("frozen/%s.json" % p)) if f["label"] == "vital"}
    votes = collections.defaultdict(list)
    for f in glob.glob("judgments/%s.*.recall.json" % p):
        for x in json.load(open(f))["facts"]:
            if x["id"] in facts:
                votes[x["id"]].append(x.get("present") == "yes" and quote_ok(x.get("quote"), ctx))
    lab = {fam: {x["id"]: x["rediscovered"] == "yes" for x in json.load(open("labels/%s.%s.json" % (p, fam)))}
           for fam in ("claude", "gpt")}
    for fid in facts:
        v = votes[fid]
        rows.append({"point": p, "arm": key[p]["arm"], "missing": not (sum(v) * 2 > len(v)),
                     "rediscovered": lab["claude"].get(fid, False) and lab["gpt"].get(fid, False)})

a = sum(r["missing"] and r["rediscovered"] for r in rows)
b = sum(r["missing"] and not r["rediscovered"] for r in rows)
c = sum(not r["missing"] and r["rediscovered"] for r in rows)
d = sum(not r["missing"] and not r["rediscovered"] for r in rows)
odds, p = fisher_exact([[a, b], [c, d]])
valid = odds >= 2 and p < 0.05
print("vital facts %d | missing: %d/%d rediscovered | present: %d/%d | OR %.2f, p %.3f | judge %s"
      % (len(rows), a, a + b, c, c + d, odds, p, "VALID" if valid else "NOT VALID"))
json.dump({"table": [[a, b], [c, d]], "odds_ratio": float(odds), "p": float(p), "valid": bool(valid)}, open("validity.json", "w"), indent=1)
