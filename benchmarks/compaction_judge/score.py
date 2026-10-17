"""score.py: aggregate step-3 judgments into per-point and per-arm results (results.md, results.json)."""
import json, glob, os, re, collections, statistics as st
os.chdir(os.path.dirname(os.path.abspath(__file__)))
key = json.load(open("key.json"))
PTS = sorted(key)

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
        # judges trim words from long quotes; any 24-char stretch must still be verbatim
        if any(part[i:i + 24] in ctx for i in range(0, max(1, len(part) - 23), 4)):
            return True
    return False

def kappa(a, b):
    n = len(a)
    if not n: return None
    po = sum(x == y for x, y in zip(a, b)) / n
    cats = set(a) | set(b)
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return None if pe == 1 else (po - pe) / (1 - pe)

rows, fact_votes, chk_votes = {}, collections.defaultdict(dict), collections.defaultdict(dict)
for p in PTS:
    ctx = norm(open("points/%s/context.txt" % p).read())
    facts = {f["id"]: f for f in json.load(open("frozen/%s.json" % p))}
    r = {"task": key[p]["task"], "arm": key[p]["arm"], "fam": {}}
    for fam in ("claude", "gpt"):
        vr, orr, qv, chk, prec, contra, unver, sup = [], [], [], [], [], [], [], []
        per_fact = collections.defaultdict(list); per_chk = collections.defaultdict(list)
        for f in sorted(glob.glob("judgments/%s.%s.*.recall.json" % (p, fam))):
            d = json.load(open(f))
            if "checklist" not in d:  # some answers put the checklist items inside "facts"
                d["checklist"] = [{"id": x["id"], "answer": x.get("answer", x.get("present"))} for x in d["facts"] if re.fullmatch(r"[RA]\d+", x["id"])]
            got = {x["id"]: x for x in d["facts"]}
            yes = {}
            for fid in facts:
                x = got.get(fid, {"present": "no"})
                ok = x["present"] == "yes" and quote_ok(x.get("quote"), ctx)
                if x["present"] == "yes": qv.append(ok)
                yes[fid] = ok; per_fact[fid].append(ok)
            v = [fid for fid in facts if facts[fid]["label"] == "vital"]; o = [fid for fid in facts if facts[fid]["label"] == "okay"]
            vr.append(sum(yes[i] for i in v) / len(v)); orr.append(sum(yes[i] for i in o) / len(o) if o else None)
            c = {x["id"]: x["answer"] == "yes" for x in d["checklist"]}
            chk.append(sum(c.values()) / len(c))
            for k2, val in c.items(): per_chk[k2].append(val)
        for f in sorted(glob.glob("judgments/%s.%s.*.precision.json" % (p, fam))):
            d = json.load(open(f)); cl = d["claims"]
            prec.append(sum(x["verdict"] == "supported" for x in cl) / len(cl))
            contra.append(sum(x["verdict"] == "contradicted" for x in cl))
            unver.append(len(d.get("unverified_success", []))); sup.append(len(d.get("superseded", [])))
        mean = lambda L: round(st.mean([x for x in L if x is not None]), 3) if [x for x in L if x is not None] else None
        r["fam"][fam] = {"n_recall": len(vr), "vital_recall": mean(vr), "vital_recall_samples": [round(x, 3) for x in vr],
                         "okay_recall": mean(orr), "quote_verified": mean(qv), "checklist_yes": mean(chk),
                         "n_precision": len(prec), "precision": mean(prec), "contradicted": mean(contra),
                         "unverified_success": mean(unver), "superseded": mean(sup)}
        for fid, L in per_fact.items(): fact_votes[(p, fid)][fam] = sum(L) * 2 > len(L)
        for k2, L in per_chk.items(): chk_votes[(p, k2)][fam] = sum(L) * 2 > len(L)
    r["context_chars"] = len(open("points/%s/context.txt" % p).read())
    r["vital_recall_per_1k"] = None
    vals = [r["fam"][f]["vital_recall"] for f in r["fam"] if r["fam"][f]["vital_recall"] is not None]
    r["vital_recall"] = round(st.mean(vals), 3) if vals else None
    pv = [r["fam"][f]["precision"] for f in r["fam"] if r["fam"][f]["precision"] is not None]
    r["precision"] = round(st.mean(pv), 3) if pv else None
    rows[p] = r

both = [v for v in fact_votes.values() if len(v) == 2]
fk = kappa([v["claude"] for v in both], [v["gpt"] for v in both])
items = collections.defaultdict(lambda: ([], []))
for (p, k2), v in chk_votes.items():
    if len(v) == 2: items[k2][0].append(v["claude"]); items[k2][1].append(v["gpt"])
chk_agree = {k2: {"agree": sum(a == b for a, b in zip(*v)), "n": len(v[0])} for k2, v in items.items()}

out = ["# Step 3 results (blinded judging of 7 compaction points)", "",
       "Fact-level agreement between judge families (majority of samples): %d facts, raw %.2f, Cohen kappa %s" % (
           len(both), sum(v["claude"] == v["gpt"] for v in both) / max(1, len(both)), "%.2f" % fk if fk is not None else "n/a"), "",
       "| point | task | arm | context chars | vital recall (Claude / GPT) | okay recall | quote verified | checklist yes | precision (C / G) | contradicted | unverified success | superseded |",
       "|---|---|---|---:|---|---|---|---|---|---|---|---|"]
for p in PTS:
    r = rows[p]; c, g = r["fam"]["claude"], r["fam"]["gpt"]
    f2 = lambda a, b: "%s / %s" % (a, b)
    out.append("| %s | %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        p, r["task"].split("-")[0], r["arm"], r["context_chars"], f2(c["vital_recall"], g["vital_recall"]),
        f2(c["okay_recall"], g["okay_recall"]), f2(c["quote_verified"], g["quote_verified"]), f2(c["checklist_yes"], g["checklist_yes"]),
        f2(c["precision"], g["precision"]), f2(c["contradicted"], g["contradicted"]), f2(c["unverified_success"], g["unverified_success"]),
        f2(c["superseded"], g["superseded"])))
out += ["", "## Arm ranges per task (vital recall, mean of both families)", ""]
gate = {}
for task in sorted({r["task"] for r in rows.values()}):
    for arm in ("prefix-cache", "blackhole"):
        v = [rows[p]["vital_recall"] for p in PTS if rows[p]["task"] == task and rows[p]["arm"] == arm]
        pr = [rows[p]["precision"] for p in PTS if rows[p]["task"] == task and rows[p]["arm"] == arm]
        gate[(task, arm)] = (v, pr)
        out.append("- %s / %s: vital recall %s, precision %s" % (task, arm, v, pr))
out += ["", "## Checklist agreement between families", ""] + ["- %s: %d/%d" % (k2, v["agree"], v["n"]) for k2, v in sorted(chk_agree.items())]
open("results.md", "w").write("\n".join(out) + "\n")
json.dump({"rows": rows, "fact_kappa": fk, "checklist_agreement": chk_agree}, open("results.json", "w"), indent=1)
print("\n".join(out))
