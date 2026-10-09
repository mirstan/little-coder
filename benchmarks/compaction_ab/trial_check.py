"""Post-trial checks and ledger row for one compaction A/B trial (plan §5.3-§5.5).

Usage: python3 trial_check.py <trial_dir> <arm> <frozen_sha> <omlx_log_offset> <ledger.jsonl>
Exit 0 when the trial counts, 1 when it is void (the reasons are in the row).
"""
import json
import re
import sys
from pathlib import Path

OMLX_LOG = Path("/opt/homebrew/var/log/omlx.log")
EXPECTED_BUDGET = 12150.0
ARM_ENV = {
    "native": {"LC_COMPACTION_ARM": None, "LITTLE_CODER_CACHE_REUSE_COMPACTION": None},
    "reuse84": {"LC_COMPACTION_ARM": None, "LITTLE_CODER_CACHE_REUSE_COMPACTION": "1"},
    "blackhole": {"LC_COMPACTION_ARM": "blackhole", "LITTLE_CODER_CACHE_REUSE_COMPACTION": None},
    "prefix-cache": {"LC_COMPACTION_ARM": "prefix-cache", "LITTLE_CODER_CACHE_REUSE_COMPACTION": None},
}
ARM_COMMANDS = {"blackhole": {"blackhole", "blackhole-memory", "blackhole-recall", "blackhole-export"}, "prefix-cache": {"prefix-compaction"}}


def jsonl(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def check(trial: Path, arm: str, sha: str, log_offset: int) -> dict:
    agent = trial / "agent"
    sdir = agent / "pi-session"
    void = []
    row = {"trial": str(trial), "arm": arm, "frozen_sha": sha}
    snap = json.loads((agent / "environment_snapshot.json").read_text()) if (agent / "environment_snapshot.json").exists() else {}
    budget = (snap.get("timeout") or {}).get("effective_timeout_sec") if isinstance(snap.get("timeout"), dict) else None
    if budget is None:
        m = re.search(r'"effective_timeout_sec":\s*([0-9.]+)', json.dumps(snap))
        budget = float(m.group(1)) if m else None
    row["effective_timeout_sec"] = budget
    if budget != EXPECTED_BUDGET:
        void.append(f"budget {budget} != {EXPECTED_BUDGET}")
    obs = jsonl(sdir / "ab-observer.jsonl")
    env_rows = [r for r in obs if r.get("kind") == "arm_env"]
    if not env_rows:
        void.append("no arm_env row")
    else:
        env = env_rows[0]["env"]
        row["arm_env"] = env
        for k, v in ARM_ENV[arm].items():
            if env.get(k) != v:
                void.append(f"env {k}={env.get(k)!r}, expected {v!r}")
        if env.get("LITTLE_CODER_PI_SESSION_DIR_IN_LOGS") != "1":
            void.append("session saving off")
        if env.get("PI_BLACKHOLE_vars"):
            void.append(f"PI_BLACKHOLE_* set: {env['PI_BLACKHOLE_vars']}")
        cmds = set(env_rows[0].get("commands") or [])
        need = ARM_COMMANDS.get(arm, set())
        if need - cmds:
            void.append(f"arm commands missing: {sorted(need - cmds)}")
        for other, names in ARM_COMMANDS.items():
            if other != arm and names & cmds:
                void.append(f"foreign arm loaded: {other}")
    tele = []
    for f in sdir.glob("*.jsonl"):
        if f.name == "ab-observer.jsonl":
            continue
        for e in jsonl(f):
            if e.get("type") == "custom" and e.get("customType") == "lc-telemetry":
                tele.append(e.get("data") or {})
    registered = {t.get("arm") for t in tele if t.get("kind") == "arm_registered"}
    reuse_rows = [t for t in tele if t.get("kind") == "compaction_reuse"]
    want_reg = {"blackhole": {"blackhole"}, "prefix-cache": {"prefix-cache"}}.get(arm, set())
    if registered != want_reg:
        void.append(f"arm_registered {sorted(registered)} != {sorted(want_reg)}")
    comps = [r for r in obs if r.get("kind") == "compact"]
    before = [r for r in obs if r.get("kind") == "before_compact"]
    row["compactions"] = [
        {**b, **{k: c.get(k) for k in ("from_extension", "compactor", "summary_chars")}} for b, c in zip(before, comps)
    ] + [{**b, "unfinished": True} for b in before[len(comps):]]
    if arm == "reuse84" and len(reuse_rows) != len(before):
        void.append(f"compaction_reuse rows {len(reuse_rows)} != compactions {len(before)}")
    row["compaction_reuse"] = reuse_rows
    req = [r for r in obs if r.get("kind") == "request"]
    row["drift"] = {
        "requests": len(req),
        "system_sha1s": len({r.get("system_sha1") for r in req}),
        "tools_sha1s": len({r.get("tools_sha1") for r in req}),
    }
    log = (agent / "little_coder.log").read_text(errors="replace") if (agent / "little_coder.log").exists() else ""
    m = re.search(r"=== stop_reason: (\w+) ===", log)
    row["stop_reason"] = m.group(1) if m else None
    row["cut_at_deadline_candidate"] = "deliberate compaction failed" in log
    try:
        res = json.loads((trial / "result.json").read_text())
        vr = res.get("verifier_result") or {}
        row["reward"] = (vr.get("rewards") or {}).get("reward")
        row["started_at"], row["finished_at"] = res.get("started_at"), res.get("finished_at")
    except (OSError, ValueError):
        void.append("no result.json")
    if log_offset >= 0 and OMLX_LOG.exists():
        with OMLX_LOG.open("rb") as fh:
            fh.seek(log_offset)
            (agent / "omlx-slice.log").write_bytes(fh.read())
    row["void"] = void
    return row


if __name__ == "__main__":
    trial, arm, sha, off, ledger = sys.argv[1:6]
    r = check(Path(trial), arm, sha, int(off))
    with open(ledger, "a") as fh:
        fh.write(json.dumps(r) + "\n")
    print(json.dumps({k: r.get(k) for k in ("arm", "reward", "stop_reason", "void", "compactions")})[:2000])
    sys.exit(1 if r["void"] else 0)
