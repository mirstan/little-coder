"""Compaction A/B, Stage 1 smoke test for one arm (plan §7 S1).

Drives one headless pi through PiRpc, the way the harbor adapter does, but with
pi's built-in `read` tool instead of the harbor-proxied ShellSession. It builds
about 40K tokens of context in four prompts that embed repo files, then
issues the RPC `compact`, which is the harness's manual compaction path. It
records:
  - whether the arm loaded (observer arm_env row: commands, tools);
  - the arm_registered and compaction_reuse telemetry (from the session file);
  - the compaction outcome: fromExtension, compactor, summary size;
  - wall time, and the omlx log lines (prompt, cached, TTFT) for requests
    issued during the compaction.
Usage:
  python3 benchmarks/compaction_ab/smoke.py --arm {native,reuse84,blackhole,prefix-cache} --out DIR
Uses GPU (omlx). Not part of any test suite.
"""
import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
import rpc_client as RC  # noqa: E402

OMLX_LOG = Path("/opt/homebrew/var/log/omlx.log")
MODEL = "omlx/tiel-coder-oq6e-fp16"
ARM_ENV = {
    "native": {},
    "reuse84": {"LITTLE_CODER_CACHE_REUSE_COMPACTION": "1"},
    "blackhole": {"LC_COMPACTION_ARM": "blackhole"},
    "prefix-cache": {"LC_COMPACTION_ARM": "prefix-cache"},
}
FILES = [
    "benchmarks/rpc_client.py",
    "benchmarks/run_watch.py",
    "benchmarks/turn_ledger.py",
    ".pi/extensions/shell-retention/retention.ts",
]
CHUNK_CHARS = 40_000


def omlx_lines_since(offset: int) -> list[str]:
    with OMLX_LOG.open("rb") as fh:
        fh.seek(offset)
        text = fh.read().decode("utf-8", "replace")
    keep = re.compile(r"Chat completion:|Prefix cache restore|507|ERROR")
    return [l for l in text.splitlines() if keep.search(l)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARM_ENV))
    ap.add_argument("--out", required=True)
    ap.add_argument("--compact-timeout", type=float, default=1500)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    sdir = out / "pi-session"
    sdir.mkdir(parents=True, exist_ok=True)
    env = {
        "LITTLE_CODER_AB_OBSERVER": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        **ARM_ENV[args.arm],
    }
    for k in ("LC_COMPACTION_ARM", "LITTLE_CODER_CACHE_REUSE_COMPACTION"):
        if k not in env:
            os.environ.pop(k, None)
    for k in [k for k in os.environ if k.startswith("PI_BLACKHOLE_")]:
        os.environ.pop(k)
    agent_bh = ROOT / ".cache" / "pi-bench-agent" / "pi-blackhole"
    if args.arm == "blackhole":
        agent_bh.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "vendor/compaction-ab/config/pi-blackhole-config.json", agent_bh / "pi-blackhole-config.json")

    report: dict = {"arm": args.arm, "env": env}
    rpc = RC.PiRpc(MODEL, allowed_tools=["read"], thinking="high", env=env, session_dir=str(sdir))
    try:
        for i, rel in enumerate(FILES):
            body = (ROOT / rel).read_text(errors="replace")[:CHUNK_CHARS]
            msg = (
                f"Part {i + 1} of a code-reading exercise. Do not call any tools. "
                f"Here is the start of {rel}:\n\n```\n{body}\n```\n\n"
                "Reply with one short sentence naming the most important function above."
            )
            t = time.time()
            res = rpc.prompt_and_collect(msg, timeout=900)
            report.setdefault("turns", []).append({"file": rel, "s": round(time.time() - t, 1), "stop": res.stop_reason, "text": (res.assistant_text or "")[:160]})
        log_off = OMLX_LOG.stat().st_size
        t0 = time.time()
        rid = rpc.request_compact()
        try:
            data = rpc.await_compact(rid, timeout=args.compact_timeout)
            report["compact"] = {"ok": True, "tokensBefore": data.get("tokensBefore"), "estimatedTokensAfter": data.get("estimatedTokensAfter"), "summary_chars": len(data.get("summary") or "")}
        except Exception as exc:  # noqa: BLE001
            report["compact"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        report["compact"]["wall_s"] = round(time.time() - t0, 1)
        time.sleep(2)
        report["omlx_during_compact"] = omlx_lines_since(log_off)
        # One more turn: proves the agent continues on the compacted context.
        t = time.time()
        res = rpc.prompt_and_collect("Continue: in one sentence, what were the four parts about? Do not call tools.", timeout=900)
        report["post_turn"] = {"s": round(time.time() - t, 1), "stop": res.stop_reason, "text": (res.assistant_text or "")[:300]}
        report["notifications"] = rpc.notifications()
        report["stderr_tail"] = rpc.stderr()[-3000:]
    finally:
        rpc.close(timeout=10)
    obs = sdir / "ab-observer.jsonl"
    report["observer"] = [json.loads(l) for l in obs.read_text().splitlines()] if obs.exists() else None
    tele = []
    for f in sdir.glob("*.jsonl"):
        if f.name == "ab-observer.jsonl":
            continue
        for line in f.read_text().splitlines():
            e = json.loads(line)
            if e.get("type") == "custom" and e.get("customType") == "lc-telemetry" and e.get("data", {}).get("kind") in ("arm_registered", "compaction_reuse"):
                tele.append(e["data"])
            if e.get("type") == "compaction":
                report["compaction_entry"] = {k: e.get(k) for k in ("fromHook", "tokensBefore", "firstKeptEntryId")} | {"details_compactor": (e.get("details") or {}).get("compactor"), "summary_head": (e.get("summary") or "")[:600]}
    report["telemetry"] = tele
    (out / "smoke.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report.get(k) for k in ("arm", "compact", "compaction_entry", "telemetry")}, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
