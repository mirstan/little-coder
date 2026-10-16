"""Write the pre-registered live schedule (plan §5.2): rounds by task, arm order
shuffled per round with a fixed seed.
Usage: python3 make_schedule.py OUT.json [--tasks a,b,...] [--seed N] [arm ...]"""
import json
import random
import sys

TASKS = ["adaptive-rejection-sampler", "make-doom-for-mips", "gcode-to-text"]
ARMS = ["native", "reuse84", "blackhole", "prefix-cache"]


def schedule(arms=ARMS, seed=20261009, tasks=TASKS):
    rng = random.Random(seed)
    rows = []
    for rnd, task in enumerate(tasks, start=1):
        order = list(arms)
        rng.shuffle(order)
        rows += [{"round": rnd, "task": task, "arm": a} for a in order]
    return rows


if __name__ == "__main__":
    out, args, kw = sys.argv[1], sys.argv[2:], {}
    while args and args[0] in ("--tasks", "--seed"):
        flag, val, args = args[0], args[1], args[2:]
        kw["tasks" if flag == "--tasks" else "seed"] = val.split(",") if flag == "--tasks" else int(val)
    rows = schedule(args or ARMS, **kw)
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(json.dumps(rows))
