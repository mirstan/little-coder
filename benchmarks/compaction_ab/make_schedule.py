"""Write the pre-registered live schedule (plan §5.2): rounds by task, arm order
shuffled per round with a fixed seed. Usage: python3 make_schedule.py OUT.json [arm ...]"""
import json
import random
import sys

TASKS = ["adaptive-rejection-sampler", "make-doom-for-mips", "gcode-to-text"]
ARMS = ["native", "reuse84", "blackhole", "prefix-cache"]


def schedule(arms=ARMS, seed=20261009):
    rng = random.Random(seed)
    rows = []
    for rnd, task in enumerate(TASKS, start=1):
        order = list(arms)
        rng.shuffle(order)
        rows += [{"round": rnd, "task": task, "arm": a} for a in order]
    return rows


if __name__ == "__main__":
    arms = sys.argv[2:] or ARMS
    with open(sys.argv[1], "w") as fh:
        json.dump(schedule(arms), fh, indent=1)
    print(json.dumps(schedule(arms)))
