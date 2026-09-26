#!/usr/bin/env python3
"""Query a frozen backwards-AI runtime."""

import json
import sys
from pathlib import Path

from backwards_ai import BackwardsAI, OutputContract, OutputExample

ROOT = Path(__file__).parent
RUNTIME = ROOT / "runtime.json"


def load() -> BackwardsAI:
    data = json.loads(RUNTIME.read_text())
    ai = BackwardsAI(
        OutputContract(
            name=data["name"],
            description="frozen",
            schema=data["schema"],
            examples=[],
        )
    )
    ai.feature_names = data["feature_names"]
    ai.prototypes = [(p["vec"], p["y"], p["w"]) for p in data["prototypes"]]
    ai._built = True
    return ai


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: cli.py predict '<json given>'")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd != "predict":
        print("only predict is implemented")
        sys.exit(1)
    given = json.loads(sys.argv[2])
    ai = load()
    print(json.dumps({"y": ai.predict(given)}, default=str))


if __name__ == "__main__":
    main()
