#!/usr/bin/env python3
"""
Backwards AI
------------
Build an AI from the desired OUTPUT inward.

Convention:
  1. Write the output contract first (schema + examples + invariants).
  2. Infer the smallest function that produces those outputs.
  3. Synthesize training pairs only as needed to lock the function.
  4. Emit a frozen forward runtime that never sees the original contract.

This is inverse design, not prompt engineering.
"""

from __future__ import annotations

import json
import math
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable


@dataclass
class OutputExample:
    given: dict[str, Any]
    must_produce: Any
    weight: float = 1.0


@dataclass
class OutputContract:
    name: str
    description: str
    schema: str  # "bool" | "int" | "float" | "str" | "label" | "json"
    examples: list[OutputExample] = field(default_factory=list)
    invariants: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    def fingerprint(self) -> str:
        blob = json.dumps(
            {
                "name": self.name,
                "schema": self.schema,
                "examples": [asdict(e) for e in self.examples],
                "invariants": self.invariants,
                "labels": self.labels,
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, bool):
            out[key] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)):
            out[key] = float(v)
        elif isinstance(v, str):
            out[key] = float(sum(ord(c) for c in v) % 997) / 997.0
        elif isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, (list, tuple)):
            out[key] = float(len(v))
        else:
            out[key] = 0.0
    return out


class BackwardsAI:
    """A model derived only from an output contract."""

    def __init__(self, contract: OutputContract):
        self.contract = contract
        self.feature_names: list[str] = []
        self.prototypes: list[tuple[dict[str, float], Any, float]] = []
        self._built = False

    def build(self) -> "BackwardsAI":
        feats: set[str] = set()
        proto: list[tuple[dict[str, float], Any, float]] = []
        for ex in self.contract.examples:
            vec = _flatten(ex.given)
            feats.update(vec.keys())
            proto.append((vec, ex.must_produce, ex.weight))
        self.feature_names = sorted(feats)
        self.prototypes = proto
        self._built = True
        return self

    def _vec(self, given: dict[str, Any]) -> list[float]:
        flat = _flatten(given)
        return [flat.get(n, 0.0) for n in self.feature_names]

    def _distance(self, a: list[float], b: list[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def predict(self, given: dict[str, Any]) -> Any:
        if not self._built:
            self.build()
        if not self.prototypes:
            return self._default()
        q = self._vec(given)
        best_d = float("inf")
        best_y: Any = self._default()
        best_w = 0.0
        votes: dict[str, float] = {}
        raw: dict[str, Any] = {}
        for pvec, y, w in self.prototypes:
            p = [pvec.get(n, 0.0) for n in self.feature_names]
            d = self._distance(q, p)
            score = w / (d + 1e-9)
            key = json.dumps(y, sort_keys=True, default=str)
            votes[key] = votes.get(key, 0.0) + score
            raw[key] = y
            if d < best_d:
                best_d = d
                best_y = y
                best_w = w
        if self.contract.schema in {"label", "str", "bool", "int"}:
            winner = max(votes, key=votes.get)
            return raw[winner]
        return best_y

    def _default(self) -> Any:
        s = self.contract.schema
        if s == "bool":
            return False
        if s == "int":
            return 0
        if s == "float":
            return 0.0
        if s == "label" and self.contract.labels:
            return self.contract.labels[0]
        if s == "json":
            return {}
        return ""

    def score(self) -> float:
        if not self.contract.examples:
            return 1.0
        ok = 0.0
        tot = 0.0
        for ex in self.contract.examples:
            yhat = self.predict(ex.given)
            tot += ex.weight
            if yhat == ex.must_produce:
                ok += ex.weight
            elif self.contract.schema == "float":
                try:
                    if abs(float(yhat) - float(ex.must_produce)) < 1e-6:
                        ok += ex.weight
                except (TypeError, ValueError):
                    pass
        return ok / tot if tot else 1.0

    def emit_runtime(self) -> str:
        payload = {
            "name": self.contract.name,
            "schema": self.contract.schema,
            "feature_names": self.feature_names,
            "prototypes": [
                {"vec": v, "y": y, "w": w} for v, y, w in self.prototypes
            ],
            "fingerprint": self.contract.fingerprint(),
            "fit": self.score(),
        }
        return json.dumps(payload, indent=2, default=str)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(self.emit_runtime())
        return path


def contract_from_goal(name: str, goal: str, examples: list[dict]) -> OutputContract:
    parsed = [
        OutputExample(given=e["given"], must_produce=e["must_produce"], weight=e.get("weight", 1.0))
        for e in examples
    ]
    schema = "json"
    if parsed:
        y0 = parsed[0].must_produce
        if isinstance(y0, bool):
            schema = "bool"
        elif isinstance(y0, int) and not isinstance(y0, bool):
            schema = "int"
        elif isinstance(y0, float):
            schema = "float"
        elif isinstance(y0, str):
            schema = "label"
    return OutputContract(name=name, description=goal, schema=schema, examples=parsed)


if __name__ == "__main__":
    contract = contract_from_goal(
        name="community-loan-gate",
        goal="Approve only when risk is low and community tie is real.",
        examples=[
            {
                "given": {"income": 42000, "years_local": 8, "prior_defaults": 0, "family_tie": True},
                "must_produce": "approve",
            },
            {
                "given": {"income": 18000, "years_local": 1, "prior_defaults": 2, "family_tie": False},
                "must_produce": "deny",
            },
            {
                "given": {"income": 55000, "years_local": 12, "prior_defaults": 0, "family_tie": True},
                "must_produce": "approve",
            },
            {
                "given": {"income": 90000, "years_local": 0, "prior_defaults": 3, "family_tie": False},
                "must_produce": "deny",
            },
        ],
    )
    ai = BackwardsAI(contract).build()
    print("fingerprint", contract.fingerprint())
    print("fit", ai.score())
    print("new case", ai.predict({"income": 48000, "years_local": 6, "prior_defaults": 0, "family_tie": True}))
    out = Path(__file__).parent / "runtime.json"
    ai.save(out)
    print("wrote", out)
