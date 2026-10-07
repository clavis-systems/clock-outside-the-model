"""A clock outside the model, evaluated on TicToc (Findings of ACL 2026, arXiv:2510.23853).

Rule: call the tool again if more than THRESHOLD_S seconds have passed since its most recent result in the
conversation, or if there is no earlier result. Otherwise answer from context. The rule reads only message timestamps.

Metric: the authors' normalized alignment rate (get_metric.py), re-implemented here:
  - skip cases whose preference contains "any" or whose score is strictly between 0.5 and 2.5;
  - label "direct" if the score is <= 0.5, "tool" if >= 2.5;
  - NAR = 0.5 * (TP/(TP+FN) + TN/(TN+FP)), where TP = calling the tool when people prefer it.

Usage (data from https://huggingface.co/datasets/yizecheng/TicToc, Apache 2.0):
    python rule.py data/test.parquet
    python rule.py data/train.parquet --threshold 1800
Optional --exclude-boundaries also skips scores of exactly 0.5 and 2.5.
Reading parquet needs pandas + pyarrow; JSON files (the authors' GitHub splits) need nothing.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime

THRESHOLD_S = 1800


def when(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def label(case: dict, exclude_boundaries: bool = False) -> str | None:
    pref, score = case["preference"], float(case["pref_score"])
    if "any" in pref or 0.5 < score < 2.5 or (exclude_boundaries and score in (0.5, 2.5)):
        return None
    if "direct" in pref and score <= 0.5:
        return "direct"
    if "tool" in pref and score >= 2.5:
        return "tool"
    raise ValueError(f"unexpected preference in {case['id']}")


def decide(history: list[dict], threshold_s: float = THRESHOLD_S) -> str:
    last = max((i for i, m in enumerate(history) if m["role"] == "tool"), default=None)
    if last is None:
        return "tool"
    age = (when(history[-1]["time"]) - when(history[last]["time"])).total_seconds()
    return "tool" if age > threshold_s else "direct"


def nar(pairs: list[tuple[str, str]]) -> tuple[float | None, dict]:
    tp = sum(g == "tool" and d == "tool" for g, d in pairs)
    fn = sum(g == "tool" and d == "direct" for g, d in pairs)
    tn = sum(g == "direct" and d == "direct" for g, d in pairs)
    fp = sum(g == "direct" and d == "tool" for g, d in pairs)
    value = 0.5 * (tp / (tp + fn) + tn / (tn + fp)) if tp + fn and tn + fp else None
    return value, {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def load(path: str) -> list[dict]:
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq                       # plain Python lists and dicts, no numpy arrays
        rows = pq.read_table(path).to_pylist()
    else:
        rows = json.loads(open(path, encoding="utf-8").read())
    return [{**r, "scenario": r.get("__file__")} for r in rows]     # same grouping for parquet and JSON


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_S)
    parser.add_argument("--exclude-boundaries", action="store_true")
    args = parser.parse_args()
    cases = load(args.data)
    scored = [(c, label(c, args.exclude_boundaries)) for c in cases]
    scored = [(c, g) for c, g in scored if g is not None]
    pairs = [(g, decide(c["history"], args.threshold)) for c, g in scored]
    value, counts = nar(pairs)
    rng = random.Random(1)
    by_case = sorted(nar([rng.choice(pairs) for _ in pairs])[0] for _ in range(2000))
    groups: dict = {}
    for (c, g), p in zip(scored, pairs):
        groups.setdefault(c.get("scenario") or c["id"].rsplit("_", 1)[0], []).append(p)
    keys = sorted(groups)
    by_scenario = []
    for _ in range(2000):
        sample = [p for k in (rng.choice(keys) for _ in keys) for p in groups[k]]
        by_scenario.append(nar(sample)[0])
    by_scenario.sort()
    print(json.dumps({"cases": len(cases), "evaluable": len(pairs), "threshold_s": args.threshold,
                      "normalized_alignment": round(value, 4), **counts,
                      "ci95_by_case": [round(by_case[50], 4), round(by_case[1949], 4)],
                      "ci95_by_group": [round(by_scenario[50], 4), round(by_scenario[1949], 4)],
                      "groups": len(keys)}, indent=1))


if __name__ == "__main__":
    main()
