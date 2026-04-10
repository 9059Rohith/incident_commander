from __future__ import annotations

import json
from math import sqrt
from statistics import mean, pstdev
from typing import Dict, List

from app.main import _rollout_episode
from app.env import TASKS


def _stats(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {"avg": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    avg = mean(scores)
    std = pstdev(scores) if len(scores) > 1 else 0.0
    margin = 1.96 * (std / sqrt(max(1, len(scores))))
    return {
        "avg": round(avg, 6),
        "std": round(std, 6),
        "ci95_low": round(max(0.0, avg - margin), 6),
        "ci95_high": round(min(1.0, avg + margin), 6),
    }


def main() -> None:
    seeds = list(range(42, 72))
    policies = ["noop", "baseline", "reasoning"]

    report: Dict[str, object] = {
        "seeds": seeds,
        "policies": policies,
        "tasks": {},
    }

    for task_id in TASKS:
        task_scores: Dict[str, List[float]] = {policy: [] for policy in policies}
        for seed in seeds:
            for policy in policies:
                rollout = _rollout_episode(task_id=task_id, seed=seed, policy=policy)
                task_scores[policy].append(float(rollout["score"]))

        report["tasks"][task_id] = {
            "noop": _stats(task_scores["noop"]),
            "baseline": _stats(task_scores["baseline"]),
            "reasoning": _stats(task_scores["reasoning"]),
            "reasoning_minus_baseline": round(mean(task_scores["reasoning"]) - mean(task_scores["baseline"]), 6),
            "baseline_minus_noop": round(mean(task_scores["baseline"]) - mean(task_scores["noop"]), 6),
        }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
