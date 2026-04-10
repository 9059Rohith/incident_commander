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


def _mean_diff_z(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    paired = [a[i] - b[i] for i in range(n)]
    avg = mean(paired)
    std = pstdev(paired) if len(paired) > 1 else 0.0
    if std == 0.0:
        return 0.0
    return round(avg / (std / sqrt(max(1, len(paired)))), 6)


def main() -> None:
    seeds = list(range(42, 72))
    policies = ["noop", "random-safe", "baseline", "reasoning", "trained"]

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
            "random-safe": _stats(task_scores["random-safe"]),
            "baseline": _stats(task_scores["baseline"]),
            "reasoning": _stats(task_scores["reasoning"]),
            "trained": _stats(task_scores["trained"]),
            "reasoning_minus_baseline": round(mean(task_scores["reasoning"]) - mean(task_scores["baseline"]), 6),
            "baseline_minus_noop": round(mean(task_scores["baseline"]) - mean(task_scores["noop"]), 6),
            "trained_minus_reasoning": round(mean(task_scores["trained"]) - mean(task_scores["reasoning"]), 6),
            "z_reasoning_vs_baseline": _mean_diff_z(task_scores["reasoning"], task_scores["baseline"]),
            "z_trained_vs_reasoning": _mean_diff_z(task_scores["trained"], task_scores["reasoning"]),
        }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
