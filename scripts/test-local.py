from __future__ import annotations

import json
import os
import sys
from typing import List

import requests

BASE_URL = os.getenv("ENV_URL", "http://localhost:7860")
TASKS: List[str] = ["easy", "medium", "hard", "longhaul", "blackout"]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _post(path: str, **kwargs):
    response = requests.post(f"{BASE_URL}{path}", timeout=20, **kwargs)
    response.raise_for_status()
    return response.json()


def _get(path: str, **kwargs):
    response = requests.get(f"{BASE_URL}{path}", timeout=20, **kwargs)
    response.raise_for_status()
    return response.json()


def run() -> None:
    health = _get("/health")
    _assert(health.get("status") == "ok", "health endpoint failed")

    tasks = _get("/tasks").get("tasks", [])
    task_ids = sorted(task["id"] for task in tasks)
    _assert(task_ids == sorted(TASKS), f"task list mismatch: {task_ids}")

    for task_id in TASKS:
        reset_a = _post("/reset", params={"task_id": task_id, "seed": 123})
        obs_a = reset_a["observation"]
        reset_b = _post("/reset", params={"task_id": task_id, "seed": 123})
        obs_b = reset_b["observation"]
        _assert(obs_a == obs_b, f"reset determinism failed for {task_id}")

        step_payload = {
            "action_type": "scale_up_replicas",
            "target_service": "frontend",
            "delta_instances": 1,
            "fallback_service": None,
            "note": "test step",
        }
        step = _post("/step", params={"task_id": task_id}, json=step_payload)
        _assert("observation" in step and "reward" in step and "done" in step, f"step response invalid for {task_id}")

        grade = _get("/grade", params={"task_id": task_id})
        score = float(grade.get("score", -1))
        _assert(0.0 <= score <= 1.0, f"grade out of range for {task_id}: {score}")

        metrics = _get("/metrics", params={"task_id": task_id})
        for key in ["uptime_score", "latency_score", "sla_score", "cost_score", "recovery_score", "action_discipline_score"]:
            value = float(metrics.get(key, -1))
            _assert(0.0 <= value <= 1.0, f"metric out of range for {task_id}:{key}={value}")
        trace = metrics.get("reward_trace", [])
        _assert(isinstance(trace, list), f"reward_trace must be list for {task_id}")

        metrics_with_trace = _get("/metrics", params={"task_id": task_id, "include_trace": "true"})
        _assert(isinstance(metrics_with_trace.get("reward_trace", None), list), f"reward_trace should be present when include_trace=true for {task_id}")

        metrics_no_trace = _get("/metrics", params={"task_id": task_id, "include_trace": "false"})
        _assert("reward_trace" not in metrics_no_trace, f"reward_trace should be omitted when include_trace=false for {task_id}")

        viz = _get("/visualize", params={"task_id": task_id})
        _assert(viz.get("task_id") == task_id, f"visualize task mismatch for {task_id}")
        _assert("ascii" in viz and isinstance(viz["ascii"], str) and len(viz["ascii"]) > 0, f"visualize ascii missing for {task_id}")

        baseline = _get("/baseline", params={"task_id": task_id, "episodes": 2})
        _assert(baseline.get("task_id") == task_id, f"baseline task mismatch for {task_id}")
        _assert(int(baseline.get("episodes", 0)) == 2, f"baseline episodes mismatch for {task_id}")
        baseline_avg = float(baseline.get("avg_score", -1))
        _assert(0.0 <= baseline_avg <= 1.0, f"baseline avg score out of range for {task_id}: {baseline_avg}")
        baseline_scores = baseline.get("scores", [])
        _assert(isinstance(baseline_scores, list) and len(baseline_scores) == 2, f"baseline scores malformed for {task_id}")

    print(json.dumps({"status": "ok", "checked_tasks": TASKS}))


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"test-local failed: {exc}")
        print("hint: start the server first, e.g. `python -m uvicorn app.main:app --host 0.0.0.0 --port 7860`")
        sys.exit(1)
