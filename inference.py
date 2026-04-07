"""Baseline inference script for Incident Commander OpenEnv.

Emits strict [START], [STEP], [END] lines.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
TASKS = ["easy", "medium", "hard", "longhaul"]
MAX_STEPS = {"easy": 30, "medium": 40, "hard": 50, "longhaul": 60}
BENCHMARK = "incident-commander"

SYSTEM_PROMPT = (
    "You are an incident commander for an AI platform. "
    "Return only strict JSON with keys action_type, target_service, delta_instances, request_fraction, target_version, note. "
    "Valid action_type values are noop, scale_service, reroute_traffic, rollback_deploy, quarantine_service, page_human."
)

client = OpenAI(api_key=OPENAI_API_KEY or HF_TOKEN, base_url=API_BASE_URL)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _safe_action() -> Dict[str, Any]:
    return {
        "action_type": "noop",
        "target_service": None,
        "delta_instances": 0,
        "request_fraction": 0.0,
        "target_version": None,
        "note": "safe fallback",
    }


def _llm_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    if not HF_TOKEN and not OPENAI_API_KEY:
        return _safe_action()

    user_msg = json.dumps(
        {
            "step": obs.get("step", 0),
            "traffic_level": obs.get("traffic_level", 0.0),
            "uptime": obs.get("uptime", 0.0),
            "sla_breaches": obs.get("sla_breaches", 0),
            "services": obs.get("services", {}),
            "incidents": obs.get("active_incidents", []),
        },
        separators=(",", ":"),
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=220,
        )
        raw = (response.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.replace("```json", "").replace("```", "").strip()
        action = json.loads(raw)
        return {
            "action_type": action.get("action_type", "noop"),
            "target_service": action.get("target_service"),
            "delta_instances": int(action.get("delta_instances", 0) or 0),
            "request_fraction": float(action.get("request_fraction", 0.0) or 0.0),
            "target_version": action.get("target_version"),
            "note": action.get("note", ""),
        }
    except Exception:
        return _safe_action()


def run_task(task_id: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    log_start(task_id, BENCHMARK, MODEL_NAME)

    try:
        reset_response = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id, "seed": 42}, timeout=15)
        reset_response.raise_for_status()
        observation = reset_response.json().get("observation", {})

        for step in range(1, MAX_STEPS[task_id] + 1):
            action_obj = _llm_action(observation)
            action_str = json.dumps(action_obj, separators=(",", ":"))
            reward = 0.0
            done = False
            last_error: Optional[str] = None

            try:
                step_response = requests.post(
                    f"{ENV_URL}/step",
                    params={"task_id": task_id},
                    json=action_obj,
                    timeout=15,
                )
                step_response.raise_for_status()
                payload = step_response.json()
                observation = payload.get("observation", {})
                reward = float(payload.get("reward", {}).get("total", 0.0))
                done = bool(payload.get("done", False))
                last_error = payload.get("info", {}).get("error", None)
            except Exception as exc:
                done = True
                last_error = str(exc)

            rewards.append(reward)
            steps_taken = step
            log_step(step, action_str, reward, done, last_error)
            if done:
                break

        score = 0.0
        try:
            grade_response = requests.get(f"{ENV_URL}/grade", params={"task_id": task_id}, timeout=15)
            grade_response.raise_for_status()
            score = float(grade_response.json().get("score", 0.0))
        except Exception:
            score = max(0.0, min(1.0, sum(rewards) / max(1.0, float(MAX_STEPS[task_id])) + 0.5))

        success = score >= 0.5
        log_end(success, steps_taken, score, rewards)
        return score
    except Exception:
        log_end(False, steps_taken, 0.0, rewards)
        return 0.0


if __name__ == "__main__":
    for task in TASKS:
        run_task(task)
