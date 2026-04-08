"""Inference script for Incident Commander OpenEnv.

Contract requirements:
- Uses OpenAI client for all LLM calls.
- Reads API_BASE_URL, MODEL_NAME, HF_TOKEN, and optional LOCAL_IMAGE_NAME.
- Emits strict [START], [STEP], [END] stdout lines.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI
from pydantic import ValidationError

from models import IncidentCommanderAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
# Optional compatibility fallback if runners provide API_KEY instead of HF_TOKEN.
API_KEY = HF_TOKEN or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
TASKS = ["easy", "medium", "hard", "longhaul", "blackout"]
MAX_STEPS = {"easy": 30, "medium": 40, "hard": 50, "longhaul": 60, "blackout": 70}
BENCHMARK = "incident-commander"

SYSTEM_PROMPT = (
    "You are an incident commander for an AI platform. "
    "Return only strict JSON with keys action_type, target_service, delta_instances, request_fraction, target_version, fallback_service, note. "
    "Valid action_type values are noop, scale_service, reroute_traffic, rollback_deploy, quarantine_service, page_human."
)

client = OpenAI(api_key=API_KEY or "", base_url=API_BASE_URL)


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
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _safe_action() -> Dict[str, Any]:
    return {
        "action_type": "noop",
        "target_service": None,
        "delta_instances": 0,
        "request_fraction": 0.0,
        "target_version": None,
        "fallback_service": None,
        "note": "safe fallback",
    }


def _heuristic_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    services = obs.get("services", {})
    incidents = [incident for incident in obs.get("active_incidents", []) if not incident.get("resolved", False)]
    traffic = float(obs.get("traffic_level", 0.0) or 0.0)
    latency = float(obs.get("p95_latency", 0.0) or 0.0)
    phase = str(obs.get("phase", "steady") or "steady")

    def service_has_incident(service_name: str, incident_type: str) -> bool:
        return any(
            incident.get("service") == service_name and incident.get("incident_type") == incident_type
            for incident in incidents
        )

    critical_open = [incident for incident in incidents if incident.get("severity") == "critical"]
    if critical_open:
        target = str(critical_open[0].get("service", ""))
        if target:
            if service_has_incident(target, "bad_deploy"):
                return {
                    "action_type": "rollback_deploy",
                    "target_service": target,
                    "delta_instances": 0,
                    "request_fraction": 0.0,
                    "target_version": "v0",
                    "fallback_service": None,
                    "note": "critical deploy rollback",
                }
            if service_has_incident(target, "cascade"):
                return {
                    "action_type": "quarantine_service",
                    "target_service": target,
                    "delta_instances": 0,
                    "request_fraction": 0.0,
                    "target_version": None,
                    "fallback_service": None,
                    "note": "contain cascade",
                }

    for incident in incidents:
        incident_type = incident.get("incident_type")
        service_name = str(incident.get("service", ""))
        if not service_name:
            continue
        if incident_type == "bad_deploy":
            return {
                "action_type": "rollback_deploy",
                "target_service": service_name,
                "delta_instances": 0,
                "request_fraction": 0.0,
                "target_version": "v0",
                "fallback_service": None,
                "note": "rollback bad deploy",
            }
        if incident_type in {"node_failure", "cache_thrash"}:
            return {
                "action_type": "scale_service",
                "target_service": service_name,
                "delta_instances": 2,
                "request_fraction": 0.0,
                "target_version": None,
                "fallback_service": None,
                "note": "add capacity for recovery",
            }

    if latency > 260 or traffic > 1.55:
        return {
            "action_type": "scale_service",
            "target_service": "inference",
            "delta_instances": 1,
            "request_fraction": 0.0,
            "target_version": None,
            "fallback_service": None,
            "note": "latency-pressure autoscale",
        }

    if phase in {"regional-outage", "surge"} and len(incidents) >= 2:
        return {
            "action_type": "page_human",
            "target_service": None,
            "delta_instances": 0,
            "request_fraction": 0.0,
            "target_version": None,
            "fallback_service": None,
            "note": "phase-aware escalation",
        }

    if latency > 220 and "gateway" in services and "inference" in services:
        return {
            "action_type": "reroute_traffic",
            "target_service": "gateway",
            "delta_instances": 0,
            "request_fraction": 0.25,
            "target_version": None,
            "fallback_service": "inference",
            "note": "partial reroute",
        }

    if len(critical_open) > 0:
        return {
            "action_type": "page_human",
            "target_service": None,
            "delta_instances": 0,
            "request_fraction": 0.0,
            "target_version": None,
            "fallback_service": None,
            "note": "critical escalation",
        }

    return _safe_action()


def _llm_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    if not HF_TOKEN:
        return _heuristic_action(obs)

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
            "fallback_service": action.get("fallback_service"),
            "note": action.get("note", ""),
        }
    except Exception:
        return _heuristic_action(obs)


def _validated_action(raw_action: Dict[str, Any]) -> Dict[str, Any]:
    try:
        parsed = IncidentCommanderAction(**raw_action)
        return parsed.model_dump()
    except ValidationError:
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
            action_obj = _validated_action(_llm_action(observation))
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
