from __future__ import annotations

from statistics import mean, pstdev
from typing import Dict

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .env import IncidentCommanderEnv, TASKS
from .models import IncidentCommanderAction
from server.tasks import GRADERS

app = FastAPI(title="Incident Commander OpenEnv", version="1.0.0")

envs: Dict[str, IncidentCommanderEnv] = {}
TASK_DESCRIPTIONS = {
    "easy": "Trace frontend failures to root cause and restore service health",
    "medium": "Investigate dependency failures across frontend/auth/db under pressure",
    "hard": "Recover from config drift, regional network issues, and high-load race conditions with verification",
    "longhaul": "Handle mixed outages (including cross-zone packet loss) with black-box telemetry and budget-aware remediation",
    "blackout": "Survive thundering-herd outages with safe, efficient SRE operations and resilient multi-region recovery",
}

SUPPORTED_REPLAY_POLICIES = ["baseline", "noop", "reasoning"]


def _get_env(task_id: str) -> IncidentCommanderEnv:
    if task_id not in envs:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return envs[task_id]


def _baseline_action(observation: dict) -> IncidentCommanderAction:
    services = observation.get("services", {})
    frontend = services.get("frontend", {})
    terminal_output = " ".join(observation.get("terminal_output", []))
    terminal_lower = terminal_output.lower()

    if "wrong port" in terminal_lower or "connection refused" in terminal_lower:
        return IncidentCommanderAction(
            action_type="edit_config_line",
            config_key="db_port",
            config_value="5432",
            note="baseline repair db config drift",
        )

    if "race condition" in terminal_lower:
        return IncidentCommanderAction(
            action_type="rollback_deployment",
            target_service="auth",
            note="baseline rollback buggy auth deploy",
        )

    if "packet loss" in terminal_lower or "replication lag" in terminal_lower:
        return IncidentCommanderAction(action_type="failover_database", note="baseline failover")

    if int(observation.get("step", 0) or 0) < 2:
        return IncidentCommanderAction(action_type="get_metrics", target_service="frontend", note="baseline initial triage")

    if float(frontend.get("observed_error_rate", frontend.get("error_rate", 0.0)) or 0.0) > 0.35:
        return IncidentCommanderAction(action_type="read_last_n_logs", target_service="auth", n_lines=30, note="baseline triage auth logs")

    if float(observation.get("p95_latency", 0.0) or 0.0) > 280:
        return IncidentCommanderAction(action_type="scale_up_replicas", target_service="frontend", delta_instances=1, note="baseline scale")

    if float(observation.get("traffic_level", 0.0) or 0.0) > 1.6:
        return IncidentCommanderAction(action_type="load_test", note="baseline repro load")

    if int(observation.get("step", 0) or 0) % 4 == 0:
        return IncidentCommanderAction(action_type="run_healthcheck", target_service="frontend", note="baseline verify")

    return IncidentCommanderAction(action_type="noop", note="baseline noop")


def _noop_action(_: dict) -> IncidentCommanderAction:
    return IncidentCommanderAction(action_type="noop", note="replay-noop")


def _reasoning_action(observation: dict) -> IncidentCommanderAction:
    terminal_output = " ".join(observation.get("terminal_output", []))
    terminal_lower = terminal_output.lower()
    services = observation.get("services", {})
    frontend = services.get("frontend", {})
    auth = services.get("auth", {})
    db = services.get("db", {})
    step = int(observation.get("step", 0) or 0)

    if step < 2:
        return IncidentCommanderAction(action_type="get_metrics", target_service="frontend", note="reasoning initial triage")

    if "wrong port" in terminal_lower or "connection refused" in terminal_lower:
        return IncidentCommanderAction(
            action_type="edit_config_line",
            config_key="db_port",
            config_value="5432",
            note="reasoning fix config drift",
        )

    if "packet loss" in terminal_lower or "replication lag" in terminal_lower:
        return IncidentCommanderAction(action_type="failover_database", note="reasoning regional failover")

    if "race condition" in terminal_lower:
        return IncidentCommanderAction(action_type="rollback_deployment", target_service="auth", note="reasoning rollback")

    if float(auth.get("error_rate", 0.0) or 0.0) > 0.35:
        return IncidentCommanderAction(action_type="read_last_n_logs", target_service="auth", n_lines=35, note="reasoning inspect auth logs")

    if float(db.get("error_rate", 0.0) or 0.0) > 0.25:
        return IncidentCommanderAction(action_type="scale_up_replicas", target_service="db", delta_instances=1, note="reasoning db headroom")

    if float(observation.get("p95_latency", 0.0) or 0.0) > 280.0:
        return IncidentCommanderAction(action_type="scale_up_replicas", target_service="frontend", delta_instances=1, note="reasoning frontend scale")

    if float(observation.get("traffic_level", 0.0) or 0.0) > 1.7 and float(frontend.get("healthy", True)):
        return IncidentCommanderAction(action_type="run_healthcheck", target_service="frontend", note="reasoning verify under traffic")

    return IncidentCommanderAction(action_type="noop", note="reasoning wait")


def _policy_action(policy: str, observation: dict) -> IncidentCommanderAction:
    if policy == "noop":
        return _noop_action(observation)
    if policy == "reasoning":
        return _reasoning_action(observation)
    return _baseline_action(observation)


def _extract_failure_taxonomy(episode_details: dict, final_score: float) -> Dict[str, bool]:
    return {
        "sla_failure": bool(episode_details.get("ended_by_sla_failure", False)),
        "budget_failure": bool(episode_details.get("ended_by_budget_failure", False)),
        "low_score_failure": final_score < 0.5,
    }


def _rollout_episode(task_id: str, seed: int, policy: str, max_steps: int | None = None) -> Dict[str, object]:
    env = IncidentCommanderEnv(TASKS[task_id], seed=seed)
    observation = env.reset(seed=seed)
    step_limit = env.task.max_steps if max_steps is None else int(np.clip(max_steps, 1, env.task.max_steps))
    steps = []

    while not env.done and env.timestep < step_limit:
        action = _policy_action(policy, observation.model_dump())
        before_state = env.get_state()
        next_observation, reward, done, info = env.step(action)
        after_state = env.get_state()

        steps.append(
            {
                "step": int(before_state.get("step", 0)),
                "phase": str(next_observation.phase),
                "action": action.model_dump(),
                "reward": reward.model_dump(),
                "done": bool(done),
                "info": info,
                "state_before": before_state,
                "state_after": after_state,
                "observation": next_observation.model_dump(),
            }
        )
        observation = next_observation

    episode_result = env.build_episode_result()
    score = float(GRADERS[task_id](episode_result))
    details = episode_result.model_dump()
    metrics_payload = env.get_metrics()
    failure_taxonomy = _extract_failure_taxonomy(details, score)

    return {
        "task_id": task_id,
        "seed": seed,
        "policy": policy,
        "steps": steps,
        "score": round(score, 6),
        "episode_details": details,
        "metrics": metrics_payload,
        "failure_taxonomy": failure_taxonomy,
        "timeline": list(metrics_payload.get("timeline", [])),
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
      <head>
        <title>Incident Commander OpenEnv</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 0; background: linear-gradient(135deg, #08111f, #11233d 60%, #1d355d); color: #eef3ff; }
          .wrap { max-width: 980px; margin: 0 auto; padding: 56px 24px; }
          .hero { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); border-radius: 20px; padding: 32px; box-shadow: 0 24px 80px rgba(0,0,0,0.22); }
          .grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr)); gap: 16px; margin-top: 24px; }
          .card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.10); border-radius: 16px; padding: 18px; }
          a { color: #8ec5ff; text-decoration: none; }
          a:hover { text-decoration: underline; }
          p { line-height: 1.6; color: #d8e5ff; }
        </style>
      </head>
      <body>
        <div class="wrap">
          <div class="hero">
            <h1>🐢 Incident Commander OpenEnv</h1>
            <p>An RL benchmark where an LLM acts as the on-call SRE for a simulated AI platform during deploy regressions, traffic spikes, and cascading outages.</p>
            <div class="grid">
              <div class="card"><h2>Quick Links</h2><ul><li><a href="/docs">API Docs</a></li><li><a href="/health">Health</a></li><li><a href="/tasks">Tasks</a></li><li><a href="/metrics">Metrics</a></li></ul></div>
              <div class="card"><h2>Objective</h2><p>Restore uptime, keep latency under control, avoid SLA breaches, and manage cost under changing incident conditions.</p></div>
              <div class="card"><h2>Get Started</h2><p>Open <a href="/docs">/docs</a>, then call <a href="/reset">/reset</a> and <a href="/step">/step</a> to try the environment.</p></div>
            </div>
          </div>
        </div>
      </body>
    </html>
    """


@app.post("/reset")
async def reset(task_id: str = "easy", seed: int = 42):
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail="Unknown task_id")
    env = IncidentCommanderEnv(TASKS[task_id], seed=seed)
    envs[task_id] = env
    obs = env.reset(seed=seed)
    return {"observation": obs.model_dump(), "task_id": task_id}


@app.post("/step")
async def step(task_id: str, action: IncidentCommanderAction):
    env = _get_env(task_id)
    obs, reward, done, info = env.step(action)
    return {"observation": obs.model_dump(), "reward": reward.model_dump(), "done": done, "info": info}


@app.get("/state")
async def state(task_id: str = "easy"):
    if task_id not in envs:
        return {"status": "not_initialized", "task_id": task_id}
    return envs[task_id].get_state()


@app.get("/grade")
async def grade(task_id: str = "easy"):
    env = _get_env(task_id)
    episode_result = env.build_episode_result()
    score = GRADERS[task_id](episode_result)
    return {"score": score, "task_id": task_id, "details": episode_result.model_dump()}


@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {
                "id": task_id,
                "description": TASK_DESCRIPTIONS[task_id],
                "max_steps": task.max_steps,
            }
            for task_id, task in TASKS.items()
        ]
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": app.version,
        "active_task_contexts": len(envs),
        "supported_tasks": list(TASKS.keys()),
    }


@app.get("/metrics")
async def metrics(task_id: str = "easy", include_trace: bool = True):
    env = _get_env(task_id)
    payload = env.get_metrics()
    if not include_trace:
        payload = dict(payload)
        payload.pop("reward_trace", None)
    return payload


@app.get("/report")
async def report(task_id: str = "easy"):
    env = _get_env(task_id)
    metrics_payload = env.get_metrics()
    episode_result = env.build_episode_result()
    grade_score = float(GRADERS[task_id](episode_result))
    top_risks = [incident.model_dump() for incident in env.active_incidents if not incident.resolved][:5]
    return {
        "task_id": task_id,
        "grade_score": round(grade_score, 6),
        "episode_result": episode_result.model_dump(),
        "metrics": metrics_payload,
        "top_unresolved_incidents": top_risks,
        "decision_summary": {
            "scenario": env.scenario.scenario_id,
            "resilience_score": metrics_payload.get("resilience_score", 0.0),
            "recommended_focus": "investigate->stabilize->verify",
        },
    }


@app.get("/visualize")
async def visualize(task_id: str = "easy"):
    env = _get_env(task_id)
    header = "[SERVICES]  STAT | CPU | MEM | LATENCY | COST | NOTES"
    lines = [header]
    for name in ("frontend", "auth", "db"):
        service = env.services[name]
        status = "UP" if service.healthy else "DEG"
        cost_tag = "$" * max(1, service.instances - service.spot_instances) + ("s" * service.spot_instances)
        note = service.last_action_result
        if name == "db" and service.p95_latency > 250:
            note = "ROOT CAUSE? " + note

        lines.append(
            f"{name:<10} {status:<4} | {service.cpu_utilization:>3.0f}% | {service.memory_utilization:>3.0f}% | "
            f"{service.p95_latency:>6.0f}ms | {cost_tag:<5} | {note}"
        )

    lines.append("")
    lines.append(
        f"phase={env.phase} step={env.timestep}/{env.task.max_steps} incidents={sum(1 for i in env.active_incidents if not i.resolved)} "
        f"sla_breaches={env.sla_breaches}"
    )
    return {"task_id": task_id, "ascii": "\n".join(lines)}


@app.get("/baseline")
async def baseline(task_id: str = "easy", episodes: int = 5):
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail="Unknown task_id")
    episodes = int(np.clip(episodes, 1, 20))
    scores = []
    for index in range(episodes):
        seed = 42 + index
        env = IncidentCommanderEnv(TASKS[task_id], seed=seed)
        obs = env.reset(seed=seed)
        while not env.done:
            action = _baseline_action(obs.model_dump())
            obs, _, done, _ = env.step(action)
            if done:
                break
        scores.append(GRADERS[task_id](env.build_episode_result()))
    return {"task_id": task_id, "episodes": episodes, "avg_score": round(float(np.mean(scores)), 3), "scores": [round(float(s), 3) for s in scores]}


@app.get("/benchmark_matrix")
async def benchmark_matrix(episodes: int = 3):
    episodes = int(np.clip(episodes, 1, 10))
    results = {}
    for task_id in TASKS:
        scores = []
        for index in range(episodes):
            seed = 101 + index
            env = IncidentCommanderEnv(TASKS[task_id], seed=seed)
            obs = env.reset(seed=seed)
            while not env.done:
                action = _baseline_action(obs.model_dump())
                obs, _, done, _ = env.step(action)
                if done:
                    break
            scores.append(float(GRADERS[task_id](env.build_episode_result())))
        results[task_id] = {
            "avg": round(float(np.mean(scores)), 4),
            "min": round(float(np.min(scores)), 4),
            "max": round(float(np.max(scores)), 4),
            "scores": [round(float(v), 4) for v in scores],
        }
    return {"episodes": episodes, "matrix": results}


@app.get("/replay")
async def replay(task_id: str = "easy", seed: int = 42, policy: str = "baseline", max_steps: int | None = None):
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail="Unknown task_id")
    if policy not in SUPPORTED_REPLAY_POLICIES:
        raise HTTPException(status_code=400, detail=f"Unsupported policy. Use one of {SUPPORTED_REPLAY_POLICIES}")

    payload = _rollout_episode(task_id=task_id, seed=int(seed), policy=policy, max_steps=max_steps)
    timeline = payload.get("timeline", [])
    tail = timeline[-6:] if isinstance(timeline, list) else []
    payload["incident_narrative"] = {
        "summary": (
            f"Policy '{policy}' completed task '{task_id}' with score {payload['score']:.3f}. "
            f"SLA failure={payload['failure_taxonomy']['sla_failure']}, budget failure={payload['failure_taxonomy']['budget_failure']}."
        ),
        "timeline_tail": tail,
        "step_count": len(payload.get("steps", [])),
    }
    return payload


@app.get("/evaluation_report")
async def evaluation_report(policy: str = "baseline", episodes_per_task: int = 3, seed_start: int = 42):
    if policy not in SUPPORTED_REPLAY_POLICIES:
        raise HTTPException(status_code=400, detail=f"Unsupported policy. Use one of {SUPPORTED_REPLAY_POLICIES}")

    episodes_per_task = int(np.clip(episodes_per_task, 1, 10))
    seed_start = int(seed_start)
    per_task = {}
    all_scores = []
    failure_totals = {
        "sla_failure": 0,
        "budget_failure": 0,
        "low_score_failure": 0,
    }

    for task_id in TASKS:
        task_scores = []
        task_failure_counts = {
            "sla_failure": 0,
            "budget_failure": 0,
            "low_score_failure": 0,
        }
        seed_runs = []

        for idx in range(episodes_per_task):
            seed = seed_start + idx
            rollout = _rollout_episode(task_id=task_id, seed=seed, policy=policy)
            score = float(rollout["score"])
            task_scores.append(score)
            all_scores.append(score)

            ft = rollout["failure_taxonomy"]
            for key in task_failure_counts:
                if ft.get(key, False):
                    task_failure_counts[key] += 1
                    failure_totals[key] += 1

            seed_runs.append(
                {
                    "seed": seed,
                    "score": round(score, 6),
                    "failure_taxonomy": ft,
                    "step_count": len(rollout.get("steps", [])),
                }
            )

        task_avg = mean(task_scores)
        task_std = pstdev(task_scores) if len(task_scores) > 1 else 0.0
        robustness = max(0.0, min(1.0, 1.0 - min(1.0, task_std)))
        per_task[task_id] = {
            "avg_score": round(task_avg, 6),
            "min_score": round(min(task_scores), 6),
            "max_score": round(max(task_scores), 6),
            "std_dev": round(task_std, 6),
            "robustness_score": round(robustness, 6),
            "failure_taxonomy": task_failure_counts,
            "seed_runs": seed_runs,
        }

    global_avg = mean(all_scores) if all_scores else 0.0
    global_std = pstdev(all_scores) if len(all_scores) > 1 else 0.0
    overall_robustness = max(0.0, min(1.0, 1.0 - min(1.0, global_std)))

    return {
        "policy": policy,
        "episodes_per_task": episodes_per_task,
        "seed_start": seed_start,
        "summary": {
            "global_avg_score": round(global_avg, 6),
            "global_std_dev": round(global_std, 6),
            "overall_robustness_score": round(overall_robustness, 6),
        },
        "failure_taxonomy_totals": failure_totals,
        "per_task": per_task,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
