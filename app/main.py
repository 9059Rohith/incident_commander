from __future__ import annotations

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
    "easy": "Recover a single degraded API service under rising latency",
    "medium": "Handle a bad deploy and traffic surge with rollback and scaling",
    "hard": "Stop a cascading failure across API, inference, and database services",
    "longhaul": "Silent-killer memory leak: keep healthy services from crashing under creeping memory pressure",
    "blackout": "Thundering-herd outage: scale fast enough to prevent DB lockup without blowing budget",
}


def _get_env(task_id: str) -> IncidentCommanderEnv:
    if task_id not in envs:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return envs[task_id]


def _baseline_action(observation: dict) -> IncidentCommanderAction:
    incidents = [incident for incident in observation.get("active_incidents", []) if not incident.get("resolved", False)]
    critical = [incident for incident in incidents if incident.get("severity") == "critical"]
    if critical:
        service = critical[0].get("service")
        if service:
            return IncidentCommanderAction(action_type="quarantine_service", target_service=service, note="baseline critical containment")

    for incident in incidents:
        service = incident.get("service")
        incident_type = incident.get("incident_type")
        if incident_type == "bad_deploy" and service:
            return IncidentCommanderAction(action_type="rollback_deploy", target_service=service, target_version="v0", note="baseline rollback")
        if incident_type in {"node_failure", "cache_thrash"} and service:
            return IncidentCommanderAction(action_type="scale_service", target_service=service, delta_instances=2, note="baseline scale")

    if float(observation.get("p95_latency", 0.0) or 0.0) > 260:
        return IncidentCommanderAction(action_type="scale_service", target_service="inference", delta_instances=1, note="baseline latency scale")
    return IncidentCommanderAction(action_type="noop", note="baseline noop")


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
    return {"status": "ok"}


@app.get("/metrics")
async def metrics(task_id: str = "easy", include_trace: bool = True):
    env = _get_env(task_id)
    payload = env.get_metrics()
    if not include_trace:
        payload = dict(payload)
        payload.pop("reward_trace", None)
    return payload


@app.get("/visualize")
async def visualize(task_id: str = "easy"):
    env = _get_env(task_id)
    header = "[SERVICES]  STAT | CPU | MEM | LATENCY | COST | NOTES"
    lines = [header]
    for name in ("gateway", "inference", "database"):
        service = env.services[name]
        status = "UP" if service.healthy else "DEG"
        cost_tag = "$" * max(1, service.instances - service.spot_instances) + ("s" * service.spot_instances)
        note = service.last_action_result
        if name == "database" and service.p95_latency > 250:
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
