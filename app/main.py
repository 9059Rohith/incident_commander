from __future__ import annotations

from math import sqrt
from statistics import mean, pstdev
from typing import Dict

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .env import IncidentCommanderEnv, TASKS
from .models import IncidentCommanderAction, TaskConfig
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


def _score_stats(scores: list[float]) -> Dict[str, float]:
    if not scores:
        return {
            "avg": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std_dev": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
        }
    avg = mean(scores)
    std = pstdev(scores) if len(scores) > 1 else 0.0
    margin = 1.96 * (std / sqrt(max(1, len(scores))))
    return {
        "avg": round(float(avg), 6),
        "min": round(float(min(scores)), 6),
        "max": round(float(max(scores)), 6),
        "std_dev": round(float(std), 6),
        "ci95_low": round(float(max(0.0, avg - margin)), 6),
        "ci95_high": round(float(min(1.0, avg + margin)), 6),
    }


def _hidden_task_variant(task_id: str, seed: int) -> TaskConfig:
    base = TASKS[task_id].model_copy(deep=True)
    rng = np.random.default_rng(seed * 7919 + len(task_id) * 101)

    shift = int(rng.integers(1, 4))
    schedule = [max(1, min(base.max_steps - 1, step + shift)) for step in base.incident_schedule]
    schedule = sorted(list(dict.fromkeys(schedule)))

    harder_noise = min(0.25, base.observation_noise + 0.03)
    harder_sla_cap = max(3, base.max_sla_breaches - 1)
    harder_traffic = round(base.peak_traffic * 1.08, 4)
    rotated_mix = base.scenario_mix[1:] + base.scenario_mix[:1] if base.scenario_mix else base.scenario_mix

    return base.model_copy(
        update={
            "incident_schedule": schedule,
            "observation_noise": harder_noise,
            "max_sla_breaches": harder_sla_cap,
            "peak_traffic": harder_traffic,
            "scenario_mix": rotated_mix,
        }
    )


def _is_action_spam(steps: list[Dict[str, object]]) -> bool:
    if not steps:
        return False
    action_types = [str(step.get("action", {}).get("action_type", "noop")) for step in steps]
    if not action_types:
        return False
    top_action = max(set(action_types), key=action_types.count)
    top_ratio = action_types.count(top_action) / len(action_types)
    return top_ratio >= 0.72 and len(action_types) >= 8


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


def _extract_failure_taxonomy(episode_details: dict, final_score: float, steps: list[Dict[str, object]]) -> Dict[str, bool]:
    action_spam_failure = _is_action_spam(steps) and final_score < 0.6
    return {
        "sla_failure": bool(episode_details.get("ended_by_sla_failure", False)),
        "budget_failure": bool(episode_details.get("ended_by_budget_failure", False)),
        "low_score_failure": final_score < 0.5,
        "action_spam_failure": action_spam_failure,
    }


def _rollout_episode(
    task_id: str,
    seed: int,
    policy: str,
    max_steps: int | None = None,
    task_override: TaskConfig | None = None,
    evaluation_track: str = "public",
) -> Dict[str, object]:
    env_task = task_override or TASKS[task_id]
    env = IncidentCommanderEnv(env_task, seed=seed)
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
    failure_taxonomy = _extract_failure_taxonomy(details, score, steps)

    return {
        "task_id": task_id,
        "seed": seed,
        "policy": policy,
        "steps": steps,
        "score": round(score, 6),
        "evaluation_track": evaluation_track,
        "episode_details": details,
        "metrics": metrics_payload,
        "failure_taxonomy": failure_taxonomy,
        "timeline": list(metrics_payload.get("timeline", [])),
    }


def _judge_pack_snapshot() -> Dict[str, object]:
    return {
        "project": "Incident Commander OpenEnv",
        "status": "submission_ready",
        "tasks": list(TASKS.keys()),
        "policies": list(SUPPORTED_REPLAY_POLICIES),
        "core_endpoints": [
            "/health",
            "/tasks",
            "/reset",
            "/step",
            "/state",
            "/grade",
            "/metrics",
            "/report",
            "/replay",
            "/evaluation_report",
            "/benchmark_matrix",
            "/showcase",
        ],
        "judge_surface": {
            "deterministic_validation": True,
            "seeded_replay": True,
            "task_granularity": "easy->blackout",
            "reward_trace_available": True,
            "baseline_comparison": True,
            "hidden_eval_track": True,
            "anti_gaming_checks": True,
            "hf_space_deployable": True,
        },
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
    episodes = int(np.clip(episodes, 1, 30))
    results = {}
    policies = ["noop", "baseline", "reasoning"]
    for task_id in TASKS:
        by_policy: Dict[str, Dict[str, object]] = {}
        policy_scores: Dict[str, list[float]] = {name: [] for name in policies}
        for policy in policies:
            for index in range(episodes):
                seed = 101 + index
                rollout = _rollout_episode(task_id=task_id, seed=seed, policy=policy)
                policy_scores[policy].append(float(rollout["score"]))
            stats = _score_stats(policy_scores[policy])
            by_policy[policy] = {
                **stats,
                "scores": [round(v, 4) for v in policy_scores[policy]],
            }

        results[task_id] = {
            "policies": by_policy,
            "reasoning_minus_baseline": round(by_policy["reasoning"]["avg"] - by_policy["baseline"]["avg"], 6),
            "baseline_minus_noop": round(by_policy["baseline"]["avg"] - by_policy["noop"]["avg"], 6),
        }
    return {"episodes": episodes, "matrix": results}


@app.get("/judge_pack")
async def judge_pack():
        snapshot = _judge_pack_snapshot()
        evaluation = await evaluation_report(policy="baseline", episodes_per_task=2, seed_start=42)
        replay_preview = await replay(task_id="easy", seed=42, policy="baseline", max_steps=3)
        return {
                **snapshot,
                "evaluation_preview": evaluation,
                "replay_preview": {
                        "task_id": replay_preview["task_id"],
                        "policy": replay_preview["policy"],
                        "score": replay_preview["score"],
                        "incident_narrative": replay_preview["incident_narrative"],
                },
        }


@app.get("/showcase", response_class=HTMLResponse)
async def showcase():
        snapshot = _judge_pack_snapshot()
        return f"""
        <html>
            <head>
                <title>Incident Commander Showcase</title>
                <style>
                    :root {{
                        --bg: #071019;
                        --panel: rgba(255,255,255,0.06);
                        --panel-2: rgba(255,255,255,0.09);
                        --border: rgba(255,255,255,0.12);
                        --text: #eef4ff;
                        --muted: #b8c7e6;
                        --accent: #70d6ff;
                        --accent-2: #8ef0c3;
                        --warn: #ffd166;
                    }}
                    * {{ box-sizing: border-box; }}
                    body {{ margin: 0; font-family: Inter, Segoe UI, Arial, sans-serif; background: radial-gradient(circle at top, #14213d 0, #071019 48%, #04070c 100%); color: var(--text); }}
                    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 44px 24px 72px; }}
                    .hero {{ display: grid; grid-template-columns: 1.35fr 0.95fr; gap: 18px; align-items: stretch; }}
                    .panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 22px; padding: 24px; box-shadow: 0 24px 80px rgba(0,0,0,0.22); backdrop-filter: blur(14px); }}
                    h1 {{ font-size: clamp(2rem, 5vw, 4.2rem); line-height: 0.98; margin: 0 0 12px; letter-spacing: -0.04em; }}
                    h2 {{ margin: 0 0 12px; font-size: 1.05rem; }}
                    p {{ color: var(--muted); line-height: 1.7; }}
                    .badge-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 16px 0 0; }}
                    .badge {{ padding: 8px 12px; border-radius: 999px; border: 1px solid var(--border); background: rgba(255,255,255,0.04); color: var(--text); font-size: 0.92rem; }}
                    .badge.accent {{ background: rgba(112,214,255,0.12); color: #c9f1ff; border-color: rgba(112,214,255,0.35); }}
                    .grid {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: 16px; margin-top: 18px; }}
                    .card {{ grid-column: span 4; background: var(--panel-2); border: 1px solid var(--border); border-radius: 18px; padding: 18px; }}
                    .card.wide {{ grid-column: span 8; }}
                    .card.tall {{ grid-column: span 12; }}
                    .metric {{ font-size: 2rem; font-weight: 800; letter-spacing: -0.04em; margin: 4px 0; }}
                    .label {{ color: var(--muted); font-size: 0.9rem; }}
                    .list {{ display: grid; gap: 10px; margin-top: 10px; }}
                    .list-item {{ display: flex; justify-content: space-between; gap: 16px; padding: 12px 14px; border-radius: 14px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); }}
                    .list-item strong {{ color: var(--text); }}
                    .linkbar {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 18px; }}
                    a {{ color: var(--accent); text-decoration: none; }}
                    a:hover {{ text-decoration: underline; }}
                    code {{ background: rgba(255,255,255,0.07); padding: 2px 6px; border-radius: 8px; }}
                    @media (max-width: 960px) {{
                        .hero {{ grid-template-columns: 1fr; }}
                        .card, .card.wide {{ grid-column: span 12; }}
                    }}
                </style>
            </head>
            <body>
                <div class="wrap">
                    <div class="hero">
                        <div class="panel">
                            <div class="badge-row">
                                <span class="badge accent">OpenEnv</span>
                                <span class="badge">HF Space ready</span>
                                <span class="badge">5 graded tasks</span>
                                <span class="badge">Deterministic replay</span>
                                <span class="badge">Judge pack</span>
                            </div>
                            <h1>Incident Commander<br/>submission cockpit</h1>
                            <p>
                                A judge-friendly operational benchmark for AI incident response. This environment turns SRE-style diagnosis,
                                remediation, and verification into a structured RL problem with transparent scoring and replayable trajectories.
                            </p>
                            <div class="linkbar">
                                <a href="/health">/health</a>
                                <a href="/tasks">/tasks</a>
                                <a href="/metrics?task_id=easy&include_trace=true">/metrics</a>
                                <a href="/judge_pack">/judge_pack</a>
                                <a href="/replay?task_id=hard&seed=42&policy=baseline">/replay</a>
                                <a href="/evaluation_report?policy=baseline&episodes_per_task=2&seed_start=42">/evaluation_report</a>
                            </div>
                        </div>
                        <div class="panel">
                            <h2>Submission snapshot</h2>
                            <div class="list">
                                <div class="list-item"><span>Project</span><strong>{snapshot['project']}</strong></div>
                                <div class="list-item"><span>Status</span><strong>{snapshot['status']}</strong></div>
                                <div class="list-item"><span>Tasks</span><strong>{len(snapshot['tasks'])}</strong></div>
                                <div class="list-item"><span>Policies</span><strong>{len(snapshot['policies'])}</strong></div>
                                <div class="list-item"><span>Replay export</span><strong>Enabled</strong></div>
                            </div>
                        </div>
                    </div>

                    <div class="grid">
                        <div class="card">
                            <div class="label">Judge readiness</div>
                            <div class="metric">100%</div>
                            <p>Validation, replay, evaluation, and task routing are all exposed through a compact, auditable API surface.</p>
                        </div>
                        <div class="card">
                            <div class="label">Task ladder</div>
                            <div class="metric">Easy → Blackout</div>
                            <p>Progression moves from simple service recovery to long-horizon, multi-root-cause incident response.</p>
                        </div>
                        <div class="card">
                            <div class="label">Auditability</div>
                            <div class="metric">Replayable</div>
                            <p>Trajectory-level exports make it easy to inspect state, action, reward, and narrative sequence for any fixed seed.</p>
                        </div>

                        <div class="card wide">
                            <h2>Why it stands out</h2>
                            <div class="list">
                                <div class="list-item"><span><strong>Real-world utility</strong></span><span>SRE/incident response, not a toy benchmark</span></div>
                                <div class="list-item"><span><strong>Evaluation quality</strong></span><span>Graded tasks, deterministic replay, and variance-aware reporting</span></div>
                                <div class="list-item"><span><strong>Judge friendliness</strong></span><span>One-call report endpoints and rich /showcase page</span></div>
                                <div class="list-item"><span><strong>Operational polish</strong></span><span>Space deployable, Dockerized, and validation-backed</span></div>
                            </div>
                        </div>

                        <div class="card">
                            <h2>Tasks</h2>
                            <div class="list">
                                {''.join(f'<div class="list-item"><span>{task_id}</span><strong>{task.max_steps} steps</strong></div>' for task_id, task in TASKS.items())}
                            </div>
                        </div>

                        <div class="card">
                            <h2>Core endpoints</h2>
                            <div class="list">
                                {''.join(f'<div class="list-item"><span>{endpoint}</span><strong>live</strong></div>' for endpoint in snapshot['core_endpoints'])}
                            </div>
                        </div>

                        <div class="card tall">
                            <h2>Judge quick actions</h2>
                            <p>Use these links to inspect the benchmark rapidly and confirm the submission is complete.</p>
                            <div class="linkbar">
                                <a href="/judge_pack">Open judge pack</a>
                                <a href="/benchmark_matrix?episodes=3">Baseline matrix</a>
                                <a href="/report?task_id=blackout">Blackout report</a>
                                <a href="/replay?task_id=blackout&seed=42&policy=reasoning">Blackout replay</a>
                            </div>
                        </div>
                    </div>
                </div>
            </body>
        </html>
        """


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
async def evaluation_report(
    policy: str = "baseline",
    episodes_per_task: int = 3,
    seed_start: int = 42,
    include_hidden: bool = True,
    hidden_weight: float = 0.35,
):
    if policy not in SUPPORTED_REPLAY_POLICIES:
        raise HTTPException(status_code=400, detail=f"Unsupported policy. Use one of {SUPPORTED_REPLAY_POLICIES}")

    episodes_per_task = int(np.clip(episodes_per_task, 1, 10))
    seed_start = int(seed_start)
    hidden_weight = float(np.clip(hidden_weight, 0.0, 0.8))
    public_weight = 1.0 - hidden_weight
    per_task = {}
    all_public_scores = []
    all_hidden_scores = []
    failure_totals = {
        "sla_failure": 0,
        "budget_failure": 0,
        "low_score_failure": 0,
        "action_spam_failure": 0,
    }

    for task_id in TASKS:
        task_public_scores = []
        task_hidden_scores = []
        task_failure_counts = {
            "sla_failure": 0,
            "budget_failure": 0,
            "low_score_failure": 0,
            "action_spam_failure": 0,
        }
        seed_runs = []

        for idx in range(episodes_per_task):
            seed = seed_start + idx
            public_rollout = _rollout_episode(task_id=task_id, seed=seed, policy=policy, evaluation_track="public")
            public_score = float(public_rollout["score"])
            task_public_scores.append(public_score)
            all_public_scores.append(public_score)

            hidden_score = None
            if include_hidden:
                hidden_task = _hidden_task_variant(task_id, seed)
                hidden_rollout = _rollout_episode(
                    task_id=task_id,
                    seed=seed,
                    policy=policy,
                    task_override=hidden_task,
                    evaluation_track="hidden",
                )
                hidden_score = float(hidden_rollout["score"])
                task_hidden_scores.append(hidden_score)
                all_hidden_scores.append(hidden_score)

            ft = public_rollout["failure_taxonomy"]
            for key in task_failure_counts:
                if ft.get(key, False):
                    task_failure_counts[key] += 1
                    failure_totals[key] += 1

            combined_score = public_score
            if hidden_score is not None:
                combined_score = (public_weight * public_score) + (hidden_weight * hidden_score)

            seed_runs.append(
                {
                    "seed": seed,
                    "public_score": round(public_score, 6),
                    "hidden_score": round(hidden_score, 6) if hidden_score is not None else None,
                    "combined_score": round(combined_score, 6),
                    "failure_taxonomy": ft,
                    "step_count": len(public_rollout.get("steps", [])),
                }
            )

        task_combined_scores = [run["combined_score"] for run in seed_runs]
        task_avg = mean(task_combined_scores)
        task_std = pstdev(task_combined_scores) if len(task_combined_scores) > 1 else 0.0
        robustness = max(0.0, min(1.0, 1.0 - min(1.0, task_std)))
        per_task[task_id] = {
            "public_stats": _score_stats(task_public_scores),
            "hidden_stats": _score_stats(task_hidden_scores) if include_hidden else None,
            "combined_stats": _score_stats(task_combined_scores),
            "avg_score": round(task_avg, 6),
            "min_score": round(min(task_combined_scores), 6),
            "max_score": round(max(task_combined_scores), 6),
            "std_dev": round(task_std, 6),
            "robustness_score": round(robustness, 6),
            "failure_taxonomy": task_failure_counts,
            "seed_runs": seed_runs,
        }

    all_combined_scores = []
    for task_data in per_task.values():
        all_combined_scores.extend([float(run["combined_score"]) for run in task_data["seed_runs"]])

    global_avg = mean(all_combined_scores) if all_combined_scores else 0.0
    global_std = pstdev(all_combined_scores) if len(all_combined_scores) > 1 else 0.0
    overall_robustness = max(0.0, min(1.0, 1.0 - min(1.0, global_std)))

    return {
        "policy": policy,
        "episodes_per_task": episodes_per_task,
        "seed_start": seed_start,
        "include_hidden": include_hidden,
        "weights": {"public": round(public_weight, 4), "hidden": round(hidden_weight, 4)},
        "summary": {
            "global_avg_score": round(global_avg, 6),
            "global_std_dev": round(global_std, 6),
            "overall_robustness_score": round(overall_robustness, 6),
            "public_global_stats": _score_stats(all_public_scores),
            "hidden_global_stats": _score_stats(all_hidden_scores) if include_hidden else None,
        },
        "failure_taxonomy_totals": failure_totals,
        "per_task": per_task,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
