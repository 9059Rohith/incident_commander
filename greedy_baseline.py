from __future__ import annotations

from statistics import mean
from typing import Callable, Dict, List

from app.env import IncidentCommanderEnv, TASKS
from app.models import IncidentCommanderAction, IncidentCommanderObservation
from server.tasks import GRADERS

PolicyFn = Callable[[IncidentCommanderObservation], IncidentCommanderAction]


def noop_policy(_: IncidentCommanderObservation) -> IncidentCommanderAction:
    return IncidentCommanderAction(action_type="noop", note="noop-baseline")


def greedy_policy(obs: IncidentCommanderObservation) -> IncidentCommanderAction:
    incidents = [incident for incident in obs.active_incidents if not incident.resolved]
    critical = [incident for incident in incidents if incident.severity == "critical"]
    if critical:
        return IncidentCommanderAction(
            action_type="quarantine_service",
            target_service=critical[0].service,
            note="critical containment",
        )

    for incident in incidents:
        if incident.incident_type == "bad_deploy":
            return IncidentCommanderAction(
                action_type="rollback_deploy",
                target_service=incident.service,
                target_version="v0",
                note="rollback bad deploy",
            )
        if incident.incident_type in {"node_failure", "cache_thrash"}:
            return IncidentCommanderAction(
                action_type="scale_service",
                target_service=incident.service,
                delta_instances=2,
                note="capacity recovery",
            )

    if obs.p95_latency > 260 or obs.traffic_level > 1.6:
        return IncidentCommanderAction(
            action_type="scale_service",
            target_service="inference",
            delta_instances=1,
            note="latency guardrail",
        )

    if obs.phase in {"regional-outage", "surge"} and len(incidents) >= 2:
        return IncidentCommanderAction(action_type="page_human", note="phase escalation")

    if obs.p95_latency > 220:
        return IncidentCommanderAction(
            action_type="reroute_traffic",
            target_service="gateway",
            fallback_service="inference",
            request_fraction=0.25,
            note="reroute pressure",
        )

    return IncidentCommanderAction(action_type="noop", note="stable")


def run_episode(task_id: str, seed: int, policy: PolicyFn) -> float:
    env = IncidentCommanderEnv(TASKS[task_id], seed=seed)
    obs = env.reset(seed=seed)
    while not env.done:
        action = policy(obs)
        obs, _, done, _ = env.step(action)
        if done:
            break
    return float(GRADERS[task_id](env.build_episode_result()))


def benchmark(task_id: str, seeds: List[int]) -> Dict[str, float]:
    noop_scores = [run_episode(task_id, seed, noop_policy) for seed in seeds]
    greedy_scores = [run_episode(task_id, seed, greedy_policy) for seed in seeds]
    return {
        "noop": round(mean(noop_scores), 3),
        "greedy": round(mean(greedy_scores), 3),
    }


def main() -> None:
    seeds = list(range(42, 52))
    print("task,noop,greedy")
    for task_id in TASKS:
        result = benchmark(task_id, seeds)
        print(f"{task_id},{result['noop']:.3f},{result['greedy']:.3f}")


if __name__ == "__main__":
    main()
