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
    frontend = obs.services.get("frontend")
    auth = obs.services.get("auth")
    db = obs.services.get("db")

    if frontend and frontend.error_rate > 0.35:
        return IncidentCommanderAction(action_type="read_last_n_logs", target_service="auth", n_lines=25, note="triage auth logs")

    if auth and auth.error_rate > 0.30:
        return IncidentCommanderAction(action_type="check_network_connectivity", target_service="auth", fallback_service="db", note="network check")

    if db and db.error_rate > 0.25:
        return IncidentCommanderAction(action_type="scale_up_replicas", target_service="db", delta_instances=1, note="db headroom")

    if obs.p95_latency > 280:
        return IncidentCommanderAction(action_type="scale_up_replicas", target_service="frontend", delta_instances=1, note="frontend scale")

    if obs.phase in {"surge", "thundering-herd"} and obs.traffic_level > 1.5:
        return IncidentCommanderAction(action_type="load_test", note="repro under pressure")

    if obs.step % 5 == 0:
        return IncidentCommanderAction(action_type="run_healthcheck", target_service="frontend", note="verify")

    return IncidentCommanderAction(action_type="noop", note="stable")


class ReasoningPolicy:
    def __init__(self) -> None:
        self.last_traffic = 0.0

    def __call__(self, obs: IncidentCommanderObservation) -> IncidentCommanderAction:
        traffic_roc = float(obs.traffic_level - self.last_traffic)
        self.last_traffic = float(obs.traffic_level)

        auth = obs.services.get("auth")
        db = obs.services.get("db")

        if obs.step < 2:
            return IncidentCommanderAction(action_type="get_metrics", target_service="frontend", note="initial triage")

        if obs.step == 2:
            return IncidentCommanderAction(action_type="read_last_n_logs", target_service="auth", n_lines=40, note="trace root cause")

        if auth and auth.error_rate > 0.40:
            return IncidentCommanderAction(action_type="edit_config_line", config_key="db_port", config_value="5432", note="fix config drift")

        if db and db.error_rate > 0.25:
            return IncidentCommanderAction(
                action_type="scale_up_replicas",
                target_service="db",
                delta_instances=1,
                note="db stability",
            )

        if obs.phase in {"surge", "thundering-herd"} and traffic_roc > 0.1:
            return IncidentCommanderAction(action_type="rollback_deployment", target_service="auth", note="mitigate heisenbug")

        if obs.step % 4 == 0:
            return IncidentCommanderAction(action_type="run_healthcheck", note="cluster verify")

        return greedy_policy(obs)


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
    reasoning_scores = [run_episode(task_id, seed, ReasoningPolicy()) for seed in seeds]
    return {
        "noop": round(mean(noop_scores), 3),
        "greedy": round(mean(greedy_scores), 3),
        "reasoning": round(mean(reasoning_scores), 3),
    }


def main() -> None:
    seeds = list(range(42, 52))
    print("task,noop,greedy,reasoning")
    for task_id in TASKS:
        result = benchmark(task_id, seeds)
        print(f"{task_id},{result['noop']:.3f},{result['greedy']:.3f},{result['reasoning']:.3f}")


if __name__ == "__main__":
    main()
