from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from .models import (
    ActiveIncident,
    EpisodeResult,
    IncidentCommanderAction,
    IncidentCommanderObservation,
    IncidentCommanderReward,
    ServiceState,
    TaskConfig,
)
from .reward import RewardCalculator


TASKS: Dict[str, TaskConfig] = {
    "easy": TaskConfig(
        task_id="easy",
        max_steps=30,
        base_traffic=1.0,
        peak_traffic=1.4,
        incident_schedule=[0],
        cost_budget=0.55,
        spike_multiplier=1.2,
    ),
    "medium": TaskConfig(
        task_id="medium",
        max_steps=40,
        base_traffic=1.05,
        peak_traffic=1.75,
        incident_schedule=[0, 7],
        cost_budget=0.65,
        spike_multiplier=1.45,
    ),
    "hard": TaskConfig(
        task_id="hard",
        max_steps=50,
        base_traffic=1.1,
        peak_traffic=2.05,
        incident_schedule=[0, 4, 10],
        cost_budget=0.72,
        spike_multiplier=1.65,
    ),
    "longhaul": TaskConfig(
        task_id="longhaul",
        max_steps=60,
        base_traffic=1.0,
        peak_traffic=2.25,
        incident_schedule=[0, 8, 18, 30, 42, 50],
        cost_budget=0.78,
        spike_multiplier=1.8,
    ),
}


class IncidentCommanderEnv:
    def __init__(self, task: TaskConfig, seed: int = 42):
        self.task = task
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.reward_calculator = RewardCalculator(task)
        self.services: Dict[str, ServiceState] = {}
        self.active_incidents: List[ActiveIncident] = []
        self.timestep = 0
        self.done = False
        self.last_reward = IncidentCommanderReward(total=0.0, uptime=0.0, latency=0.0, sla=0.0, cost=0.0, recovery=0.0)
        self.last_action_result = "initialized"
        self.sla_breaches = 0
        self.total_incidents = 0
        self.resolved_incidents = 0
        self.cumulative_cost = 0.0
        self.uptime_history: List[float] = []
        self.latency_history: List[float] = []
        self.phase = "steady"
        self.human_attention_steps = 0
        self._init_state()

    def _init_state(self) -> None:
        self.services = {
            "gateway": ServiceState(name="gateway", instances=3, desired_instances=3, version="v1", rollback_version="v0"),
            "inference": ServiceState(name="inference", instances=4, desired_instances=4, version="v1", rollback_version="v0"),
            "database": ServiceState(name="database", instances=2, desired_instances=2, version="v1", rollback_version="v0"),
        }
        self.active_incidents = []
        self.timestep = 0
        self.done = False
        self.last_action_result = "reset complete"
        self.sla_breaches = 0
        self.total_incidents = 0
        self.resolved_incidents = 0
        self.cumulative_cost = 0.0
        self.uptime_history = []
        self.latency_history = []
        self.phase = "steady"
        self.human_attention_steps = 0

    def reset(self, seed: Optional[int] = None) -> IncidentCommanderObservation:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.reward_calculator = RewardCalculator(self.task)
        self._init_state()
        self._spawn_scheduled_incidents(step_index=0)
        return self._get_observation()

    def step(self, action: IncidentCommanderAction):
        if self.done:
            return self._get_observation(), self.last_reward, True, {"error": "Episode already done"}

        clean_action = self._sanitize_action(action)
        action_result = self._apply_action(clean_action)
        self._advance_incidents()
        self._spawn_scheduled_incidents(step_index=self.timestep)

        traffic_profile = self._current_traffic_profile()
        service_loads, uptime_ratio, p95_latency, cost_per_step = self._simulate_traffic(traffic_profile)
        self.cumulative_cost += cost_per_step
        self.uptime_history.append(uptime_ratio)
        self.latency_history.append(p95_latency)

        if self.human_attention_steps > 0:
            self.human_attention_steps -= 1
            if self.human_attention_steps == 0:
                self._resolve_highest_severity_incident(force=True)

        unresolved_critical = sum(1 for incident in self.active_incidents if not incident.resolved and incident.severity == "critical")
        self.sla_breaches += self._sla_breaches_this_step(p95_latency, traffic_profile)

        observation = self._get_observation(traffic_profile=traffic_profile, uptime=uptime_ratio, p95_latency=p95_latency)
        reward = self.reward_calculator.calculate(
            obs=observation,
            action=clean_action,
            resolved_incidents=self.resolved_incidents,
            unresolved_critical=unresolved_critical,
            total_incidents=self.total_incidents,
            cost_ratio=self.cumulative_cost / max(1e-6, self.task.cost_budget * max(1, self.task.max_steps)),
            uptime_ratio=uptime_ratio,
        )

        self.last_reward = reward
        self.last_action_result = action_result
        self.timestep += 1
        self.done = self.timestep >= self.task.max_steps or self.sla_breaches >= 10 or uptime_ratio < 0.25

        info = {
            "error": None,
            "action_result": action_result,
            "service_loads": service_loads,
            "resolved_incidents": self.resolved_incidents,
            "total_incidents": self.total_incidents,
        }
        return observation, reward, self.done, info

    def _sanitize_action(self, action: IncidentCommanderAction) -> IncidentCommanderAction:
        target_service = action.target_service if action.target_service in self.services else None
        fallback_service = action.fallback_service if action.fallback_service in self.services else None
        delta_instances = int(np.clip(action.delta_instances, -2, 3))
        request_fraction = float(np.clip(action.request_fraction, 0.0, 1.0))
        return IncidentCommanderAction(
            action_type=action.action_type,
            target_service=target_service,
            delta_instances=delta_instances,
            request_fraction=request_fraction,
            target_version=action.target_version,
            fallback_service=fallback_service,
            note=action.note,
        )

    def _apply_action(self, action: IncidentCommanderAction) -> str:
        if action.action_type == "noop":
            return "no operation"

        if action.action_type == "scale_service" and action.target_service:
            service = self.services[action.target_service]
            service.desired_instances = max(1, min(8, service.desired_instances + action.delta_instances))
            service.scale_cooldown_steps = 1
            if action.delta_instances > 0:
                service.instances = service.desired_instances
            else:
                service.instances = service.desired_instances
            return f"scaled {action.target_service} by {action.delta_instances}"

        if action.action_type == "rollback_deploy" and action.target_service:
            service = self.services[action.target_service]
            service.version = service.rollback_version
            service.rolling_back_steps = 2
            self._resolve_incidents_for_service(action.target_service, incident_type="bad_deploy")
            return f"rolled back {action.target_service}"

        if action.action_type == "quarantine_service" and action.target_service:
            service = self.services[action.target_service]
            service.quarantined = True
            service.healthy = False
            return f"quarantined {action.target_service}"

        if action.action_type == "reroute_traffic" and action.target_service and action.fallback_service:
            source = self.services[action.target_service]
            source.redirected_fraction = action.request_fraction
            return f"rerouted {action.target_service} to {action.fallback_service}"

        if action.action_type == "page_human":
            self.human_attention_steps = max(self.human_attention_steps, 2)
            return "human page acknowledged"

        return "action ignored"

    def _spawn_scheduled_incidents(self, step_index: int) -> None:
        if step_index not in self.task.incident_schedule:
            return

        schedule_index = self.task.incident_schedule.index(step_index)
        payloads = [
            ("gateway", "traffic_spike", "medium"),
            ("inference", "bad_deploy", "high"),
            ("database", "node_failure", "critical"),
            ("gateway", "cascade", "critical"),
            ("inference", "traffic_spike", "high"),
            ("database", "cache_thrash", "medium"),
        ]
        service, incident_type, severity = payloads[schedule_index % len(payloads)]
        incident = ActiveIncident(
            incident_id=f"{self.task.task_id}-{step_index}-{schedule_index}",
            incident_type=incident_type,
            service=service,
            severity=severity,
        )
        self.active_incidents.append(incident)
        self.total_incidents += 1

    def _current_traffic_profile(self) -> float:
        if self.task.task_id == "easy":
            return self.task.base_traffic + 0.03 * self.timestep
        if self.task.task_id == "medium":
            if self.timestep < 10:
                self.phase = "steady"
                return self.task.base_traffic
            if self.timestep < 22:
                self.phase = "spike"
                return self.task.peak_traffic
            self.phase = "recovery"
            return self.task.base_traffic * 0.95
        if self.task.task_id == "hard":
            if self.timestep < 8:
                self.phase = "incident"
                return self.task.base_traffic
            if self.timestep < 30:
                self.phase = "surge"
                return self.task.peak_traffic
            self.phase = "containment"
            return self.task.base_traffic * 0.9
        self.phase = "longhaul"
        if self.timestep < 15:
            return self.task.base_traffic * 0.95
        if self.timestep < 35:
            return self.task.peak_traffic
        if self.timestep < 50:
            return self.task.base_traffic * 1.1
        return self.task.base_traffic

    def _simulate_traffic(self, traffic_profile: float):
        service_loads: Dict[str, float] = defaultdict(float)
        base_weights = {"gateway": 0.25, "inference": 0.50, "database": 0.25}
        traffic_multiplier = 1.0 + 0.12 * len([i for i in self.active_incidents if not i.resolved])

        for name, service in self.services.items():
            incident_pressure = 0.0
            for incident in self.active_incidents:
                if incident.resolved or incident.service != name:
                    continue
                if incident.incident_type == "traffic_spike":
                    incident_pressure += 0.4 * self.task.spike_multiplier
                elif incident.incident_type == "bad_deploy":
                    incident_pressure += 0.35
                elif incident.incident_type == "node_failure":
                    incident_pressure += 0.55
                elif incident.incident_type == "database_lock":
                    incident_pressure += 0.3
                elif incident.incident_type == "cache_thrash":
                    incident_pressure += 0.25
                elif incident.incident_type == "cascade":
                    incident_pressure += 0.45

            redirected = 1.0 - service.redirected_fraction if service.redirected_fraction > 0 else 1.0
            demand = traffic_profile * base_weights[name] * traffic_multiplier
            demand *= (1.0 + incident_pressure) * redirected
            if service.quarantined:
                demand *= 0.15
            service_loads[name] = demand

        served_total = 0.0
        demand_total = 0.0
        latency_values: List[float] = []

        for name, service in self.services.items():
            demand = service_loads[name]
            demand_total += demand
            capacity = float(service.instances) * (1.0 if service.healthy else 0.45)
            if service.scale_cooldown_steps > 0:
                capacity *= 0.75
                service.scale_cooldown_steps -= 1
            if service.rolling_back_steps > 0:
                capacity *= 0.85
                service.rolling_back_steps -= 1

            if service.quarantined:
                capacity *= 0.2

            served = min(demand, capacity)
            served_total += served
            backlog = max(0.0, demand - served)
            service.queue_depth = int(round(backlog * 10))

            latency = 65.0 + 110.0 * (backlog / max(1.0, capacity)) + 15.0 * len([i for i in self.active_incidents if not i.resolved and i.service == name])
            if service.version != "v1":
                latency *= 0.88
            if service.quarantined:
                latency *= 1.1
            service.p95_latency = round(float(np.clip(latency, 20.0, 1000.0)), 2)
            service.error_rate = round(float(np.clip(backlog / max(1.0, demand + 1e-6), 0.0, 1.0)), 4)
            service.last_action_result = "serving" if backlog <= 0.15 else "degraded"
            latency_values.append(service.p95_latency)

        uptime_ratio = 1.0 if demand_total <= 0 else served_total / demand_total
        p95_latency = float(np.percentile(latency_values, 95)) if latency_values else 0.0
        cost_per_step = sum(0.03 * service.instances for service in self.services.values())
        cost_per_step += 0.04 * sum(1 for incident in self.active_incidents if not incident.resolved)
        cost_per_step += 0.03 * sum(1 for service in self.services.values() if service.quarantined)
        return service_loads, float(np.clip(uptime_ratio, 0.0, 1.0)), float(p95_latency), float(cost_per_step)

    def _sla_breaches_this_step(self, p95_latency: float, traffic_profile: float) -> int:
        breaches = 0
        if p95_latency > 280:
            breaches += 1
        if p95_latency > 420:
            breaches += 1
        if traffic_profile > self.task.base_traffic * 1.5 and p95_latency > 250:
            breaches += 1
        if any(not incident.resolved and incident.severity == "critical" and incident.age_steps >= 4 for incident in self.active_incidents):
            breaches += 1
        return breaches

    def _advance_incidents(self) -> None:
        for incident in self.active_incidents:
            if incident.resolved:
                continue
            incident.age_steps += 1
            if incident.resolution_timer > 0:
                incident.resolution_timer -= 1
                if incident.resolution_timer == 0:
                    incident.resolved = True
                    self.resolved_incidents += 1
                    self.services[incident.service].healthy = True
                    self.services[incident.service].quarantined = False
            elif incident.incident_type == "traffic_spike" and self.timestep > 0 and incident.age_steps >= 3:
                incident.resolved = True
                self.resolved_incidents += 1
            elif incident.incident_type == "bad_deploy" and self.services[incident.service].version == "v0":
                incident.resolved = True
                self.resolved_incidents += 1
                self.services[incident.service].healthy = True
            elif incident.incident_type == "node_failure" and self.services[incident.service].instances >= 3:
                incident.resolved = True
                self.resolved_incidents += 1
                self.services[incident.service].healthy = True
            elif incident.incident_type == "cache_thrash" and self.services[incident.service].instances >= 4:
                incident.resolved = True
                self.resolved_incidents += 1
            elif incident.incident_type == "cascade" and self.services[incident.service].quarantined:
                incident.resolved = True
                self.resolved_incidents += 1

        if self.timestep >= 2 and self.task.task_id in {"hard", "longhaul"}:
            unresolved = [i for i in self.active_incidents if not i.resolved]
            if len(unresolved) >= 2:
                self.active_incidents.append(
                    ActiveIncident(
                        incident_id=f"cascade-{self.task.task_id}-{self.timestep}",
                        incident_type="cascade",
                        service="gateway",
                        severity="critical",
                    )
                )
                self.total_incidents += 1

    def _resolve_incidents_for_service(self, service_name: str, incident_type: Optional[str] = None) -> None:
        for incident in self.active_incidents:
            if incident.resolved or incident.service != service_name:
                continue
            if incident_type is None or incident.incident_type == incident_type:
                incident.resolved = True
                self.resolved_incidents += 1
        self.services[service_name].healthy = True
        self.services[service_name].quarantined = False

    def _resolve_highest_severity_incident(self, force: bool = False) -> None:
        unresolved = [incident for incident in self.active_incidents if not incident.resolved]
        if not unresolved:
            return
        priority = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        target = max(unresolved, key=lambda item: priority[item.severity])
        target.resolved = True
        self.resolved_incidents += 1
        self.services[target.service].healthy = True
        self.services[target.service].quarantined = False
        if force:
            target.resolution_timer = 0

    def _get_observation(self, traffic_profile: float = 0.0, uptime: float = 1.0, p95_latency: float = 80.0) -> IncidentCommanderObservation:
        return IncidentCommanderObservation(
            services={name: service.model_copy(deep=True) for name, service in self.services.items()},
            active_incidents=[incident.model_copy(deep=True) for incident in self.active_incidents],
            step=self.timestep,
            step_budget=self.task.max_steps,
            traffic_level=round(traffic_profile, 4),
            uptime=round(uptime, 4),
            p95_latency=round(p95_latency, 4),
            sla_breaches=self.sla_breaches,
            cost_per_step=round(self.cumulative_cost, 4),
            last_action_result=self.last_action_result,
            phase=self.phase,
        )

    def get_state(self) -> Dict[str, object]:
        return {
            "task_id": self.task.task_id,
            "step": self.timestep,
            "services": {name: service.model_dump() for name, service in self.services.items()},
            "active_incidents": [incident.model_dump() for incident in self.active_incidents],
            "sla_breaches": self.sla_breaches,
            "resolved_incidents": self.resolved_incidents,
            "total_incidents": self.total_incidents,
            "cumulative_cost": round(self.cumulative_cost, 4),
            "phase": self.phase,
        }

    def get_metrics(self) -> Dict[str, float]:
        uptime_score = float(np.mean(self.uptime_history)) if self.uptime_history else 1.0
        avg_latency = float(np.mean(self.latency_history)) if self.latency_history else 80.0
        latency_score = max(0.0, min(1.0, 1.0 - avg_latency / 700.0))
        sla_score = max(0.0, min(1.0, 1.0 - self.sla_breaches / max(1, self.task.max_steps // 3)))
        cost_score = max(0.0, min(1.0, 1.0 - self.cumulative_cost / max(1e-6, self.task.cost_budget * self.task.max_steps)))
        recovery_score = max(0.0, min(1.0, self.resolved_incidents / max(1, self.total_incidents)))
        return {
            "uptime_score": round(uptime_score, 6),
            "avg_latency": round(avg_latency, 6),
            "latency_score": round(latency_score, 6),
            "sla_score": round(sla_score, 6),
            "cost_score": round(cost_score, 6),
            "recovery_score": round(recovery_score, 6),
        }

    def build_episode_result(self) -> EpisodeResult:
        metrics = self.get_metrics()
        total = (
            0.35 * metrics["uptime_score"]
            + 0.20 * metrics["latency_score"]
            + 0.20 * metrics["sla_score"]
            + 0.15 * metrics["cost_score"]
            + 0.10 * metrics["recovery_score"]
        )
        return EpisodeResult(
            task_id=self.task.task_id,
            uptime_score=metrics["uptime_score"],
            latency_score=metrics["latency_score"],
            sla_score=metrics["sla_score"],
            cost_score=metrics["cost_score"],
            recovery_score=metrics["recovery_score"],
            incident_clearance_rate=metrics["recovery_score"],
            ended_by_sla_failure=self.sla_breaches >= 10,
            total_score=round(max(0.0, min(1.0, total)), 6),
        )
