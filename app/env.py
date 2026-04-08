from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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
        peak_traffic=1.45,
        incident_schedule=[0],
        cost_budget=0.55,
        spike_multiplier=1.2,
        max_sla_breaches=10,
        observation_noise=0.02,
        spot_disruption_chance=0.0,
        memory_leak_rate=0.0,
        thundering_herd=False,
    ),
    "medium": TaskConfig(
        task_id="medium",
        max_steps=40,
        base_traffic=1.05,
        peak_traffic=1.8,
        incident_schedule=[0, 7],
        cost_budget=0.65,
        spike_multiplier=1.45,
        max_sla_breaches=9,
        observation_noise=0.04,
        spot_disruption_chance=0.01,
        memory_leak_rate=0.0,
        thundering_herd=False,
    ),
    "hard": TaskConfig(
        task_id="hard",
        max_steps=50,
        base_traffic=1.1,
        peak_traffic=2.1,
        incident_schedule=[0, 4, 10],
        cost_budget=0.72,
        spike_multiplier=1.65,
        max_sla_breaches=8,
        observation_noise=0.08,
        spot_disruption_chance=0.02,
        memory_leak_rate=0.0,
        thundering_herd=False,
    ),
    "longhaul": TaskConfig(
        task_id="longhaul",
        max_steps=60,
        base_traffic=1.0,
        peak_traffic=2.3,
        incident_schedule=[0, 8, 18, 30, 42, 50],
        cost_budget=0.78,
        spike_multiplier=1.85,
        max_sla_breaches=8,
        observation_noise=0.11,
        spot_disruption_chance=0.03,
        memory_leak_rate=0.05,
        thundering_herd=False,
    ),
    "blackout": TaskConfig(
        task_id="blackout",
        max_steps=70,
        base_traffic=1.15,
        peak_traffic=2.7,
        incident_schedule=[0, 5, 11, 17, 26, 34, 41, 50, 60],
        cost_budget=0.86,
        spike_multiplier=2.1,
        max_sla_breaches=7,
        observation_noise=0.14,
        spot_disruption_chance=0.05,
        memory_leak_rate=0.0,
        thundering_herd=True,
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
        self.unforced_pages = 0
        self.last_action_type = "noop"
        self.action_streak = 0
        self.ended_by_budget_failure = False
        self.action_counts: Dict[str, int] = {}

        self.reward_trace: List[Dict[str, object]] = []
        self.downtime_used = 0.0
        self.burn_budget_total = max(0.05, self.task.max_steps * 0.001)
        self.previous_traffic = self.task.base_traffic
        self.last_observed_metrics: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

        self.last_scale_target: Optional[str] = None
        self.last_scale_direction = 0

        self._init_state()

    def _init_state(self) -> None:
        self.services = {
            "gateway": ServiceState(name="gateway", instances=3, desired_instances=3, version="v1", rollback_version="v0"),
            "inference": ServiceState(name="inference", instances=4, desired_instances=4, version="v1", rollback_version="v0"),
            "database": ServiceState(name="database", instances=2, desired_instances=2, version="v1", rollback_version="v0"),
        }

        if self.task.task_id in {"hard", "longhaul", "blackout"}:
            self.services["gateway"].spot_instances = 1
            self.services["inference"].spot_instances = 1
            self.services["database"].spot_instances = 0

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
        self.unforced_pages = 0
        self.last_action_type = "noop"
        self.action_streak = 0
        self.ended_by_budget_failure = False
        self.action_counts = {
            "scale_service": 0,
            "reroute_traffic": 0,
            "rollback_deploy": 0,
            "quarantine_service": 0,
            "page_human": 0,
            "noop": 0,
        }

        self.reward_trace = []
        self.downtime_used = 0.0
        self.previous_traffic = self.task.base_traffic
        self.last_observed_metrics = {}
        self.last_scale_target = None
        self.last_scale_direction = 0

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
        contradictory_action = self._is_contradictory_scaling(clean_action)

        unresolved_high_or_critical = sum(
            1
            for incident in self.active_incidents
            if not incident.resolved and incident.severity in {"high", "critical"}
        )
        unforced_page = clean_action.action_type == "page_human" and unresolved_high_or_critical == 0
        if unforced_page:
            self.unforced_pages += 1

        if clean_action.action_type == self.last_action_type:
            self.action_streak += 1
        else:
            self.action_streak = 1
            self.last_action_type = clean_action.action_type

        if clean_action.action_type in self.action_counts:
            self.action_counts[clean_action.action_type] += 1

        action_result = self._apply_action(clean_action)
        resolved_ages = self._advance_incidents()

        self._spawn_scheduled_incidents(step_index=self.timestep)

        traffic_profile = self._current_traffic_profile()
        service_loads, uptime_ratio, p95_latency, cost_per_step = self._simulate_traffic(traffic_profile)

        self.previous_traffic = traffic_profile
        self.cumulative_cost += cost_per_step
        self.uptime_history.append(uptime_ratio)
        self.latency_history.append(p95_latency)
        self.downtime_used += max(0.0, 1.0 - uptime_ratio)

        if self.human_attention_steps > 0:
            self.human_attention_steps -= 1
            if self.human_attention_steps == 0:
                self._resolve_highest_severity_incident(force=True)

        unresolved_critical = sum(1 for incident in self.active_incidents if not incident.resolved and incident.severity == "critical")
        self.sla_breaches += self._sla_breaches_this_step(p95_latency, traffic_profile)

        observation = self._get_observation(traffic_profile=traffic_profile, uptime=uptime_ratio, p95_latency=p95_latency)
        budget_total = max(1e-6, self.task.cost_budget * max(1, self.task.max_steps))
        budget_ratio = self.cumulative_cost / budget_total
        budget_exhausted = budget_ratio > 1.35
        if budget_exhausted:
            self.ended_by_budget_failure = True

        burn_budget_ratio = self.downtime_used / max(1e-6, self.burn_budget_total)

        reward = self.reward_calculator.calculate(
            obs=observation,
            action=clean_action,
            resolved_incidents=self.resolved_incidents,
            unresolved_critical=unresolved_critical,
            total_incidents=self.total_incidents,
            cost_ratio=budget_ratio,
            uptime_ratio=uptime_ratio,
            unforced_page=unforced_page,
            action_streak=self.action_streak,
            mttr_resolved_ages=resolved_ages,
            burn_budget_ratio=burn_budget_ratio,
            contradictory_action=contradictory_action,
        )

        self.last_reward = reward
        self.last_action_result = action_result

        self.reward_trace.append(
            {
                "step": self.timestep,
                "phase": self.phase,
                "action": clean_action.action_type,
                "action_result": action_result,
                "traffic_level": round(traffic_profile, 4),
                "uptime": round(uptime_ratio, 4),
                "p95_latency": round(p95_latency, 4),
                "cost_per_step": round(cost_per_step, 5),
                "sla_breaches": self.sla_breaches,
                "reward": reward.model_dump(),
                "contradictory_action": contradictory_action,
                "burn_budget_ratio": round(burn_budget_ratio, 5),
            }
        )

        self.timestep += 1
        self.done = (
            self.timestep >= self.task.max_steps
            or self.sla_breaches >= self.task.max_sla_breaches
            or uptime_ratio < 0.22
            or budget_exhausted
        )

        info = {
            "error": None,
            "action_result": action_result,
            "service_loads": service_loads,
            "resolved_incidents": self.resolved_incidents,
            "total_incidents": self.total_incidents,
            "budget_ratio": round(budget_ratio, 6),
            "budget_exhausted": budget_exhausted,
            "unforced_pages": self.unforced_pages,
            "burn_budget_ratio": round(burn_budget_ratio, 6),
            "contradictory_action": contradictory_action,
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

    def _is_contradictory_scaling(self, action: IncidentCommanderAction) -> bool:
        contradictory = False
        if action.action_type == "scale_service" and action.target_service and action.delta_instances != 0:
            direction = 1 if action.delta_instances > 0 else -1
            contradictory = (
                self.last_scale_target == action.target_service
                and self.last_scale_direction != 0
                and direction != self.last_scale_direction
            )
            self.last_scale_target = action.target_service
            self.last_scale_direction = direction
        return contradictory

    def _apply_action(self, action: IncidentCommanderAction) -> str:
        if action.action_type == "noop":
            return "no operation"

        if action.action_type == "scale_service" and action.target_service:
            service = self.services[action.target_service]
            service.desired_instances = max(1, min(8, service.desired_instances + action.delta_instances))
            service.scale_cooldown_steps = 1
            service.instances = service.desired_instances

            if action.delta_instances > 0 and self.task.task_id in {"longhaul", "blackout"}:
                spot_add = int(round(action.delta_instances * 0.5))
                service.spot_instances = max(0, min(service.instances, service.spot_instances + spot_add))
            if action.delta_instances < 0:
                service.spot_instances = min(service.spot_instances, service.instances)

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
            self.phase = "steady"
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

        if self.task.task_id == "longhaul":
            if self.timestep < 16:
                self.phase = "silent-leak"
                return self.task.base_traffic * 0.9
            if self.timestep < 38:
                self.phase = "surge"
                return self.task.peak_traffic
            self.phase = "stabilization"
            return self.task.base_traffic * 1.08

        if self.timestep < 12:
            self.phase = "brownout"
            return self.task.base_traffic * 1.15
        if self.timestep < 36:
            self.phase = "thundering-herd"
            return self.task.peak_traffic
        if self.timestep < 56:
            self.phase = "rollback-window"
            return self.task.base_traffic * 1.45
        self.phase = "stabilization"
        return self.task.base_traffic * 1.05

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
                    incident_pressure += 0.35
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
            self._apply_spot_disruption(name, service)

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

            utilization = demand / max(1.0, capacity)
            base_latency = 65.0 + 105.0 * utilization
            latency = base_latency + 13.0 * len([i for i in self.active_incidents if not i.resolved and i.service == name])

            if service.version != "v1":
                latency *= 0.88
            if service.quarantined:
                latency *= 1.1

            service.cpu_utilization = float(np.clip(35.0 + 60.0 * utilization, 5.0, 100.0))
            service.memory_utilization = float(np.clip(service.memory_utilization + 3.0 * utilization, 10.0, 99.0))

            if self.task.task_id == "longhaul" and name == "inference" and self.phase == "silent-leak":
                service.memory_utilization = float(np.clip(service.memory_utilization + 100.0 * self.task.memory_leak_rate, 10.0, 100.0))
                # Keep silent-leak progression deterministic so the delayed-credit pattern is auditable.
                service.memory_utilization = float(max(service.memory_utilization, 72.0 + float(self.timestep)))

            if service.memory_utilization > 96.0 and service.healthy:
                service.healthy = False
                self._emit_incident_once(name, "node_failure", "high")
                service.last_action_result = "memory leak crash"

            service.p95_latency = round(float(np.clip(latency, 20.0, 1000.0)), 2)
            service.error_rate = round(float(np.clip(backlog / max(1.0, demand + 1e-6), 0.0, 1.0)), 4)
            service.last_action_result = "serving" if backlog <= 0.15 else service.last_action_result or "degraded"
            latency_values.append(service.p95_latency)

        self._apply_dependency_hell()
        self._apply_longhaul_failure_window()
        self._apply_thundering_herd_rules(traffic_profile)

        uptime_ratio = 1.0 if demand_total <= 0 else served_total / demand_total
        p95_latency = float(np.percentile(latency_values, 95)) if latency_values else 0.0

        cost_per_step = 0.0
        for service in self.services.values():
            on_demand = max(0, service.instances - service.spot_instances)
            cost_per_step += 0.03 * on_demand
            cost_per_step += 0.006 * service.spot_instances
        cost_per_step += 0.04 * sum(1 for incident in self.active_incidents if not incident.resolved)
        cost_per_step += 0.03 * sum(1 for service in self.services.values() if service.quarantined)

        return service_loads, float(np.clip(uptime_ratio, 0.0, 1.0)), float(p95_latency), float(cost_per_step)

    def _apply_dependency_hell(self) -> None:
        database = self.services["database"]
        gateway = self.services["gateway"]

        if database.p95_latency > 200:
            overload = (database.p95_latency - 200.0) / 220.0
            gateway.cpu_utilization = float(np.clip(gateway.cpu_utilization + 40.0 * overload, 0.0, 100.0))
            gateway.p95_latency = round(float(np.clip(gateway.p95_latency * (1.0 + 0.45 * overload), 20.0, 1000.0)), 2)
            gateway.error_rate = round(float(np.clip(gateway.error_rate + 0.12 * overload, 0.0, 1.0)), 4)
            gateway.last_action_result = "db-latency backpressure"

    def _apply_spot_disruption(self, name: str, service: ServiceState) -> None:
        if service.spot_instances <= 0 or self.task.spot_disruption_chance <= 0.0:
            return

        lost = 0
        for _ in range(service.spot_instances):
            if self.rng.random() < self.task.spot_disruption_chance:
                lost += 1

        if lost <= 0:
            return

        service.instances = max(1, service.instances - lost)
        service.desired_instances = max(service.instances, service.desired_instances - lost)
        service.spot_instances = max(0, service.spot_instances - lost)
        self._emit_incident_once(name, "node_failure", "high")
        service.last_action_result = f"spot interruption lost={lost}"

    def _apply_thundering_herd_rules(self, traffic_profile: float) -> None:
        if not self.task.thundering_herd:
            return

        traffic_roc = traffic_profile - self.previous_traffic
        inference = self.services["inference"]

        if self.phase == "thundering-herd" and traffic_roc > 0.18 and inference.instances <= 5:
            self._emit_incident_once("database", "database_lock", "critical")
            self.services["database"].last_action_result = "lock contention from under-scaling"

        if self.phase == "thundering-herd" and inference.instances >= 8:
            self.cumulative_cost += 0.08

    def _apply_longhaul_failure_window(self) -> None:
        if self.task.task_id != "longhaul" or self.phase != "surge":
            return

        inference = self.services["inference"]
        # If inference is still underscaled when surge begins after leak accumulation,
        # force node-failure pressure that proactive scaling would have avoided.
        if inference.instances <= 4 and inference.memory_utilization >= 88.0:
            self._emit_incident_once("inference", "node_failure", "critical")
            inference.p95_latency = round(float(np.clip(inference.p95_latency * 1.35, 20.0, 1000.0)), 2)
            inference.error_rate = round(float(np.clip(inference.error_rate + 0.22, 0.0, 1.0)), 4)
            inference.healthy = False
            inference.last_action_result = "surge collapse after silent leak"

    def _emit_incident_once(self, service: str, incident_type: str, severity: str) -> None:
        for incident in self.active_incidents:
            if not incident.resolved and incident.service == service and incident.incident_type == incident_type:
                return

        self.active_incidents.append(
            ActiveIncident(
                incident_id=f"auto-{self.task.task_id}-{self.timestep}-{service}-{incident_type}",
                incident_type=incident_type,
                service=service,
                severity=severity,
            )
        )
        self.total_incidents += 1

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

    def _advance_incidents(self) -> List[int]:
        resolved_ages: List[int] = []
        for incident in self.active_incidents:
            if incident.resolved:
                continue
            incident.age_steps += 1
            if incident.resolution_timer > 0:
                incident.resolution_timer -= 1
                if incident.resolution_timer == 0:
                    incident.resolved = True
                    resolved_ages.append(incident.age_steps)
                    self.resolved_incidents += 1
                    self.services[incident.service].healthy = True
                    self.services[incident.service].quarantined = False
            elif incident.incident_type == "traffic_spike" and self.timestep > 0 and incident.age_steps >= 3:
                incident.resolved = True
                resolved_ages.append(incident.age_steps)
                self.resolved_incidents += 1
            elif incident.incident_type == "bad_deploy" and self.services[incident.service].version == "v0":
                incident.resolved = True
                resolved_ages.append(incident.age_steps)
                self.resolved_incidents += 1
                self.services[incident.service].healthy = True
            elif incident.incident_type == "node_failure" and self.services[incident.service].instances >= 3:
                incident.resolved = True
                resolved_ages.append(incident.age_steps)
                self.resolved_incidents += 1
                self.services[incident.service].healthy = True
            elif incident.incident_type == "cache_thrash" and self.services[incident.service].instances >= 4:
                incident.resolved = True
                resolved_ages.append(incident.age_steps)
                self.resolved_incidents += 1
            elif incident.incident_type == "cascade" and self.services[incident.service].quarantined:
                incident.resolved = True
                resolved_ages.append(incident.age_steps)
                self.resolved_incidents += 1

        if self.timestep >= 2 and self.task.task_id in {"hard", "longhaul", "blackout"}:
            unresolved = [i for i in self.active_incidents if not i.resolved]
            required_unresolved = 2 if self.task.task_id != "blackout" else 3
            if len(unresolved) >= required_unresolved:
                self._emit_incident_once("gateway", "cascade", "critical")

        return resolved_ages

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

    def _apply_observation_noise(self, services: Dict[str, ServiceState]) -> None:
        if self.timestep <= 0 or self.task.observation_noise <= 0.0:
            for name, service in services.items():
                service.observed_p95_latency = service.p95_latency
                service.observed_error_rate = service.error_rate
                service.metric_staleness_steps = 0
                self.last_observed_metrics[name] = (service.p95_latency, service.error_rate)
            return

        for name, service in services.items():
            delayed = self.rng.random() < self.task.observation_noise
            missing = self.rng.random() < (self.task.observation_noise * 0.35)

            prev_latency, prev_error = self.last_observed_metrics.get(name, (service.p95_latency, service.error_rate))
            if missing:
                service.observed_p95_latency = None
                service.observed_error_rate = prev_error
                service.metric_staleness_steps += 1
            elif delayed:
                service.observed_p95_latency = prev_latency
                service.observed_error_rate = prev_error
                service.metric_staleness_steps = max(1, service.metric_staleness_steps + 1)
            else:
                service.observed_p95_latency = service.p95_latency
                service.observed_error_rate = service.error_rate
                service.metric_staleness_steps = 0

            self.last_observed_metrics[name] = (service.observed_p95_latency, service.observed_error_rate)

    def _get_observation(self, traffic_profile: float = 0.0, uptime: float = 1.0, p95_latency: float = 80.0) -> IncidentCommanderObservation:
        services = {name: service.model_copy(deep=True) for name, service in self.services.items()}
        self._apply_observation_noise(services)

        return IncidentCommanderObservation(
            services=services,
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
            "burn_budget_ratio": round(self.downtime_used / max(1e-6, self.burn_budget_total), 6),
        }

    def get_metrics(self) -> Dict[str, object]:
        uptime_score = float(np.mean(self.uptime_history)) if self.uptime_history else 1.0
        avg_latency = float(np.mean(self.latency_history)) if self.latency_history else 80.0
        latency_score = max(0.0, min(1.0, 1.0 - avg_latency / 700.0))
        strict_sla_window = self.task.max_steps // 8 if self.task.task_id in {"hard", "longhaul", "blackout"} else self.task.max_steps // 4
        sla_score = max(0.0, min(1.0, 1.0 - self.sla_breaches / max(1, strict_sla_window)))
        budget_scale = 0.85 if self.task.task_id in {"hard", "longhaul", "blackout"} else 1.0
        cost_score = max(0.0, min(1.0, 1.0 - self.cumulative_cost / max(1e-6, self.task.cost_budget * self.task.max_steps * budget_scale)))
        recovery_score = max(0.0, min(1.0, self.resolved_incidents / max(1, self.total_incidents)))
        action_total = max(1, self.timestep)
        passive_ratio = (self.action_counts["noop"] + self.unforced_pages) / action_total
        action_discipline_score = max(0.0, min(1.0, 1.0 - passive_ratio))

        return {
            "uptime_score": round(uptime_score, 6),
            "avg_latency": round(avg_latency, 6),
            "latency_score": round(latency_score, 6),
            "sla_score": round(sla_score, 6),
            "cost_score": round(cost_score, 6),
            "recovery_score": round(recovery_score, 6),
            "action_discipline_score": round(action_discipline_score, 6),
            "burn_budget_ratio": round(self.downtime_used / max(1e-6, self.burn_budget_total), 6),
            "reward_trace": self.reward_trace,
        }

    def build_episode_result(self) -> EpisodeResult:
        metrics = self.get_metrics()
        total = (
            0.35 * float(metrics["uptime_score"])
            + 0.20 * float(metrics["latency_score"])
            + 0.20 * float(metrics["sla_score"])
            + 0.15 * float(metrics["cost_score"])
            + 0.10 * float(metrics["recovery_score"])
        )
        return EpisodeResult(
            task_id=self.task.task_id,
            uptime_score=float(metrics["uptime_score"]),
            latency_score=float(metrics["latency_score"]),
            sla_score=float(metrics["sla_score"]),
            cost_score=float(metrics["cost_score"]),
            recovery_score=float(metrics["recovery_score"]),
            incident_clearance_rate=float(metrics["recovery_score"]),
            ended_by_sla_failure=self.sla_breaches >= self.task.max_sla_breaches,
            ended_by_budget_failure=self.ended_by_budget_failure,
            action_discipline_score=float(metrics["action_discipline_score"]),
            escalations_used=self.action_counts["page_human"],
            sla_breaches=self.sla_breaches,
            burn_budget_ratio=float(metrics["burn_budget_ratio"]),
            total_score=round(max(0.0, min(1.0, total)), 6),
        )
