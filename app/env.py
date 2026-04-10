from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .models import (
    ActiveIncident,
    EmergencyUnitState,
    EpisodeResult,
    IncidentCommanderAction,
    IncidentCommanderObservation,
    IncidentCommanderReward,
    ScenarioState,
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
        max_sla_breaches=10,
        observation_noise=0.02,
        adversarial_shift_rate=0.01,
        scenario_mix=["resource_exhaustion"],
    ),
    "medium": TaskConfig(
        task_id="medium",
        max_steps=40,
        base_traffic=1.05,
        peak_traffic=1.8,
        incident_schedule=[0, 8],
        cost_budget=0.64,
        max_sla_breaches=9,
        observation_noise=0.05,
        adversarial_shift_rate=0.02,
        scenario_mix=["resource_exhaustion", "config_drift"],
    ),
    "hard": TaskConfig(
        task_id="hard",
        max_steps=50,
        base_traffic=1.1,
        peak_traffic=2.1,
        incident_schedule=[0, 10, 20],
        cost_budget=0.72,
        max_sla_breaches=8,
        observation_noise=0.08,
        adversarial_shift_rate=0.035,
        scenario_mix=["config_drift", "heisenbug", "regional_outage"],
    ),
    "longhaul": TaskConfig(
        task_id="longhaul",
        max_steps=60,
        base_traffic=1.0,
        peak_traffic=2.3,
        incident_schedule=[0, 10, 22, 34, 48],
        cost_budget=0.8,
        max_sla_breaches=8,
        observation_noise=0.1,
        adversarial_shift_rate=0.05,
        scenario_mix=["resource_exhaustion", "config_drift", "heisenbug", "regional_outage"],
    ),
    "blackout": TaskConfig(
        task_id="blackout",
        max_steps=70,
        base_traffic=1.15,
        peak_traffic=2.7,
        incident_schedule=[0, 6, 14, 24, 36, 50, 62],
        cost_budget=0.88,
        max_sla_breaches=7,
        observation_noise=0.14,
        adversarial_shift_rate=0.07,
        thundering_herd=True,
        scenario_mix=["resource_exhaustion", "config_drift", "heisenbug", "regional_outage"],
    ),
}


class IncidentCommanderEnv:
    def __init__(self, task: TaskConfig, seed: int = 42):
        self.task = task
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.reward_calculator = RewardCalculator(task)

        self.services: Dict[str, ServiceState] = {}
        self.emergency_units: Dict[str, EmergencyUnitState] = {}
        self.active_incidents: List[ActiveIncident] = []
        self.scenario = ScenarioState(scenario_id="resource_exhaustion")

        self.timestep = 0
        self.done = False
        self.last_reward = IncidentCommanderReward(total=0.0, uptime=0.0, latency=0.0, sla=0.0, cost=0.0, recovery=0.0)
        self.last_action_result = "initialized"
        self.phase = "steady"

        self.sla_breaches = 0
        self.total_incidents = 0
        self.resolved_incidents = 0
        self.cumulative_cost = 0.0
        self.downtime_used = 0.0
        self.burn_budget_total = max(0.05, self.task.max_steps * 0.002)

        self.uptime_history: List[float] = []
        self.latency_history: List[float] = []
        self.reward_trace: List[Dict[str, object]] = []
        self.live_timeline: List[str] = []

        self.investigation_log: List[str] = []
        self.investigation_targets: Set[str] = set()
        self.verification_successes = 0
        self.incorrect_actions = 0
        self.unsafe_actions = 0
        self.root_cause_resolutions = 0

        self.tmp_files: List[str] = []
        self.config_default: Dict[str, str] = {
            "db_host": "db",
            "db_port": "5432",
            "auth_mode": "strict",
            "feature_flag_race_guard": "false",
        }
        self.config_broken: Dict[str, str] = dict(self.config_default)
        self.config_runtime: Dict[str, str] = {}

        self.dev_hint_cooldown = 0
        self.load_test_active_steps = 0
        self.open_sockets = 0
        self.declared_emergency = False
        self.civilian_risk = 0.22
        self.incident_severity = 0.30
        self.incident_type = "infra_outage"
        self.weather_condition = "clear"
        self.region_status: Dict[str, float] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        self.link_outage_ratio = 0.0
        self.civilians_saved = 0
        self.wrong_dispatches = 0
        self.delayed_response_steps = 0
        self.recent_dispatch_effect = 0.0
        self.commitment_mode = "adaptive"
        self.last_strategy_level: Optional[str] = None
        self.commitment_switches = 0
        self.institutional_trust = 0.78
        self.economic_stability = 0.82
        self.legal_risk = 0.14
        self.misinformation_index = 0.20

        self.last_observed_metrics: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

        self.action_counts: Dict[str, int] = defaultdict(int)
        self.last_action_type = "noop"
        self.action_streak = 0
        self.action_history: List[Tuple[int, str, Optional[str]]] = []
        self.ended_by_budget_failure = False

        self._init_state()

    def _init_state(self) -> None:
        self.services = {
            "frontend": ServiceState(name="frontend", instances=3, desired_instances=3, version="v1", rollback_version="v0"),
            "auth": ServiceState(name="auth", instances=3, desired_instances=3, version="v1", rollback_version="v0"),
            "db": ServiceState(name="db", instances=2, desired_instances=2, version="v1", rollback_version="v0"),
        }
        self.services["frontend"].reachable_upstreams = ["auth"]
        self.services["auth"].reachable_upstreams = ["db"]
        self.services["db"].reachable_upstreams = []
        self.dependency_graph = {
            "frontend": ["auth"],
            "auth": ["db"],
            "db": [],
        }
        self.region_status = {
            "zone_a": 1.0,
            "zone_b": 1.0,
            "zone_c": 1.0,
        }
        self.link_outage_ratio = 0.0

        self.active_incidents = []
        self.emergency_units = {
            "fire": EmergencyUnitState(unit_type="fire", available=2, deployed=0),
            "police": EmergencyUnitState(unit_type="police", available=2, deployed=0),
            "medical": EmergencyUnitState(unit_type="medical", available=2, deployed=0),
            "drone": EmergencyUnitState(unit_type="drone", available=3, deployed=0),
            "evacuation": EmergencyUnitState(unit_type="evacuation", available=2, deployed=0),
        }
        self.timestep = 0
        self.done = False
        self.phase = "steady"
        self.last_action_result = "reset complete"

        self.sla_breaches = 0
        self.total_incidents = 0
        self.resolved_incidents = 0
        self.cumulative_cost = 0.0
        self.downtime_used = 0.0

        self.uptime_history = []
        self.latency_history = []
        self.reward_trace = []
        self.live_timeline = []

        self.investigation_log = []
        self.investigation_targets = set()
        self.verification_successes = 0
        self.incorrect_actions = 0
        self.unsafe_actions = 0
        self.root_cause_resolutions = 0

        self.tmp_files = []
        self.config_broken = dict(self.config_default)
        self.config_runtime = dict(self.config_broken)

        self.dev_hint_cooldown = 0
        self.load_test_active_steps = 0
        self.open_sockets = 0
        self.declared_emergency = False
        self.civilian_risk = 0.22
        self.incident_severity = 0.30
        self.incident_type = "infra_outage"
        self.weather_condition = "clear"
        self.commitment_mode = "adaptive"
        self.last_strategy_level = None
        self.commitment_switches = 0
        self.institutional_trust = 0.78
        self.economic_stability = 0.82
        self.legal_risk = 0.14
        self.misinformation_index = 0.20
        self.civilians_saved = 0
        self.wrong_dispatches = 0
        self.delayed_response_steps = 0
        self.recent_dispatch_effect = 0.0
        self.last_observed_metrics = {}

        self.action_counts = defaultdict(int)
        self.last_action_type = "noop"
        self.action_streak = 0
        self.action_history = []
        self.ended_by_budget_failure = False

        scenario_id = self._select_scenario()
        self.scenario = self._build_scenario(scenario_id)
        self._apply_scenario_bootstrap()

    def reset(self, seed: Optional[int] = None) -> IncidentCommanderObservation:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.reward_calculator = RewardCalculator(self.task)
        self._init_state()
        self._append_timeline("Environment reset; ephemeral filesystem restored")
        self._append_timeline(f"Scenario seeded: {self.scenario.scenario_id}")
        self._append_timeline("Emergency units standing by: fire, police, medical, drone, evacuation")
        return self._get_observation()

    def step(self, action: IncidentCommanderAction):
        if self.done:
            return self._get_observation(), self.last_reward, True, {"error": "Episode already done"}

        clean_action = self._sanitize_action(action)
        action_level = clean_action.strategy_level
        if action_level is None:
            if clean_action.action_type in {"declare_emergency", "allocate_resources", "request_national_support"}:
                action_level = "strategic"
            elif clean_action.action_type in {"dispatch_fire_truck", "send_medical_team", "deploy_drone_scan", "evacuate_zone", "request_backup"}:
                action_level = "tactical"
        if action_level is not None and self.last_strategy_level is not None and action_level != self.last_strategy_level:
            self.commitment_switches += 1
        if action_level is not None:
            self.last_strategy_level = action_level
            self.commitment_mode = action_level

        unresolved_critical_before = sum(1 for i in self.active_incidents if not i.resolved and i.severity == "critical")
        self.action_counts[clean_action.action_type] += 1

        if clean_action.action_type == self.last_action_type:
            self.action_streak += 1
        else:
            self.action_streak = 1
            self.last_action_type = clean_action.action_type

        incorrect_action = False
        unsafe_action = self._is_unsafe_action(clean_action)
        if unsafe_action:
            self.unsafe_actions += 1

        action_output, action_effective, incorrect_action = self._apply_action(clean_action)
        if incorrect_action:
            self.incorrect_actions += 1

        self._inject_scheduled_incident_event()

        self._simulate_scenario_drift()
        self._simulate_disaster_progression()

        traffic_profile = self._current_traffic_profile()
        service_loads, uptime_ratio, p95_latency, cost_per_step = self._simulate_traffic(traffic_profile)

        self.cumulative_cost += cost_per_step
        self.uptime_history.append(uptime_ratio)
        self.latency_history.append(p95_latency)
        self.downtime_used += max(0.0, 1.0 - uptime_ratio)

        self._update_incidents()
        unresolved_critical = sum(1 for i in self.active_incidents if not i.resolved and i.severity == "critical")
        if unresolved_critical > 0:
            self.delayed_response_steps += 1
        contradictory_action = self._is_contradictory_action(clean_action, incorrect_action, unresolved_critical_before)

        self.sla_breaches += self._sla_breaches_this_step(p95_latency, traffic_profile)

        if self.load_test_active_steps > 0:
            self.load_test_active_steps -= 1

        if self.dev_hint_cooldown > 0:
            self.dev_hint_cooldown -= 1

        observation = self._get_observation(traffic_profile=traffic_profile, uptime=uptime_ratio, p95_latency=p95_latency)

        budget_total = max(1e-6, self.task.cost_budget * max(1, self.task.max_steps))
        budget_ratio = self.cumulative_cost / budget_total
        if budget_ratio > 1.35:
            self.ended_by_budget_failure = True

        burn_budget_ratio = self.downtime_used / max(1e-6, self.burn_budget_total)
        investigation_coverage = len(self.investigation_targets) / 3.0
        unforced_page = self._is_unforced_page(clean_action, unresolved_critical, traffic_profile, p95_latency)

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
            mttr_resolved_ages=[incident.age_steps for incident in self.active_incidents if incident.resolved],
            burn_budget_ratio=burn_budget_ratio,
            contradictory_action=contradictory_action,
            incorrect_action=incorrect_action,
            unsafe_action=unsafe_action,
            investigation_coverage=investigation_coverage,
            steps_taken=self.timestep + 1,
            ideal_steps=3,
            civilians_saved=self.civilians_saved,
            civilian_risk=self.civilian_risk,
            delayed_response_steps=self.delayed_response_steps,
            wrong_dispatches=self.wrong_dispatches,
            commitment_switches=self.commitment_switches,
            graph_outage_ratio=self.link_outage_ratio,
            institutional_trust=self.institutional_trust,
            legal_risk=self.legal_risk,
            economic_stability=self.economic_stability,
            misinformation_index=self.misinformation_index,
        )

        self.last_reward = reward
        self.last_action_result = action_output

        self.reward_trace.append(
            {
                "step": self.timestep,
                "phase": self.phase,
                "scenario": self.scenario.scenario_id,
                "action": clean_action.action_type,
                "effective_action": action_effective,
                "action_result": action_output,
                "traffic_level": round(traffic_profile, 4),
                "uptime": round(uptime_ratio, 4),
                "p95_latency": round(p95_latency, 4),
                "cost_per_step": round(cost_per_step, 5),
                "sla_breaches": self.sla_breaches,
                "reward": reward.model_dump(),
                "investigation_coverage": round(investigation_coverage, 4),
                "civilian_risk": round(self.civilian_risk, 4),
                "incident_severity": round(self.incident_severity, 4),
                "civilians_saved": self.civilians_saved,
                "institutional_trust": round(self.institutional_trust, 4),
                "economic_stability": round(self.economic_stability, 4),
                "legal_risk": round(self.legal_risk, 4),
                "misinformation_index": round(self.misinformation_index, 4),
            }
        )
        self.action_history.append((self.timestep, clean_action.action_type, clean_action.target_service))
        if len(self.action_history) > 30:
            self.action_history = self.action_history[-30:]

        self.timestep += 1
        self.done = (
            self.timestep >= self.task.max_steps
            or self.sla_breaches >= self.task.max_sla_breaches
            or uptime_ratio < 0.22
            or self.ended_by_budget_failure
            or self._all_services_recovered()
        )

        if self.done:
            # Ensure the environment leaves no simulated sockets dangling.
            self.open_sockets = 0

        info = {
            "error": None,
            "action_result": action_output,
            "service_loads": service_loads,
            "resolved_incidents": self.resolved_incidents,
            "total_incidents": self.total_incidents,
            "budget_ratio": round(budget_ratio, 6),
            "burn_budget_ratio": round(burn_budget_ratio, 6),
            "open_sockets": self.open_sockets,
            "scenario": self.scenario.scenario_id,
        }
        return observation, reward, self.done, info

    def _sanitize_action(self, action: IncidentCommanderAction) -> IncidentCommanderAction:
        target_service = action.target_service if action.target_service in self.services else None
        fallback_service = action.fallback_service if action.fallback_service in self.services else None
        return IncidentCommanderAction(
            action_type=action.action_type,
            target_service=target_service,
            delta_instances=int(np.clip(action.delta_instances, -2, 3)),
            request_fraction=float(np.clip(action.request_fraction, 0.0, 1.0)),
            target_version=action.target_version,
            fallback_service=fallback_service,
            n_lines=int(np.clip(action.n_lines, 1, 100)),
            config_key=action.config_key,
            config_value=action.config_value,
            question=action.question,
            command=action.command,
            note=action.note,
            unit_type=action.unit_type,
            target_zone=action.target_zone,
            priority=action.priority,
            strategy_level=action.strategy_level,
        )

    def _is_unsafe_action(self, action: IncidentCommanderAction) -> bool:
        if action.action_type != "run_command":
            return False
        if not action.command:
            return False
        dangerous = ["rm -rf", "del /s", "format ", "mkfs", "dd if="]
        cmd = action.command.lower()
        return any(token in cmd for token in dangerous) and len(self.investigation_targets) == 0

    def _apply_action(self, action: IncidentCommanderAction) -> Tuple[str, str, bool]:
        mapped = self._map_legacy_action(action)

        if mapped.action_type == "declare_emergency":
            self.declared_emergency = True
            self.incident_severity = float(np.clip(self.incident_severity + 0.04, 0.0, 1.0))
            self._append_timeline("Strategic action: emergency state declared")
            return "emergency declared and command mode enabled", "declare_emergency", False

        if mapped.action_type == "allocate_resources":
            self.recent_dispatch_effect += 0.05
            self._append_timeline("Strategic action: cross-service resource allocation applied")
            return "resource allocation rebalanced", "allocate_resources", False

        if mapped.action_type == "request_national_support":
            for unit in self.emergency_units.values():
                unit.available += 1
            self.recent_dispatch_effect += 0.08
            self.legal_risk = float(np.clip(self.legal_risk + 0.02, 0.0, 1.0))
            self.institutional_trust = float(np.clip(self.institutional_trust + 0.02, 0.0, 1.0))
            self._append_timeline("Strategic action: external support requested")
            return "national support requested; reserve units added", "request_national_support", False

        if mapped.action_type == "issue_public_briefing":
            self.institutional_trust = float(np.clip(self.institutional_trust + 0.07, 0.0, 1.0))
            self.misinformation_index = float(np.clip(self.misinformation_index - 0.10, 0.0, 1.0))
            self._append_timeline("Strategic action: public briefing delivered with transparent metrics")
            return "public briefing issued", "issue_public_briefing", False

        if mapped.action_type == "impose_restriction_order":
            self.incident_severity = float(np.clip(self.incident_severity - 0.09, 0.0, 1.0))
            self.civilian_risk = float(np.clip(self.civilian_risk - 0.07, 0.0, 1.0))
            self.economic_stability = float(np.clip(self.economic_stability - 0.06, 0.0, 1.0))
            self.legal_risk = float(np.clip(self.legal_risk + 0.06, 0.0, 1.0))
            self._append_timeline("Strategic action: temporary movement restrictions enacted")
            return "restriction order imposed", "impose_restriction_order", False

        if mapped.action_type == "authorize_emergency_procurement":
            self.recent_dispatch_effect += 0.09
            self.economic_stability = float(np.clip(self.economic_stability - 0.03, 0.0, 1.0))
            self.legal_risk = float(np.clip(self.legal_risk + 0.04, 0.0, 1.0))
            self._append_timeline("Strategic action: emergency procurement approved")
            return "emergency procurement approved", "authorize_emergency_procurement", False

        if mapped.action_type == "counter_misinformation_campaign":
            self.misinformation_index = float(np.clip(self.misinformation_index - 0.12, 0.0, 1.0))
            self.institutional_trust = float(np.clip(self.institutional_trust + 0.05, 0.0, 1.0))
            self._append_timeline("Strategic action: misinformation counter-campaign launched")
            return "counter-misinformation campaign launched", "counter_misinformation_campaign", False

        if mapped.action_type == "coordinate_cyber_command":
            self.link_outage_ratio = float(np.clip(self.link_outage_ratio - 0.12, 0.0, 1.0))
            self.legal_risk = float(np.clip(self.legal_risk + 0.03, 0.0, 1.0))
            self._append_timeline("Strategic action: cyber command coordinating infrastructure hardening")
            return "cyber command coordination active", "coordinate_cyber_command", False

        if mapped.action_type in {"dispatch_fire_truck", "send_medical_team", "deploy_drone_scan", "evacuate_zone", "request_backup"}:
            dispatch_map = {
                "dispatch_fire_truck": "fire",
                "send_medical_team": "medical",
                "deploy_drone_scan": "drone",
                "evacuate_zone": "evacuation",
                "request_backup": mapped.unit_type or "police",
            }
            chosen = dispatch_map[mapped.action_type]
            unit = self.emergency_units.get(chosen)
            if unit is None or unit.available <= 0:
                self.wrong_dispatches += 1
                return f"dispatch failed: no {chosen} units available", mapped.action_type, True

            unit.available -= 1
            unit.deployed += 1
            unit.cooldown_steps = 2
            effect = 0.04
            if mapped.action_type == "evacuate_zone":
                self.civilians_saved += int(max(1, round(2 + self.civilian_risk * 6)))
                self.civilian_risk = float(np.clip(self.civilian_risk - 0.12, 0.0, 1.0))
                effect = 0.12
            elif mapped.action_type == "send_medical_team":
                self.civilians_saved += int(max(1, round(1 + self.civilian_risk * 3)))
                self.civilian_risk = float(np.clip(self.civilian_risk - 0.08, 0.0, 1.0))
                effect = 0.10
            elif mapped.action_type == "dispatch_fire_truck":
                self.incident_severity = float(np.clip(self.incident_severity - 0.10, 0.0, 1.0))
                effect = 0.12
            elif mapped.action_type == "deploy_drone_scan":
                self.investigation_targets.update(self.services.keys())
                effect = 0.07

            if mapped.target_zone and mapped.target_zone not in {"zone_a", "zone_b", "zone_c", "city_core"}:
                self.wrong_dispatches += 1
                effect *= 0.3

            self.recent_dispatch_effect += effect
            self._append_timeline(f"Tactical dispatch: {chosen} unit deployed to {mapped.target_zone or 'unspecified zone'}")
            return f"{chosen} unit dispatched", mapped.action_type, False

        if mapped.action_type == "noop":
            self._append_timeline("Agent paused for observation")
            return "no operation", "noop", False

        if mapped.action_type == "get_metrics":
            target = mapped.target_service or "frontend"
            self.investigation_targets.add(target)
            svc = self.services[target]
            line = f"metrics[{target}] cpu={svc.cpu_utilization:.0f}% mem={svc.memory_utilization:.0f}% p95={svc.p95_latency:.0f}ms err={svc.error_rate:.2f}"
            self.investigation_log.append(line)
            self._append_timeline(f"Agent pulled metrics for {target}")
            return line, "get_metrics", False

        if mapped.action_type == "list_processes":
            lines = [f"{name}: workers={svc.instances} healthy={str(svc.healthy).lower()}" for name, svc in self.services.items()]
            self.investigation_targets.update(self.services.keys())
            output = " | ".join(lines)
            self.investigation_log.append(output)
            self._append_timeline("Agent listed cluster processes")
            return output, "list_processes", False

        if mapped.action_type == "read_last_n_logs":
            target = mapped.target_service or "frontend"
            self.investigation_targets.add(target)
            output = self._mock_service_log(target, mapped.n_lines)
            self.investigation_log.append(output)
            self._append_timeline(f"Agent inspected logs for {target}")
            return output, "read_last_n_logs", False

        if mapped.action_type == "check_network_connectivity":
            source = mapped.target_service or "frontend"
            dest = mapped.fallback_service or (self.services[source].reachable_upstreams[0] if self.services[source].reachable_upstreams else "db")
            self.investigation_targets.update({source, dest})
            ok = self._network_path_ok(source, dest)
            output = f"netcheck {source}->{dest}: {'ok' if ok else 'timeout'}"
            self.investigation_log.append(output)
            self._append_timeline(f"Agent ran network check {source}->{dest}")
            return output, "check_network_connectivity", False

        if mapped.action_type == "failover_database":
            if self.scenario.scenario_id != "regional_outage":
                return "failover skipped: primary region healthy", "failover_database", True
            self.scenario.db_primary_zone = "zone-b"
            self.scenario.db_failover_complete = True
            self.scenario.cross_zone_packet_loss = max(0.0, self.scenario.cross_zone_packet_loss - 0.75)
            self._resolve_root_cause("regional_outage")
            self._append_timeline("Agent executed cross-zone database failover")
            return "db failover complete zone-a->zone-b", "failover_database", False

        if mapped.action_type == "run_healthcheck":
            target = mapped.target_service
            ok = self._healthcheck(target)
            if ok:
                self.verification_successes += 1
            subject = target or "cluster"
            output = f"healthcheck {subject}: {'pass' if ok else 'fail'}"
            self._append_timeline(f"Agent verified health for {subject}")
            return output, "run_healthcheck", False

        if mapped.action_type == "ask_developer":
            hint = self._developer_hint(mapped.question)
            self.investigation_log.append(f"developer: {hint}")
            self._append_timeline("Agent requested developer context")
            return hint, "ask_developer", False

        if mapped.action_type == "load_test":
            self.load_test_active_steps = max(self.load_test_active_steps, 2)
            self._append_timeline("Agent triggered load test")
            return "load test started", "load_test", False

        if mapped.action_type == "run_command":
            command = (mapped.command or "").strip().lower()
            if "kill noisy-neighbor" in command:
                self.scenario.noisy_neighbor_io = max(0.0, self.scenario.noisy_neighbor_io - 0.75)
                self._resolve_root_cause("resource_exhaustion")
                self._append_timeline("Agent killed noisy-neighbor process")
                return "killed noisy-neighbor", "run_command", False
            if self._is_unsafe_action(mapped):
                self._append_timeline("Unsafe destructive command attempted")
                return "blocked unsafe command", "run_command", True
            return "command had no effect", "run_command", True

        if mapped.action_type == "restart_service":
            if not mapped.target_service:
                return "restart ignored: missing target", "restart_service", True
            svc = self.services[mapped.target_service]
            svc.healthy = True
            svc.error_rate = max(0.0, svc.error_rate - 0.2)
            svc.p95_latency = max(70.0, svc.p95_latency * 0.82)
            svc.memory_utilization = max(35.0, svc.memory_utilization * 0.8)
            self._append_timeline(f"Agent restarted {mapped.target_service}")
            return f"restarted {mapped.target_service}", "restart_service", False

        if mapped.action_type == "rollback_deployment":
            if not mapped.target_service:
                return "rollback ignored: missing target", "rollback_deployment", True
            svc = self.services[mapped.target_service]
            svc.version = svc.rollback_version
            svc.healthy = True
            if mapped.target_service == "auth":
                self._resolve_root_cause("heisenbug")
            self._append_timeline(f"Agent rolled back {mapped.target_service}")
            return f"rolled back {mapped.target_service}", "rollback_deployment", False

        if mapped.action_type == "scale_up_replicas":
            if not mapped.target_service:
                return "scale ignored: missing target", "scale_up_replicas", True
            svc = self.services[mapped.target_service]
            delta = max(1, mapped.delta_instances or 1)
            svc.instances = int(np.clip(svc.instances + delta, 1, 10))
            svc.desired_instances = svc.instances
            svc.error_rate = max(0.0, svc.error_rate - 0.15)
            self._append_timeline(f"Agent scaled {mapped.target_service} by +{delta}")
            return f"scaled {mapped.target_service} to {svc.instances}", "scale_up_replicas", False

        if mapped.action_type == "edit_config_line":
            if not mapped.config_key:
                return "config edit ignored: missing key", "edit_config_line", True
            self.config_runtime[mapped.config_key] = mapped.config_value or ""
            if mapped.config_key == "db_port":
                self.scenario.db_port_actual = int(mapped.config_value or "0") if str(mapped.config_value or "").isdigit() else -1
                if self.scenario.db_port_actual == self.scenario.db_port_expected:
                    self._resolve_root_cause("config_drift")
            self._append_timeline(f"Agent edited config {mapped.config_key}")
            return f"edited config {mapped.config_key}", "edit_config_line", False

        return "action ignored", mapped.action_type, True

    def _map_legacy_action(self, action: IncidentCommanderAction) -> IncidentCommanderAction:
        mapping = {
            "scale_service": "scale_up_replicas",
            "rollback_deploy": "rollback_deployment",
            "page_human": "ask_developer",
        }
        mapped_type = mapping.get(action.action_type, action.action_type)
        return IncidentCommanderAction(
            action_type=mapped_type,
            target_service=action.target_service,
            delta_instances=action.delta_instances,
            request_fraction=action.request_fraction,
            target_version=action.target_version,
            fallback_service=action.fallback_service,
            n_lines=action.n_lines,
            config_key=action.config_key,
            config_value=action.config_value,
            question=action.question,
            command=action.command,
            note=action.note,
            unit_type=action.unit_type,
            target_zone=action.target_zone,
            priority=action.priority,
            strategy_level=action.strategy_level,
        )

    def _simulate_scenario_drift(self) -> None:
        for unit in self.emergency_units.values():
            if unit.cooldown_steps > 0:
                unit.cooldown_steps -= 1
            if unit.cooldown_steps == 0 and unit.deployed > 0:
                unit.available += unit.deployed
                unit.deployed = 0

        if self.scenario.scenario_id == "resource_exhaustion":
            self.scenario.noisy_neighbor_io = float(np.clip(self.scenario.noisy_neighbor_io + 0.04, 0.0, 1.2))
            self.tmp_files.append(f"/tmp/io-spike-{self.timestep}.tmp")
            if len(self.tmp_files) > 40:
                self.tmp_files = self.tmp_files[-40:]

        if self.scenario.scenario_id == "heisenbug" and self.load_test_active_steps > 0:
            self.scenario.heisenbug_armed = True
            if self._effective_traffic_level() > self.task.base_traffic * 1.6:
                self.scenario.heisenbug_triggered = True

        if self.scenario.scenario_id == "regional_outage" and not self.scenario.db_failover_complete:
            self.scenario.cross_zone_packet_loss = float(np.clip(self.scenario.cross_zone_packet_loss + 0.03, 0.0, 0.98))

    def _simulate_disaster_progression(self) -> None:
        weather_impact = {
            "clear": 0.00,
            "windy": 0.04,
            "storm": 0.08,
            "heatwave": 0.05,
        }
        if self.task.task_id in {"hard", "longhaul", "blackout"}:
            if self.timestep % 9 == 0 and self.timestep > 0:
                self.weather_condition = str(self.rng.choice(["clear", "windy", "storm", "heatwave"]))

        escalation = 0.015 + weather_impact.get(self.weather_condition, 0.0)
        if self.phase in {"surge", "thundering-herd"}:
            escalation += 0.03
        if self.recent_dispatch_effect > 0:
            escalation -= min(0.09, self.recent_dispatch_effect)
        self.recent_dispatch_effect = max(0.0, self.recent_dispatch_effect * 0.5)

        self.incident_severity = float(np.clip(self.incident_severity + escalation, 0.0, 1.0))
        self.civilian_risk = float(np.clip(self.civilian_risk + 0.5 * escalation, 0.0, 1.0))

        if self.incident_severity > 0.72:
            self.incident_type = "compound_outage"
        elif self.incident_severity > 0.48:
            self.incident_type = "critical_service_disruption"
        else:
            self.incident_type = "infra_outage"

        if self.task.adversarial_shift_rate > 0 and self.rng.random() < self.task.adversarial_shift_rate:
            burst = float(self.rng.uniform(0.05, 0.12))
            self.incident_severity = float(np.clip(self.incident_severity + burst, 0.0, 1.0))
            self.civilian_risk = float(np.clip(self.civilian_risk + (burst * 0.7), 0.0, 1.0))
            self.misinformation_index = float(np.clip(self.misinformation_index + (burst * 0.9), 0.0, 1.0))
            self._append_timeline("Adversarial shift: unexpected secondary outage wave")

        trust_decay = (self.incident_severity * 0.03) + (self.misinformation_index * 0.02)
        if self.weather_condition == "storm":
            trust_decay += 0.01
        self.institutional_trust = float(np.clip(self.institutional_trust - trust_decay, 0.0, 1.0))
        self.economic_stability = float(np.clip(self.economic_stability - (0.015 + 0.02 * self.incident_severity), 0.0, 1.0))
        self.legal_risk = float(np.clip(self.legal_risk + max(0.0, self.incident_severity - 0.6) * 0.04, 0.0, 1.0))
        self.misinformation_index = float(np.clip(self.misinformation_index + (0.01 + 0.03 * self.civilian_risk), 0.0, 1.0))

        self._simulate_topology_disruptions()

    def _simulate_topology_disruptions(self) -> None:
        if not self.region_status:
            return
        for zone in list(self.region_status.keys()):
            decay = 0.0
            if self.scenario.scenario_id == "regional_outage":
                decay += 0.02
            decay += max(0.0, self.incident_severity - 0.55) * 0.04
            if self.weather_condition == "storm":
                decay += 0.03
            if zone == "zone_b" and self.scenario.db_primary_zone == "zone-a" and not self.scenario.db_failover_complete:
                decay += 0.02
            self.region_status[zone] = float(np.clip(self.region_status[zone] - decay, 0.25, 1.0))
        self.scenario.region_link_health = {k: round(v, 4) for k, v in self.region_status.items()}

        healthy_links = [v for v in self.region_status.values() if v >= 0.7]
        self.link_outage_ratio = 1.0 - (len(healthy_links) / max(1, len(self.region_status)))

    def _inject_scheduled_incident_event(self) -> None:
        # Scheduled shocks make long-horizon tasks non-trivial and improve task separation by difficulty.
        if self.timestep <= 0 or self.timestep not in self.task.incident_schedule:
            return

        if self.scenario.scenario_id == "resource_exhaustion":
            self.scenario.noisy_neighbor_io = float(np.clip(self.scenario.noisy_neighbor_io + 0.22, 0.0, 1.2))
            self.tmp_files.append(f"/tmp/noisy-neighbor-burst-{self.timestep}.tmp")
            self._append_timeline("Scheduled burst: noisy-neighbor IO intensified")
            return

        if self.scenario.scenario_id == "config_drift":
            drift_port = int(15000 + self.rng.integers(200, 3000))
            self.scenario.db_port_actual = drift_port
            self.config_runtime["db_port"] = str(drift_port)
            self._append_timeline("Scheduled drift: auth config regressed to stale DB port")
            return

        if self.scenario.scenario_id == "heisenbug":
            self.scenario.heisenbug_armed = True
            if self._effective_traffic_level() >= self.task.base_traffic * 1.2:
                self.scenario.heisenbug_triggered = True
            self._append_timeline("Scheduled fault: race-condition bug re-armed under load")
            return

        if self.scenario.scenario_id == "regional_outage":
            self.scenario.db_failover_complete = False
            self.scenario.cross_zone_packet_loss = float(np.clip(self.scenario.cross_zone_packet_loss + 0.24, 0.0, 0.98))
            self._append_timeline("Scheduled outage wave: cross-zone packet loss spiked")

    def _is_unforced_page(
        self,
        action: IncidentCommanderAction,
        unresolved_critical: int,
        traffic_profile: float,
        p95_latency: float,
    ) -> bool:
        if action.action_type != "ask_developer":
            return False
        if unresolved_critical > 0:
            return False
        if p95_latency >= 250.0:
            return False
        return traffic_profile < (self.task.base_traffic * 1.2)

    def _is_contradictory_action(
        self,
        action: IncidentCommanderAction,
        incorrect_action: bool,
        unresolved_critical_before: int,
    ) -> bool:
        if incorrect_action:
            return False

        if action.action_type == "load_test" and unresolved_critical_before > 0:
            return True

        if action.action_type == "failover_database" and self.scenario.scenario_id != "regional_outage":
            return True

        recent = self.action_history[-2:]
        if len(recent) >= 2 and action.action_type in {"restart_service", "rollback_deployment", "scale_up_replicas"}:
            same_target_repeats = 0
            for _, recent_type, recent_target in recent:
                if recent_type == action.action_type and recent_target == action.target_service:
                    same_target_repeats += 1
            if same_target_repeats >= 2:
                return True

        if action.action_type == "ask_developer" and self.dev_hint_cooldown > 0:
            return True

        return False

    def _current_traffic_profile(self) -> float:
        if self.task.task_id == "easy":
            self.phase = "investigation"
            base = self.task.base_traffic + (0.02 * self.timestep)
        elif self.task.task_id == "medium":
            self.phase = "degradation" if self.timestep < 16 else "recovery"
            base = self.task.base_traffic if self.timestep < 12 else self.task.peak_traffic * 0.9
        elif self.task.task_id == "hard":
            self.phase = "surge" if self.timestep >= 10 else "degradation"
            base = self.task.peak_traffic if self.timestep >= 10 else self.task.base_traffic
        elif self.task.task_id == "longhaul":
            self.phase = "slow-burn" if self.timestep < 24 else "surge"
            base = self.task.base_traffic * 0.95 if self.timestep < 24 else self.task.peak_traffic
        else:
            self.phase = "thundering-herd" if self.timestep >= 8 else "brownout"
            base = self.task.peak_traffic if self.timestep >= 8 else self.task.base_traffic * 1.15

        if self.load_test_active_steps > 0:
            base *= 1.35
        base *= 1.0 + (self.incident_severity * 0.12)
        return float(base)

    def _effective_traffic_level(self) -> float:
        return self._current_traffic_profile()

    def _simulate_traffic(self, traffic_profile: float):
        service_loads: Dict[str, float] = defaultdict(float)
        base_weights = {"frontend": 0.45, "auth": 0.30, "db": 0.25}

        db_penalty = 0.0
        if self.scenario.scenario_id == "resource_exhaustion":
            db_penalty += 0.35 * self.scenario.noisy_neighbor_io
        if self.scenario.scenario_id == "config_drift" and self.scenario.db_port_actual != self.scenario.db_port_expected:
            db_penalty += 0.6
        if self.scenario.scenario_id == "regional_outage" and not self.scenario.db_failover_complete:
            db_penalty += 0.55 * self.scenario.cross_zone_packet_loss

        systemic_pressure = 1.0 + (self.incident_severity * 0.25) + (self.civilian_risk * 0.10) + (self.link_outage_ratio * 0.30)

        for name, svc in self.services.items():
            pressure = 1.0
            if name == "db":
                pressure += db_penalty
            if name == "auth" and self.scenario.scenario_id == "heisenbug" and self.scenario.heisenbug_triggered:
                pressure += 0.7
            demand = traffic_profile * base_weights[name] * pressure * systemic_pressure
            if not svc.healthy:
                demand *= 1.1
            service_loads[name] = demand

        served_total = 0.0
        demand_total = 0.0
        latency_values: List[float] = []

        self._apply_dependency_failures()

        for name, svc in self.services.items():
            demand = service_loads[name]
            demand_total += demand
            capacity = float(max(1, svc.instances)) * (1.0 if svc.healthy else 0.5)
            served = min(demand, capacity)
            served_total += served
            backlog = max(0.0, demand - served)

            util = demand / max(1.0, capacity)
            svc.queue_depth = int(round(backlog * 20))
            svc.cpu_utilization = float(np.clip(35.0 + (util * 52.0), 5.0, 100.0))
            svc.memory_utilization = float(np.clip(svc.memory_utilization + (util * 4.0), 10.0, 99.0))
            svc.p95_latency = float(np.clip(75.0 + util * 180.0 + backlog * 20.0, 20.0, 1000.0))
            svc.error_rate = float(np.clip(backlog / max(0.5, demand), 0.0, 1.0))

            latency_values.append(svc.p95_latency)

        self._apply_dependency_failures()

        cost_per_step = 0.0
        for svc in self.services.values():
            cost_per_step += 0.03 * svc.instances
        cost_per_step += 0.0015 * len(self.tmp_files)

        uptime_ratio = 1.0 if demand_total <= 0 else served_total / demand_total
        p95_latency = float(np.percentile(latency_values, 95)) if latency_values else 0.0

        return service_loads, float(np.clip(uptime_ratio, 0.0, 1.0)), p95_latency, float(cost_per_step)

    def _apply_dependency_failures(self) -> None:
        db = self.services["db"]
        auth = self.services["auth"]
        frontend = self.services["frontend"]

        db_connection_ok = db.healthy and self.scenario.db_port_actual == self.scenario.db_port_expected
        if self.scenario.scenario_id == "regional_outage" and not self.scenario.db_failover_complete:
            db_connection_ok = db_connection_ok and self.scenario.cross_zone_packet_loss < 0.35

        if not db_connection_ok:
            auth.healthy = False
            auth.error_rate = max(auth.error_rate, 0.55)
            auth.p95_latency = max(auth.p95_latency, 320.0)
            auth.last_action_result = "upstream db connect timeout"
        else:
            auth.healthy = True

        if not auth.healthy:
            frontend.healthy = False
            frontend.error_rate = max(frontend.error_rate, 0.50)
            frontend.p95_latency = max(frontend.p95_latency, 360.0)
            frontend.last_action_result = "auth dependency failure -> 404/500"
        else:
            frontend.healthy = True

    def _update_incidents(self) -> None:
        for incident in self.active_incidents:
            if incident.resolved:
                continue
            incident.age_steps += 1

        self._sync_incident("frontend", "cascade", "critical", not self.services["frontend"].healthy)
        self._sync_incident("auth", "bad_deploy", "high", not self.services["auth"].healthy)
        self._sync_incident("db", "database_lock", "critical", not self.services["db"].healthy or self.scenario.db_port_actual != self.scenario.db_port_expected)

        for incident in self.active_incidents:
            if incident.resolved:
                continue
            service_ok = self.services[incident.service].healthy
            if incident.incident_type == "database_lock":
                service_ok = service_ok and self.scenario.db_port_actual == self.scenario.db_port_expected
            if service_ok:
                incident.resolved = True
                self.resolved_incidents += 1

    def _sync_incident(self, service: str, incident_type: str, severity: str, condition: bool) -> None:
        existing = None
        for incident in self.active_incidents:
            if incident.service == service and incident.incident_type == incident_type and not incident.resolved:
                existing = incident
                break

        if condition and existing is None:
            self.active_incidents.append(
                ActiveIncident(
                    incident_id=f"{self.task.task_id}-{self.timestep}-{service}-{incident_type}",
                    incident_type=incident_type,
                    service=service,
                    severity=severity,
                )
            )
            self.total_incidents += 1

    def _sla_breaches_this_step(self, p95_latency: float, traffic_profile: float) -> int:
        breaches = 0
        if p95_latency > 280.0:
            breaches += 1
        if p95_latency > 420.0:
            breaches += 1
        if traffic_profile > self.task.base_traffic * 1.4 and p95_latency > 250.0:
            breaches += 1
        if not self.services["frontend"].healthy:
            breaches += 1
        return breaches

    def _all_services_recovered(self) -> bool:
        if self.timestep < 3:
            return False
        all_healthy = all(service.healthy for service in self.services.values())
        return all_healthy and self.scenario.db_port_actual == self.scenario.db_port_expected and self.scenario.noisy_neighbor_io < 0.2 and not self.scenario.heisenbug_triggered

    def _network_path_ok(self, source: str, dest: str) -> bool:
        if source not in self.services or dest not in self.services:
            return False
        if dest == "db" and self.scenario.db_port_actual != self.scenario.db_port_expected:
            return False
        if self.link_outage_ratio > 0.45 and source in self.dependency_graph and dest in self.dependency_graph.get(source, []):
            return False
        return self.services[source].healthy and self.services[dest].healthy

    def _healthcheck(self, target: Optional[str]) -> bool:
        if target and target in self.services:
            svc = self.services[target]
            if target == "db":
                return svc.healthy and self.scenario.db_port_actual == self.scenario.db_port_expected
            return svc.healthy and svc.error_rate < 0.2 and svc.p95_latency < 260
        return self._all_services_recovered()

    def _mock_service_log(self, service: str, n_lines: int) -> str:
        if service == "frontend":
            if not self.services["frontend"].healthy:
                return f"[{n_lines} lines] GET /api/login -> 404 upstream auth unavailable"
            return f"[{n_lines} lines] frontend stable"
        if service == "auth":
            if self.scenario.db_port_actual != self.scenario.db_port_expected:
                return f"[{n_lines} lines] auth db dial tcp db:{self.scenario.db_port_actual} connection refused"
            if self.scenario.heisenbug_triggered:
                return f"[{n_lines} lines] panic: race condition in token cache under high load"
            return f"[{n_lines} lines] auth stable"
        if service == "db":
            if self.scenario.noisy_neighbor_io > 0.4:
                return f"[{n_lines} lines] db io wait above threshold noisy-neighbor detected"
            if self.scenario.scenario_id == "regional_outage" and not self.scenario.db_failover_complete:
                return f"[{n_lines} lines] db replication lag high in {self.scenario.db_primary_zone}; cross-zone packet loss={self.scenario.cross_zone_packet_loss:.2f}"
            if self.scenario.db_port_actual != self.scenario.db_port_expected:
                return f"[{n_lines} lines] db listening on {self.scenario.db_port_expected}; clients using wrong port"
            return f"[{n_lines} lines] db stable"
        return f"[{n_lines} lines] unknown service"

    def _developer_hint(self, question: Optional[str]) -> str:
        if self.dev_hint_cooldown > 0:
            return "Developer is busy. Try again next step."
        self.dev_hint_cooldown = 2

        if self.scenario.scenario_id == "config_drift":
            return "We pushed an auth config update 10 minutes ago. Might have changed DB connection settings."
        if self.scenario.scenario_id == "heisenbug":
            return "Auth started flaking only during traffic spikes after the latest deployment."
        if self.scenario.scenario_id == "regional_outage":
            return "Cross-zone packet loss is spiking. Failing over DB primary might stabilize auth latency."
        return "Infra reported a noisy-neighbor process saturating disk IO on the DB node."

    def _resolve_root_cause(self, scenario_id: str) -> None:
        if self.scenario.scenario_id != scenario_id:
            return
        self.root_cause_resolutions += 1
        if scenario_id == "resource_exhaustion":
            self.scenario.noisy_neighbor_io = 0.0
        if scenario_id == "config_drift":
            self.scenario.db_port_actual = self.scenario.db_port_expected
            self.config_runtime["db_port"] = str(self.scenario.db_port_expected)
        if scenario_id == "heisenbug":
            self.scenario.heisenbug_triggered = False
            self.scenario.heisenbug_armed = False
        if scenario_id == "regional_outage":
            self.scenario.cross_zone_packet_loss = 0.0
            self.scenario.db_failover_complete = True
            for zone in self.region_status:
                self.region_status[zone] = min(1.0, self.region_status[zone] + 0.10)

    def _select_scenario(self) -> str:
        choices = self.task.scenario_mix or ["resource_exhaustion"]
        index = int(self.rng.integers(0, len(choices)))
        return choices[index]

    def _build_scenario(self, scenario_id: str) -> ScenarioState:
        if scenario_id == "resource_exhaustion":
            return ScenarioState(
                scenario_id="resource_exhaustion",
                noisy_neighbor_io=0.45,
                region_link_health={"zone_a": 0.96, "zone_b": 0.94, "zone_c": 0.95},
            )
        if scenario_id == "config_drift":
            return ScenarioState(
                scenario_id="config_drift",
                db_port_expected=5432,
                db_port_actual=15432,
                region_link_health={"zone_a": 0.95, "zone_b": 0.93, "zone_c": 0.94},
            )
        if scenario_id == "regional_outage":
            return ScenarioState(
                scenario_id="regional_outage",
                db_port_expected=5432,
                db_port_actual=5432,
                cross_zone_packet_loss=0.42,
                db_primary_zone="zone-a",
                db_failover_complete=False,
                region_link_health={"zone_a": 0.72, "zone_b": 0.65, "zone_c": 0.78},
            )
        return ScenarioState(
            scenario_id="heisenbug",
            heisenbug_armed=False,
            heisenbug_triggered=False,
            region_link_health={"zone_a": 0.93, "zone_b": 0.90, "zone_c": 0.91},
        )

    def _apply_scenario_bootstrap(self) -> None:
        if self.scenario.region_link_health:
            for zone, health in self.scenario.region_link_health.items():
                self.region_status[zone] = float(np.clip(health, 0.25, 1.0))
        if self.scenario.scenario_id == "config_drift":
            self.config_broken["db_port"] = str(self.scenario.db_port_actual)
            self.config_runtime = dict(self.config_broken)
        if self.scenario.scenario_id == "heisenbug":
            self.services["auth"].version = "v1-buggy"
        if self.scenario.scenario_id == "regional_outage":
            self.services["db"].last_action_result = "cross-zone packet loss detected on primary"

    def _append_timeline(self, message: str) -> None:
        timestamp = f"12:00:{self.timestep:02d}"
        self.live_timeline.append(f"{timestamp} - {message}")
        if len(self.live_timeline) > 120:
            self.live_timeline = self.live_timeline[-120:]

    def _apply_observation_noise(self, services: Dict[str, ServiceState]) -> None:
        if self.task.observation_noise <= 0:
            for name, service in services.items():
                service.observed_p95_latency = service.p95_latency
                service.observed_error_rate = service.error_rate
                service.metric_staleness_steps = 0
                self.last_observed_metrics[name] = (service.p95_latency, service.error_rate)
            return

        for name, service in services.items():
            delayed = self.rng.random() < self.task.observation_noise
            prev_latency, prev_error = self.last_observed_metrics.get(name, (service.p95_latency, service.error_rate))
            if delayed and self.timestep > 0:
                service.observed_p95_latency = prev_latency
                service.observed_error_rate = prev_error
                service.metric_staleness_steps = max(1, service.metric_staleness_steps + 1)
            else:
                service.observed_p95_latency = service.p95_latency
                service.observed_error_rate = service.error_rate
                service.metric_staleness_steps = 0
            self.last_observed_metrics[name] = (service.observed_p95_latency, service.observed_error_rate)

    def _symptoms_view(self) -> List[str]:
        frontend = self.services["frontend"]
        auth = self.services["auth"]
        symptoms: List[str] = []
        if not frontend.healthy:
            symptoms.append("frontend returns 404/500 for login and checkout")
        if frontend.p95_latency > 280:
            symptoms.append("frontend p95 latency above SLO")
        if auth.error_rate > 0.3:
            symptoms.append("auth error burst detected")
        if self.sla_breaches > 0:
            symptoms.append("SLA breach counter increasing")
        if not symptoms:
            symptoms.append("customer traffic looks stable")
        return symptoms

    def _masked_incidents(self) -> List[ActiveIncident]:
        masked: List[ActiveIncident] = []
        for incident in self.active_incidents:
            label = incident.incident_type
            if incident.service in {"auth", "db"} and incident.service not in self.investigation_targets:
                label = "cascade"
            masked.append(
                ActiveIncident(
                    incident_id=incident.incident_id,
                    incident_type=label,
                    service=incident.service,
                    severity=incident.severity,
                    age_steps=incident.age_steps,
                    resolved=incident.resolved,
                    resolution_timer=incident.resolution_timer,
                )
            )
        return masked

    def _available_actions(self) -> List[str]:
        return [
            "get_metrics",
            "list_processes",
            "read_last_n_logs",
            "check_network_connectivity",
            "failover_database",
            "restart_service",
            "rollback_deployment",
            "scale_up_replicas",
            "edit_config_line",
            "run_healthcheck",
            "ask_developer",
            "load_test",
            "run_command",
            "declare_emergency",
            "allocate_resources",
            "request_national_support",
            "issue_public_briefing",
            "impose_restriction_order",
            "authorize_emergency_procurement",
            "counter_misinformation_campaign",
            "coordinate_cyber_command",
            "dispatch_fire_truck",
            "send_medical_team",
            "deploy_drone_scan",
            "evacuate_zone",
            "request_backup",
            "noop",
        ]

    def _get_observation(self, traffic_profile: float = 0.0, uptime: float = 1.0, p95_latency: float = 80.0) -> IncidentCommanderObservation:
        services = {name: service.model_copy(deep=True) for name, service in self.services.items()}
        self._apply_observation_noise(services)

        return IncidentCommanderObservation(
            services=services,
            active_incidents=self._masked_incidents(),
            step=self.timestep,
            step_budget=self.task.max_steps,
            traffic_level=round(traffic_profile, 4),
            uptime=round(uptime, 4),
            p95_latency=round(p95_latency, 4),
            sla_breaches=self.sla_breaches,
            cost_per_step=round(self.cumulative_cost, 4),
            last_action_result=self.last_action_result,
            phase=self.phase,
            symptoms=self._symptoms_view(),
            terminal_output=self.investigation_log[-5:],
            investigation_log=self.investigation_log[-20:],
            live_timeline=self.live_timeline[-30:],
            available_actions=self._available_actions(),
            scenario_hint="Root cause is hidden until investigated",
            incident_type=self.incident_type,
            incident_severity=round(self.incident_severity, 4),
            weather_condition=self.weather_condition,
            civilian_risk=round(self.civilian_risk, 4),
            emergency_units={k: v.model_copy(deep=True) for k, v in self.emergency_units.items()},
            strategic_options=[
                "declare_emergency",
                "allocate_resources",
                "request_national_support",
                "issue_public_briefing",
                "impose_restriction_order",
                "authorize_emergency_procurement",
                "counter_misinformation_campaign",
                "coordinate_cyber_command",
            ],
            tactical_options=["dispatch_fire_truck", "send_medical_team", "deploy_drone_scan", "evacuate_zone", "request_backup"],
            region_status={k: round(v, 4) for k, v in self.region_status.items()},
            dependency_graph={k: list(v) for k, v in self.dependency_graph.items()},
            commitment_mode=self.commitment_mode,
            institutional_trust=round(self.institutional_trust, 4),
            economic_stability=round(self.economic_stability, 4),
            legal_risk=round(self.legal_risk, 4),
            misinformation_index=round(self.misinformation_index, 4),
        )

    def get_state(self) -> Dict[str, object]:
        return {
            "task_id": self.task.task_id,
            "step": self.timestep,
            "services": {name: service.model_dump() for name, service in self.services.items()},
            "active_incidents": [incident.model_dump() for incident in self.active_incidents],
            "scenario": self.scenario.model_dump(),
            "config_runtime": dict(self.config_runtime),
            "tmp_files": list(self.tmp_files),
            "sla_breaches": self.sla_breaches,
            "resolved_incidents": self.resolved_incidents,
            "total_incidents": self.total_incidents,
            "cumulative_cost": round(self.cumulative_cost, 4),
            "phase": self.phase,
            "open_sockets": self.open_sockets,
            "burn_budget_ratio": round(self.downtime_used / max(1e-6, self.burn_budget_total), 6),
            "incident_type": self.incident_type,
            "incident_severity": round(self.incident_severity, 4),
            "weather_condition": self.weather_condition,
            "civilian_risk": round(self.civilian_risk, 4),
            "civilians_saved": self.civilians_saved,
            "emergency_units": {k: v.model_dump() for k, v in self.emergency_units.items()},
            "region_status": {k: round(v, 4) for k, v in self.region_status.items()},
            "dependency_graph": {k: list(v) for k, v in self.dependency_graph.items()},
            "commitment_mode": self.commitment_mode,
            "commitment_switches": self.commitment_switches,
            "link_outage_ratio": round(self.link_outage_ratio, 6),
            "institutional_trust": round(self.institutional_trust, 6),
            "economic_stability": round(self.economic_stability, 6),
            "legal_risk": round(self.legal_risk, 6),
            "misinformation_index": round(self.misinformation_index, 6),
        }

    def get_metrics(self) -> Dict[str, object]:
        uptime_score = float(np.mean(self.uptime_history)) if self.uptime_history else 1.0
        avg_latency = float(np.mean(self.latency_history)) if self.latency_history else 80.0
        latency_score = max(0.0, min(1.0, 1.0 - avg_latency / 700.0))
        strict_sla_window = self.task.max_steps // 8 if self.task.task_id in {"hard", "longhaul", "blackout"} else self.task.max_steps // 4
        sla_score = max(0.0, min(1.0, 1.0 - self.sla_breaches / max(1, strict_sla_window)))
        cost_score = max(0.0, min(1.0, 1.0 - self.cumulative_cost / max(1e-6, self.task.cost_budget * self.task.max_steps)))
        recovery_score = max(0.0, min(1.0, self.resolved_incidents / max(1, self.total_incidents)))

        action_total = max(1, self.timestep)
        passive_ratio = self.action_counts["noop"] / action_total
        action_discipline_score = max(0.0, min(1.0, 1.0 - passive_ratio))

        root_cause_identification_rate = max(0.0, min(1.0, len(self.investigation_targets) / 3.0))
        safe_ops_score = max(0.0, min(1.0, 1.0 - (self.unsafe_actions / action_total)))
        verification_score = max(0.0, min(1.0, self.verification_successes / max(1, self.action_counts["run_healthcheck"])))
        resilience_score = max(0.0, min(1.0, (0.5 * uptime_score) + (0.3 * safe_ops_score) + (0.2 * verification_score)))

        return {
            "uptime_score": round(uptime_score, 6),
            "avg_latency": round(avg_latency, 6),
            "latency_score": round(latency_score, 6),
            "sla_score": round(sla_score, 6),
            "cost_score": round(cost_score, 6),
            "recovery_score": round(recovery_score, 6),
            "action_discipline_score": round(action_discipline_score, 6),
            "root_cause_identification_rate": round(root_cause_identification_rate, 6),
            "safe_ops_score": round(safe_ops_score, 6),
            "verification_score": round(verification_score, 6),
            "resilience_score": round(resilience_score, 6),
            "burn_budget_ratio": round(self.downtime_used / max(1e-6, self.burn_budget_total), 6),
            "incident_severity": round(self.incident_severity, 6),
            "civilian_risk": round(self.civilian_risk, 6),
            "civilians_saved": int(self.civilians_saved),
            "wrong_dispatches": int(self.wrong_dispatches),
            "delayed_response_steps": int(self.delayed_response_steps),
            "commitment_switches": int(self.commitment_switches),
            "link_outage_ratio": round(self.link_outage_ratio, 6),
            "institutional_trust": round(self.institutional_trust, 6),
            "economic_stability": round(self.economic_stability, 6),
            "legal_risk": round(self.legal_risk, 6),
            "misinformation_index": round(self.misinformation_index, 6),
            "governance_score": round(max(0.0, min(1.0, (0.45 * self.institutional_trust) + (0.35 * self.economic_stability) + (0.20 * (1.0 - self.legal_risk)))), 6),
            "timeline": self.live_timeline,
            "reward_trace": self.reward_trace,
        }

    def build_episode_result(self) -> EpisodeResult:
        metrics = self.get_metrics()
        total = (
            0.28 * float(metrics["uptime_score"])
            + 0.16 * float(metrics["latency_score"])
            + 0.18 * float(metrics["sla_score"])
            + 0.10 * float(metrics["cost_score"])
            + 0.10 * float(metrics["recovery_score"])
            + 0.08 * float(metrics["root_cause_identification_rate"])
            + 0.05 * float(metrics["safe_ops_score"])
            + 0.05 * float(metrics["verification_score"])
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
            escalations_used=self.action_counts["ask_developer"],
            sla_breaches=self.sla_breaches,
            burn_budget_ratio=float(metrics["burn_budget_ratio"]),
            root_cause_identification_rate=float(metrics["root_cause_identification_rate"]),
            safe_ops_score=float(metrics["safe_ops_score"]),
            verification_score=float(metrics["verification_score"]),
            commitment_switches=int(metrics["commitment_switches"]),
            db_recovered=self.services["db"].healthy and self.scenario.db_port_actual == self.scenario.db_port_expected,
            auth_recovered=self.services["auth"].healthy,
            frontend_recovered=self.services["frontend"].healthy,
            total_score=round(max(0.0, min(1.0, total)), 6),
        )
