from __future__ import annotations

from typing import List

from .models import IncidentCommanderAction, IncidentCommanderObservation, IncidentCommanderReward, TaskConfig


class RewardCalculator:
    def __init__(self, task: TaskConfig):
        self.task = task

    def calculate(
        self,
        obs: IncidentCommanderObservation,
        action: IncidentCommanderAction,
        resolved_incidents: int,
        unresolved_critical: int,
        total_incidents: int,
        cost_ratio: float,
        uptime_ratio: float,
        unforced_page: bool,
        action_streak: int,
        mttr_resolved_ages: List[int],
        burn_budget_ratio: float,
        contradictory_action: bool,
        incorrect_action: bool,
        unsafe_action: bool,
        investigation_coverage: float,
        steps_taken: int,
        ideal_steps: int,
    ) -> IncidentCommanderReward:
        uptime = max(0.0, min(1.0, uptime_ratio))
        latency = max(0.0, min(1.0, 1.0 - obs.p95_latency / 700.0))
        sla = max(0.0, min(1.0, 1.0 - obs.sla_breaches / max(1, total_incidents + 2)))
        cost = max(0.0, min(1.0, 1.0 - cost_ratio))
        recovery = max(0.0, min(1.0, resolved_incidents / max(1, total_incidents)))

        success = (0.35 * uptime) + (0.20 * latency) + (0.20 * sla) + (0.10 * cost) + (0.15 * recovery)

        latency_penalty = max(0.0, min(0.35, (obs.p95_latency - 220.0) / 1200.0))
        resource_waste_penalty = max(0.0, min(0.35, max(0.0, cost_ratio - 1.0) * 0.30))
        incorrect_action_penalty = 0.14 if incorrect_action else 0.0
        safety_penalty = 0.55 if unsafe_action else 0.0

        total = success - latency_penalty - resource_waste_penalty - incorrect_action_penalty - safety_penalty
        if action.action_type == "noop" and unresolved_critical > 0:
            total -= 0.10
        if unforced_page:
            total -= 0.05
        if action_streak >= 5 and action.action_type in {"noop", "page_human", "ask_developer"}:
            total -= 0.04

        mttr_bonus = 0.0
        for age in mttr_resolved_ages:
            if age <= 3:
                mttr_bonus += min(0.18, (2 ** (10 - max(0, age))) / 2048.0)

        burn_budget_penalty = 0.0
        if burn_budget_ratio <= 1.0:
            burn_budget_penalty = 0.02 * burn_budget_ratio
        else:
            burn_budget_penalty = min(0.26, 0.03 + 0.12 * (burn_budget_ratio - 1.0) + 0.08)

        anti_panic_penalty = 0.08 if contradictory_action else 0.0

        if steps_taken > ideal_steps:
            overrun = (steps_taken - ideal_steps) / max(1.0, float(ideal_steps))
            total -= min(0.25, 0.22 * overrun)

        if investigation_coverage < 0.25 and action.action_type in {
            "restart_service",
            "rollback_deployment",
            "scale_up_replicas",
            "edit_config_line",
            "scale_service",
            "rollback_deploy",
        }:
            total -= 0.18

        total += mttr_bonus
        total -= burn_budget_penalty
        total -= anti_panic_penalty

        total = max(0.0, min(1.0, total))
        return IncidentCommanderReward(
            total=round(total, 4),
            uptime=round(uptime, 4),
            latency=round(latency, 4),
            sla=round(sla, 4),
            cost=round(cost, 4),
            recovery=round(recovery, 4),
            mttr_bonus=round(mttr_bonus, 4),
            burn_budget_penalty=round(burn_budget_penalty, 4),
            anti_panic_penalty=round(anti_panic_penalty, 4),
            latency_penalty=round(latency_penalty, 4),
            resource_waste_penalty=round(resource_waste_penalty, 4),
            incorrect_action_penalty=round(incorrect_action_penalty, 4),
            safety_penalty=round(safety_penalty, 4),
        )
