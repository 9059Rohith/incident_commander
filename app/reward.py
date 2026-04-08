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
    ) -> IncidentCommanderReward:
        uptime = max(0.0, min(1.0, uptime_ratio))
        latency = max(0.0, min(1.0, 1.0 - obs.p95_latency / 700.0))
        sla = max(0.0, min(1.0, 1.0 - obs.sla_breaches / max(1, total_incidents + 2)))
        cost = max(0.0, min(1.0, 1.0 - cost_ratio))
        recovery = max(0.0, min(1.0, resolved_incidents / max(1, total_incidents)))

        total = (0.34 * uptime) + (0.19 * latency) + (0.21 * sla) + (0.14 * cost) + (0.12 * recovery)
        if action.action_type == "noop" and unresolved_critical > 0:
            total -= 0.12
        if obs.p95_latency > 350:
            total -= 0.08
        if obs.sla_breaches > 0:
            total -= min(0.12, 0.02 * obs.sla_breaches)
        if unforced_page:
            total -= 0.08
        if action_streak >= 5 and action.action_type in {"noop", "page_human"}:
            total -= 0.06

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
        )
