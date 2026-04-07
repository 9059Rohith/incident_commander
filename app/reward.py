from __future__ import annotations

from .models import ActiveIncident, IncidentCommanderAction, IncidentCommanderObservation, IncidentCommanderReward, TaskConfig


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
    ) -> IncidentCommanderReward:
        uptime = max(0.0, min(1.0, uptime_ratio))
        latency = max(0.0, min(1.0, 1.0 - obs.p95_latency / 700.0))
        sla = max(0.0, min(1.0, 1.0 - obs.sla_breaches / max(1, total_incidents + 2)))
        cost = max(0.0, min(1.0, 1.0 - cost_ratio))
        recovery = max(0.0, min(1.0, resolved_incidents / max(1, total_incidents)))

        total = (0.35 * uptime) + (0.20 * latency) + (0.20 * sla) + (0.15 * cost) + (0.10 * recovery)
        if action.action_type == "noop" and unresolved_critical > 0:
            total -= 0.12
        if obs.p95_latency > 350:
            total -= 0.08
        if obs.sla_breaches > 0:
            total -= min(0.12, 0.02 * obs.sla_breaches)

        total = max(0.0, min(1.0, total))
        return IncidentCommanderReward(
            total=round(total, 4),
            uptime=round(uptime, 4),
            latency=round(latency, 4),
            sla=round(sla, 4),
            cost=round(cost, 4),
            recovery=round(recovery, 4),
        )
