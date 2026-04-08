from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel


class ServiceState(BaseModel):
    name: str
    healthy: bool = True
    instances: int = 2
    desired_instances: int = 2
    version: str = "v1"
    rollback_version: str = "v0"
    queue_depth: int = 0
    p95_latency: float = 80.0
    error_rate: float = 0.0
    cpu_utilization: float = 35.0
    memory_utilization: float = 35.0
    quarantined: bool = False
    rolling_back_steps: int = 0
    scale_cooldown_steps: int = 0
    redirected_fraction: float = 0.0
    spot_instances: int = 0
    observed_p95_latency: Optional[float] = None
    observed_error_rate: Optional[float] = None
    metric_staleness_steps: int = 0
    last_action_result: str = "idle"


class ActiveIncident(BaseModel):
    incident_id: str
    incident_type: Literal[
        "traffic_spike",
        "bad_deploy",
        "node_failure",
        "database_lock",
        "cache_thrash",
        "cascade",
    ]
    service: str
    severity: Literal["low", "medium", "high", "critical"]
    age_steps: int = 0
    resolved: bool = False
    resolution_timer: int = 0


class IncidentCommanderObservation(BaseModel):
    services: Dict[str, ServiceState]
    active_incidents: List[ActiveIncident]
    step: int
    step_budget: int
    traffic_level: float
    uptime: float
    p95_latency: float
    sla_breaches: int
    cost_per_step: float
    last_action_result: str
    phase: str


class IncidentCommanderAction(BaseModel):
    action_type: Literal[
        "scale_service",
        "reroute_traffic",
        "rollback_deploy",
        "quarantine_service",
        "page_human",
        "noop",
    ]
    target_service: Optional[str] = None
    delta_instances: int = 0
    request_fraction: float = 0.0
    target_version: Optional[str] = None
    fallback_service: Optional[str] = None
    note: Optional[str] = None


class IncidentCommanderReward(BaseModel):
    total: float
    uptime: float
    latency: float
    sla: float
    cost: float
    recovery: float
    mttr_bonus: float = 0.0
    burn_budget_penalty: float = 0.0
    anti_panic_penalty: float = 0.0


class TaskConfig(BaseModel):
    task_id: Literal["easy", "medium", "hard", "longhaul", "blackout"]
    max_steps: int
    base_traffic: float
    peak_traffic: float
    incident_schedule: List[int]
    cost_budget: float
    spike_multiplier: float = 1.0
    max_sla_breaches: int = 10
    observation_noise: float = 0.0
    spot_disruption_chance: float = 0.0
    memory_leak_rate: float = 0.0
    thundering_herd: bool = False


class EpisodeResult(BaseModel):
    task_id: str
    uptime_score: float
    latency_score: float
    sla_score: float
    cost_score: float
    recovery_score: float
    incident_clearance_rate: float
    ended_by_sla_failure: bool
    ended_by_budget_failure: bool = False
    action_discipline_score: float = 1.0
    escalations_used: int = 0
    sla_breaches: int = 0
    burn_budget_ratio: float = 0.0
    total_score: float
