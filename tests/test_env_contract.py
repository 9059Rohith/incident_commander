from __future__ import annotations

from fastapi.testclient import TestClient

from app.env import IncidentCommanderEnv, TASKS
from app.main import app
from app.models import IncidentCommanderAction, TaskConfig
from server.tasks import GRADERS


def test_reset_is_deterministic_for_same_seed() -> None:
    env = IncidentCommanderEnv(TASKS["hard"], seed=123)
    obs_a = env.reset(seed=123)
    obs_b = env.reset(seed=123)
    assert obs_a.model_dump() == obs_b.model_dump()


def test_grade_range_for_all_tasks() -> None:
    for task_id, config in TASKS.items():
        env = IncidentCommanderEnv(config, seed=42)
        obs = env.reset(seed=42)
        for _ in range(3):
            obs, _, done, _ = env.step(IncidentCommanderAction(action_type="noop"))
            if done:
                break
        score = float(GRADERS[task_id](env.build_episode_result()))
        assert 0.0 <= score <= 1.0


def test_scheduled_incident_injection_increases_pressure() -> None:
    task = TaskConfig(
        task_id="easy",
        max_steps=6,
        base_traffic=1.0,
        peak_traffic=1.3,
        incident_schedule=[1],
        cost_budget=0.5,
        max_sla_breaches=10,
        observation_noise=0.0,
        scenario_mix=["resource_exhaustion"],
    )
    env = IncidentCommanderEnv(task, seed=11)
    env.reset(seed=11)
    before = env.scenario.noisy_neighbor_io
    env.step(IncidentCommanderAction(action_type="noop"))
    env.step(IncidentCommanderAction(action_type="noop"))
    after = env.scenario.noisy_neighbor_io
    assert after - before >= 0.20


def test_health_endpoint_reports_runtime_metadata() -> None:
    client = TestClient(app)
    response = client.get("/health")
    response.raise_for_status()
    payload = response.json()
    assert payload["status"] == "ok"
    assert isinstance(payload.get("supported_tasks"), list)
    assert "easy" in payload["supported_tasks"]


def test_replay_endpoint_is_deterministic_for_same_seed() -> None:
    client = TestClient(app)
    first = client.get("/replay", params={"task_id": "easy", "seed": 123, "policy": "baseline"})
    second = client.get("/replay", params={"task_id": "easy", "seed": 123, "policy": "baseline"})
    first.raise_for_status()
    second.raise_for_status()

    payload_a = first.json()
    payload_b = second.json()
    assert payload_a["score"] == payload_b["score"]
    assert payload_a["failure_taxonomy"] == payload_b["failure_taxonomy"]
    assert payload_a["incident_narrative"]["step_count"] == payload_b["incident_narrative"]["step_count"]


def test_evaluation_report_endpoint_returns_summary() -> None:
    client = TestClient(app)
    response = client.get(
        "/evaluation_report",
        params={"policy": "baseline", "episodes_per_task": 2, "seed_start": 42},
    )
    response.raise_for_status()
    payload = response.json()
    assert payload["policy"] == "baseline"
    assert payload["episodes_per_task"] == 2
    assert "summary" in payload
    assert "per_task" in payload
    assert "easy" in payload["per_task"]


def test_judge_pack_and_showcase_endpoints_exist() -> None:
    client = TestClient(app)
    judge_pack = client.get("/judge_pack")
    showcase = client.get("/showcase")

    judge_pack.raise_for_status()
    showcase.raise_for_status()

    payload = judge_pack.json()
    assert payload["status"] == "submission_ready"
    assert "/showcase" in payload["core_endpoints"]
    assert "judge_surface" in payload
    assert showcase.headers["content-type"].startswith("text/html")


def test_forensic_and_quickstart_endpoints_exist() -> None:
    client = TestClient(app)
    forensic = client.get("/forensic_audit", params={"task_id": "hard", "seed": 42, "policy": "baseline"})
    quickstart = client.get("/judge_quickstart")

    forensic.raise_for_status()
    quickstart.raise_for_status()

    forensic_payload = forensic.json()
    quickstart_payload = quickstart.json()
    assert "diagnostics" in forensic_payload
    assert "failure_taxonomy" in forensic_payload
    assert isinstance(quickstart_payload.get("steps", []), list)
    assert quickstart_payload.get("expected_signals", {}).get("counterfactual_diagnostics", False) is True
