from __future__ import annotations

from app.env import IncidentCommanderEnv, TASKS
from app.models import IncidentCommanderAction
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
