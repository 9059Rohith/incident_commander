from __future__ import annotations

from app.models import EpisodeResult


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, score))


def grade_easy(result: EpisodeResult) -> float:
    return _clamp(0.45 * result.uptime_score + 0.25 * result.latency_score + 0.15 * result.sla_score + 0.15 * result.recovery_score)


def grade_medium(result: EpisodeResult) -> float:
    return _clamp(0.40 * result.uptime_score + 0.20 * result.latency_score + 0.20 * result.sla_score + 0.10 * result.cost_score + 0.10 * result.recovery_score)


def grade_hard(result: EpisodeResult) -> float:
    return _clamp(0.35 * result.uptime_score + 0.20 * result.latency_score + 0.20 * result.sla_score + 0.15 * result.cost_score + 0.10 * result.recovery_score)


def grade_longhaul(result: EpisodeResult) -> float:
    return _clamp(0.30 * result.uptime_score + 0.20 * result.latency_score + 0.20 * result.sla_score + 0.15 * result.cost_score + 0.15 * result.recovery_score)


GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
    "longhaul": grade_longhaul,
}
