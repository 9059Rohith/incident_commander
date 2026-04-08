from __future__ import annotations

from app.models import EpisodeResult


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, score))


def _failure_penalty(result: EpisodeResult) -> float:
    penalty = 0.0
    if result.ended_by_sla_failure:
        penalty += 0.16
    if result.ended_by_budget_failure:
        penalty += 0.14
    return penalty


def _discipline_term(result: EpisodeResult, low_escalation: int, high_escalation: int) -> float:
    discipline = 0.08 * result.action_discipline_score
    if result.escalations_used > high_escalation:
        discipline -= 0.07
    elif result.escalations_used <= low_escalation:
        discipline += 0.03
    return discipline


def grade_easy(result: EpisodeResult) -> float:
    score = (
        0.44 * result.uptime_score
        + 0.24 * result.latency_score
        + 0.15 * result.sla_score
        + 0.14 * result.recovery_score
        + 0.03 * result.cost_score
    )
    score += _discipline_term(result, low_escalation=1, high_escalation=3)
    score -= _failure_penalty(result)
    return _clamp(score)


def grade_medium(result: EpisodeResult) -> float:
    score = (
        0.38 * result.uptime_score
        + 0.20 * result.latency_score
        + 0.20 * result.sla_score
        + 0.11 * result.cost_score
        + 0.11 * result.recovery_score
    )
    score += _discipline_term(result, low_escalation=2, high_escalation=4)
    score -= _failure_penalty(result)
    return _clamp(score)


def grade_hard(result: EpisodeResult) -> float:
    score = (
        0.30 * result.uptime_score
        + 0.18 * result.latency_score
        + 0.22 * result.sla_score
        + 0.14 * result.cost_score
        + 0.08 * result.recovery_score
    )
    score += _discipline_term(result, low_escalation=2, high_escalation=5)
    score -= min(0.22, 0.035 * result.sla_breaches)
    score -= min(0.14, 0.09 * max(0.0, result.burn_budget_ratio - 1.0))
    score -= _failure_penalty(result)
    return _clamp(score)


def grade_longhaul(result: EpisodeResult) -> float:
    score = (
        0.24 * result.uptime_score
        + 0.17 * result.latency_score
        + 0.24 * result.sla_score
        + 0.18 * result.cost_score
        + 0.17 * result.recovery_score
    )
    score += _discipline_term(result, low_escalation=3, high_escalation=6)
    score -= min(0.28, 0.04 * result.sla_breaches)
    score -= min(0.18, 0.11 * max(0.0, result.burn_budget_ratio - 1.0))
    score -= _failure_penalty(result)
    return _clamp(score)


def grade_blackout(result: EpisodeResult) -> float:
    # Blackout prioritizes SLA survival and sustained containment under prolonged pressure.
    score = (
        0.22 * result.uptime_score
        + 0.17 * result.latency_score
        + 0.27 * result.sla_score
        + 0.18 * result.cost_score
        + 0.16 * result.recovery_score
    )
    score += _discipline_term(result, low_escalation=3, high_escalation=7)
    score -= min(0.32, 0.045 * result.sla_breaches)
    score -= min(0.22, 0.13 * max(0.0, result.burn_budget_ratio - 1.0))
    score -= _failure_penalty(result)
    return _clamp(score)


GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
    "longhaul": grade_longhaul,
    "blackout": grade_blackout,
}
