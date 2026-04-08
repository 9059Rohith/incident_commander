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


def _ground_truth_recovery(result: EpisodeResult) -> float:
    recovered = sum([result.db_recovered, result.auth_recovered, result.frontend_recovered])
    return recovered / 3.0


def grade_easy(result: EpisodeResult) -> float:
    recovery_truth = _ground_truth_recovery(result)
    score = (
        0.36 * result.uptime_score
        + 0.18 * result.latency_score
        + 0.16 * result.sla_score
        + 0.12 * result.recovery_score
        + 0.06 * result.cost_score
        + 0.07 * recovery_truth
        + 0.03 * result.verification_score
        + 0.02 * result.safe_ops_score
    )
    score += _discipline_term(result, low_escalation=1, high_escalation=3)
    score -= _failure_penalty(result)
    return _clamp(score)


def grade_medium(result: EpisodeResult) -> float:
    recovery_truth = _ground_truth_recovery(result)
    score = (
        0.32 * result.uptime_score
        + 0.18 * result.latency_score
        + 0.19 * result.sla_score
        + 0.11 * result.cost_score
        + 0.10 * result.recovery_score
        + 0.05 * result.root_cause_identification_rate
        + 0.03 * result.verification_score
        + 0.02 * recovery_truth
    )
    score += _discipline_term(result, low_escalation=2, high_escalation=4)
    score -= _failure_penalty(result)
    return _clamp(score)


def grade_hard(result: EpisodeResult) -> float:
    recovery_truth = _ground_truth_recovery(result)
    score = (
        0.26 * result.uptime_score
        + 0.16 * result.latency_score
        + 0.22 * result.sla_score
        + 0.12 * result.cost_score
        + 0.08 * result.recovery_score
        + 0.07 * result.root_cause_identification_rate
        + 0.04 * result.verification_score
        + 0.05 * recovery_truth
    )
    score += _discipline_term(result, low_escalation=2, high_escalation=5)
    score -= min(0.22, 0.035 * result.sla_breaches)
    score -= min(0.14, 0.09 * max(0.0, result.burn_budget_ratio - 1.0))
    score -= _failure_penalty(result)
    return _clamp(score)


def grade_longhaul(result: EpisodeResult) -> float:
    recovery_truth = _ground_truth_recovery(result)
    score = (
        0.22 * result.uptime_score
        + 0.15 * result.latency_score
        + 0.23 * result.sla_score
        + 0.18 * result.cost_score
        + 0.14 * result.recovery_score
        + 0.05 * result.root_cause_identification_rate
        + 0.03 * result.verification_score
        + 0.04 * recovery_truth
    )
    score += _discipline_term(result, low_escalation=3, high_escalation=6)
    score -= min(0.28, 0.04 * result.sla_breaches)
    score -= min(0.18, 0.11 * max(0.0, result.burn_budget_ratio - 1.0))
    score -= _failure_penalty(result)
    return _clamp(score)


def grade_blackout(result: EpisodeResult) -> float:
    # Blackout prioritizes SLA survival and sustained containment under prolonged pressure.
    recovery_truth = _ground_truth_recovery(result)
    score = (
        0.20 * result.uptime_score
        + 0.15 * result.latency_score
        + 0.25 * result.sla_score
        + 0.18 * result.cost_score
        + 0.12 * result.recovery_score
        + 0.05 * result.root_cause_identification_rate
        + 0.03 * result.verification_score
        + 0.02 * recovery_truth
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
