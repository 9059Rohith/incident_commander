# Hackathon Submission Brief: Incident Commander OpenEnv

## Elevator pitch

Incident Commander is a realistic RL benchmark where an agent acts as an on-call SRE and must restore reliability in a noisy, stateful, failure-coupled AI platform.

## Problem

Most agent benchmarks test code synthesis or single-turn QA. They do not measure operational decision quality under outage pressure, delayed consequences, and cross-service coupling.

## What is new

- Black-box incident investigation loop, not root-cause disclosure.
- Coupled service topology (`frontend -> auth -> db`) with cascade dynamics.
- Long-horizon failure phases including `longhaul` and `blackout`.
- Cost-aware reliability trade-offs with burn-budget penalties.
- Action discipline incentives that penalize unstable panic behavior.

## Why this matters

This benchmark measures whether a policy can:

- reason under partial observability,
- sequence multi-step remediation safely,
- protect SLA while managing finite budget,
- avoid reward-hacking shortcuts.

## Evidence package

- Deterministic task + grader setup across 5 tasks.
- Endpoint-level smoke tests and OpenEnv validation.
- Dockerized deployment path for reproducible review.
- Baseline comparison showing meaningful difficulty separation.

## Reviewer fast path

1. Run `/health`, `/reset`, `/tasks` on hosted Space.
2. Run `python -m pytest -q`.
3. Run `python -m openenv.cli validate`.
4. Optionally run `scripts/validate-submission.sh` or `scripts/validate-submission.ps1`.

## Project links

- Judge guide: `JUDGES_GUIDE.md`
- Checklist: `PRE_SUBMISSION_CHECKLIST.md`
- Technical details: `README.md`
