# Incident Commander Judge Guide

This guide is optimized for fast technical review.

## 1. What this project proves

Incident Commander evaluates whether an agent can perform realistic, multi-step SRE decision-making under uncertainty:

- partial and noisy observability,
- coupled service failures (`frontend -> auth -> db`),
- delayed reward effects and long-horizon trade-offs,
- cost-aware reliability decisions under burn-budget pressure.

This is not a code-generation benchmark. It is an operations-reasoning benchmark.

## 2. Fast validation path (under 3 minutes)

1. Verify hosted API:
   - `GET /health`
   - `POST /reset`
   - `GET /tasks`
2. Run local package checks:
   - `python -m pytest -q`
   - `python -m openenv.cli validate`
3. Run full validator:
   - Linux/macOS: `bash scripts/validate-submission.sh <space_url> .`
   - Windows: `./scripts/validate-submission.ps1 -PingUrl "<space_url>" -RepoDir "."`

Expected outcome: all checks pass.

## 3. Why this should score well

- Task ladder has increasing complexity (`easy` to `blackout`) and is hard to game.
- Reward is decomposed and auditable (uptime, latency, SLA, cost, recovery, discipline).
- Anti-exploit controls are explicit (passive-loop penalty, unsafe-action penalties, budget cutoff).
- Endpoints include judge-facing reports and benchmark matrix for quick comparative analysis.

## 4. Demonstrated baseline signal

Run:

```bash
python greedy_baseline.py
```

The baseline table demonstrates meaningful separation between reactive and phase-aware reasoning, especially on `blackout`, where shallow heuristics collapse.

## 5. Suggested review criteria

Use this rubric for a fair technical review:

- Realism: Does the environment model production-style failure coupling and operational constraints?
- Learnability: Do task and reward signals support stable policy improvement?
- Robustness: Are exploit paths and brittle shortcuts penalized?
- Reproducibility: Can an external reviewer run checks and reproduce reported behavior quickly?
- Clarity: Is behavior interpretable through `/metrics`, `/report`, and timeline artifacts?

## 6. Repo landmarks

- API app: `app/main.py`
- Core simulation: `app/env.py`
- Reward logic: `app/reward.py`
- Graders: `server/tasks.py`
- Baseline policies: `greedy_baseline.py`
- Inference runner: `inference.py`
- Full checklist: `PRE_SUBMISSION_CHECKLIST.md`
