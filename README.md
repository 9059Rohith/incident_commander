---
title: Incident Commander OpenEnv
emoji: "🐢"
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - llm-ops
  - inference
  - scheduling
  - sre
  - agent
  - rl-environment
license: mit
short_description: LLM incident response benchmark for AI platforms
---

# Incident Commander OpenEnv

> A real-world RL environment where an agent acts as the on-call incident commander for a simulated AI platform.

[![openenv](https://img.shields.io/badge/openenv-compatible-blue)](https://meta-pytorch.org/OpenEnv/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-required-informational)](https://www.docker.com/)
[![CPU Only](https://img.shields.io/badge/GPU-not%20required-success)](https://meta-pytorch.org/OpenEnv/)

Quick links: [Judge Guide](JUDGES_GUIDE.md) | [Submission Brief](HACKATHON_SUBMISSION_BRIEF.md) | [Pre-Submission Checklist](PRE_SUBMISSION_CHECKLIST.md)

---

## Overview

Incident Commander OpenEnv is a CPU-only RL benchmark for real-world incident response. An agent acts as the on-call commander for a simulated AI platform and must diagnose, remediate, and verify service outages under partial observability, changing load, and budget constraints.

The environment is designed for hackathon submission quality: typed models, deterministic graders, reproducible baseline runs, Dockerized deployment, and judge-friendly inspection endpoints.

## What it simulates

The simulated platform has three coupled services:

- `frontend`: customer-facing API and web edge
- `auth`: identity and token service
- `db`: persistent backing service for auth/session data

The agent sees noisy operational telemetry, masked incidents, and a timeline of recent actions. Root causes are intentionally hidden until the agent investigates logs, metrics, and connectivity.

## Task ladder

The project includes five deterministic tasks with increasing difficulty:

| Task | Steps | Focus |
|---|---:|---|
| `easy` | 30 | Recover a degraded service quickly |
| `medium` | 40 | Calibrate rollback and scaling under pressure |
| `hard` | 50 | Resolve cascading failures across frontend/auth/db |
| `longhaul` | 60 | Handle delayed-credit, long-horizon incidents |
| `blackout` | 70 | Survive thundering-herd pressure while protecting budget |

## Environment design

The control loop is intentionally simple but operationally realistic:

1. Observe the current incident state.
2. Choose an action from the typed toolbelt.
3. Apply action effects and scheduled scenario shocks.
4. Simulate service load, latency, uptime, and cost.
5. Compute decomposed reward and update incident state.
6. End the episode on max steps, SLA failure, budget failure, or full recovery.

### Observation space

Each step returns a structured snapshot containing:

| Field | Description |
|---|---|
| `services` | Per-service health, instances, latency, error rate, and noisy observations |
| `active_incidents` | Masked incident cards and severity |
| `step`, `step_budget` | Current position and episode horizon |
| `traffic_level` | Demand pressure for the current step |
| `uptime`, `p95_latency`, `sla_breaches` | Core operational metrics |
| `cost_per_step` | Accumulated operating cost |
| `last_action_result` | Result of the previous action |
| `phase` | Current episode phase |
| `symptoms`, `terminal_output`, `investigation_log`, `live_timeline` | Human-readable incident context |
| `available_actions` | Explicit action list for the policy |

### Action space

The environment supports these actions:

| Action | Use |
|---|---|
| `get_metrics` | Inspect telemetry |
| `list_processes` | Inspect process state |
| `read_last_n_logs` | Inspect logs |
| `check_network_connectivity` | Probe service paths |
| `failover_database` | Fail over DB during regional incidents |
| `restart_service` | Restart a service pool |
| `rollback_deployment` | Roll back a service release |
| `scale_up_replicas` | Add capacity |
| `edit_config_line` | Patch runtime config |
| `run_healthcheck` | Verify remediation |
| `ask_developer` | Request a human hint |
| `load_test` | Reproduce under load |
| `run_command` | Simulated operational command |
| `noop` | Do nothing |

## Reward and grading

Reward is dense and inspectable. It combines uptime, latency, SLA compliance, cost, recovery, MTTR bonus, burn-budget penalty, anti-panic penalty, and safety penalties.

Each task has a deterministic grader that returns a score in `[0.0, 1.0]` and rewards partial progress instead of only binary success.

## API surface

The main endpoints are:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /grade`
- `GET /tasks`
- `GET /health`
- `GET /metrics`
- `GET /report`
- `GET /benchmark_matrix`
- `GET /replay`
- `GET /evaluation_report`
- `GET /judge_pack`
- `GET /showcase`

The `/metrics` endpoint supports `include_trace=true|false`. The `/replay` endpoint exports a deterministic full trajectory for a fixed task, seed, and policy. The `/evaluation_report` endpoint returns compact benchmark analytics across all tasks. The `/judge_pack` and `/showcase` endpoints are judge-facing helpers for fast inspection.

## Quick start

Local run:

```bash
pip install -e .
python -m uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Validation:

```bash
bash scripts/validate-submission.sh https://your-space.hf.space .
```

PowerShell:

```powershell
./scripts/validate-submission.ps1 -PingUrl "https://your-space.hf.space" -RepoDir "."
```

OpenEnv validation:

```bash
python -m openenv.cli validate
```

Local smoke test:

```bash
python scripts/test-local.py
```

## Inference

Environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- optional `LOCAL_IMAGE_NAME`

The inference script uses the OpenAI client and emits strict `[START]`, `[STEP]`, and `[END]` log lines. If the model call is unavailable, it falls back to a deterministic heuristic controller instead of a noop policy.

Linux/macOS:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token_here"
export ENV_URL="http://localhost:7860"
python inference.py
```

PowerShell:

```powershell
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
$env:HF_TOKEN = "hf_your_token_here"
$env:ENV_URL = "http://localhost:7860"
python inference.py
```

## Baseline behavior

Use the baseline comparison script to reproduce the policy table:

```bash
python greedy_baseline.py
```

It prints a CSV-style summary across fixed seeds so judges can compare noop, reactive, and reasoning policies under the same episode distribution.

## Submission checklist

- HF Space responds at `/health`
- `openenv.yaml` includes metadata and task definitions
- `docker build` works from the repo root
- `inference.py` runs from the repo root
- `/reset`, `/step`, `/state`, `/grade`, `/metrics`, and `/visualize` are implemented
- `replay`, `evaluation_report`, `judge_pack`, and `showcase` are available for deeper inspection

For the full validator flow, use [PRE_SUBMISSION_CHECKLIST.md](PRE_SUBMISSION_CHECKLIST.md).

## Deployment

The repository is Dockerized and ready for Hugging Face Spaces deployment with the `openenv` tag. The live Space URL is the submission target; the `huggingface.co/spaces/...` page is the repo view.

## Notes for judges

- The project focuses on a real operational domain: incident response for an AI platform.
- The task ladder is deterministic and reproducible.
- The API is designed to be inspectable, replayable, and easy to validate.
- The benchmark exposes enough structure to compare policies meaningfully without requiring custom tooling.
- The longhaul task adds a stronger delayed-credit signal without making the environment brittle.

## Repository structure

```text
incident-commander/
├── Dockerfile
├── README.md
├── inference.py
├── greedy_baseline.py
├── openenv.yaml
├── pyproject.toml
├── requirements.txt
├── scripts/
│   ├── validate-submission.sh
│   └── test-local.py
├── tests/
│   └── test_env_contract.py
├── app/
│   ├── main.py
│   ├── env.py
│   ├── reward.py
│   └── models.py
└── server/
    ├── app.py
    └── tasks.py
```

## License

MIT
