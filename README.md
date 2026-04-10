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
short_description: Simulated incident response benchmark for AI platform operations
---

# Incident Commander OpenEnv

A CPU-only OpenEnv benchmark where an LLM agent acts as the incident commander for a simulated AI platform. The agent investigates masked outages, chooses structured remediation actions, and is scored on recovery, latency, cost, and operational discipline.

[![openenv](https://img.shields.io/badge/openenv-compatible-blue)](https://meta-pytorch.org/OpenEnv/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-required-informational)](https://www.docker.com/)
[![GPU Not Required](https://img.shields.io/badge/GPU-not%20required-success)](https://meta-pytorch.org/OpenEnv/)

Quick links: [Judge Guide](JUDGES_GUIDE.md) | [Submission Brief](HACKATHON_SUBMISSION_BRIEF.md) | [Pre-Submission Checklist](PRE_SUBMISSION_CHECKLIST.md)

## Overview

Incident Commander models the on-call workflow for a small AI platform with three coupled services: `frontend`, `auth`, and `db`. Each episode is a black-box incident response problem. The agent observes the current state, inspects logs or metrics, takes a typed action, and tries to recover service without burning budget or causing avoidable escalation.

The environment is designed to be deterministic under fixed seeds, easy to validate locally and in CI, compatible with Hugging Face Spaces and Docker, and transparent enough for judges to inspect replays, reports, and baseline comparisons.

## Tasks

The project ships with five canonical tasks and deterministic graders:

| Task | Steps | Focus |
|---|---:|---|
| `easy` | 30 | Trace frontend failures to root cause and restore service health |
| `medium` | 40 | Investigate dependency failures across frontend/auth/db under pressure |
| `hard` | 50 | Recover from config drift, regional network issues, and high-load race conditions with verification |
| `longhaul` | 60 | Handle mixed outages with black-box telemetry and budget-aware remediation |
| `blackout` | 70 | Survive thundering-herd outages with safe, efficient SRE operations and resilient multi-region recovery |

All task scores are normalized to `[0.0, 1.0]`, and the success threshold is `0.5`.

## Control Loop

The environment follows a simple but operationally realistic loop:

1. Call `/reset` for a chosen `task_id`.
2. Inspect the returned observation.
3. Choose a single structured action.
4. Apply the action with `/step`.
5. Monitor state, reward, and incident progress.
6. Finish when the episode ends, the task is recovered, or a failure condition is reached.

### Observation Space

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

### Action Space

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

## Reward and Grading

Reward is dense and inspectable. It combines uptime, latency, SLA compliance, cost, recovery, MTTR bonus, burn-budget penalty, anti-panic penalty, and safety penalties.

Each task has a deterministic grader in `server.tasks` that returns a score in `[0.0, 1.0]` and rewards partial progress instead of only binary success.

## API Surface

The main endpoints are:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /grade`
- `GET /tasks`
- `GET /health`
- `GET /metrics`
- `GET /report`
- `GET /visualize`
- `GET /baseline`
- `GET /benchmark_matrix`
- `GET /replay`
- `GET /evaluation_report`
- `GET /judge_pack`
- `GET /showcase`

The `/metrics` endpoint supports `include_trace=true|false`. The `/replay` endpoint exports a deterministic full trajectory for a fixed task, seed, and policy. The `/evaluation_report` endpoint returns compact benchmark analytics across all tasks. The `/judge_pack` and `/showcase` endpoints are judge-facing helpers for fast inspection.

## Quick Start

Local run:

```bash
pip install -e .
python -m uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Docker run:

```bash
docker build -t incident-commander -f server/Dockerfile .
docker run -p 7860:7860 incident-commander
```

Validation:

```bash
python -m openenv.cli validate
```

```bash
bash scripts/validate-submission.sh https://your-space.hf.space .
```

PowerShell:

```powershell
./scripts/validate-submission.ps1 -PingUrl "https://your-space.hf.space" -RepoDir "."
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
- `ENV_URL`
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

## Baseline Behavior

Use the baseline comparison script to reproduce the policy table:

```bash
python greedy_baseline.py
```

It prints a summary across fixed seeds so judges can compare noop, reactive, and reasoning policies under the same episode distribution.

## Submission Checklist

- HF Space responds at `/health`
- `openenv.yaml` includes metadata and task definitions
- `docker build` works from the repo root
- `inference.py` runs from the repo root
- `/reset`, `/step`, `/state`, `/grade`, `/metrics`, and `/visualize` are implemented
- `replay`, `evaluation_report`, `judge_pack`, and `showcase` are available for deeper inspection

For the full validator flow, use [PRE_SUBMISSION_CHECKLIST.md](PRE_SUBMISSION_CHECKLIST.md).

## Deployment

The repository is Dockerized and ready for Hugging Face Spaces deployment with the `openenv` tag. The live Space URL is the submission target; the `huggingface.co/spaces/...` page is the repo view.

## Notes for Judges

- The project focuses on a real operational domain: incident response for an AI platform.
- The task ladder is deterministic and reproducible.
- The API is designed to be inspectable, replayable, and easy to validate.
- The benchmark exposes enough structure to compare policies meaningfully without requiring custom tooling.
- The `longhaul` task adds a stronger delayed-credit signal without making the environment brittle.

## Repository Structure

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
