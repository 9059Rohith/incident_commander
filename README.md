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
short_description: Incident response RL benchmark for AI platform ops
---

# Incident Commander OpenEnv

Incident Commander is a CPU-only OpenEnv benchmark where an LLM agent acts as the incident commander for a simulated AI platform. The agent investigates masked outages, chooses structured remediation actions, and is scored on recovery, latency, cost, and operational discipline.

[![openenv](https://img.shields.io/badge/openenv-compatible-blue)](https://meta-pytorch.org/OpenEnv/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-required-informational)](https://www.docker.com/)
[![GPU Not Required](https://img.shields.io/badge/GPU-not%20required-success)](https://meta-pytorch.org/OpenEnv/)

## Why This Benchmark Exists

Incident Commander is designed to be more than a thin API wrapper or a single-step troubleshooting task. It asks a policy to manage a stateful incident over time, make decisions under partial observability, and balance recovery speed against operational cost and safety.

Compared with a simpler scheduling benchmark, the advantage here is realism in the control loop:

- The agent does not get the answer directly.
- Actions have persistent consequences across steps.
- Recovery often requires diagnosis before remediation.
- The best action can change as the incident evolves.
- The reward is shaped so that progress matters, not just final success.

That combination makes the environment useful for reinforcement learning, agent evaluation, and judge-facing benchmark design.

## Core Simulation

The simulated platform has three coupled services:

- `frontend`: customer-facing entry point and usual first symptom surface.
- `auth`: identity and token validation service.
- `db`: persistence layer where latency and dependency issues often originate.

Each episode is a black-box incident drill. The agent observes structured telemetry, masked incident context, and recent action history. It must infer root cause from symptoms rather than reading a direct label.

The environment is intentionally small enough to run locally, but the coupling between services creates realistic decision pressure. A fix that helps one layer can destabilize another, and that makes the policy's sequence of actions more important than any single move.

## Task Ladder

The project ships with five canonical tasks and deterministic graders:

| Task | Steps | Focus |
|---|---:|---|
| `easy` | 30 | Trace frontend failures to root cause and restore service health |
| `medium` | 40 | Investigate dependency failures across frontend/auth/db under pressure |
| `hard` | 50 | Recover from config drift, regional network issues, and high-load race conditions with verification |
| `longhaul` | 60 | Handle mixed outages with black-box telemetry and budget-aware remediation |
| `blackout` | 70 | Survive thundering-herd outages with safe, efficient SRE operations and resilient multi-region recovery |

All task scores are normalized to `[0.0, 1.0]`, and the success threshold is `0.5`.

The tasks are intentionally not equivalent in how they reward action:

- `easy` verifies that the environment, action model, and grader wiring work.
- `medium` introduces recovery sequencing and makes diagnosis matter.
- `hard` adds mixed failure modes, so the policy must balance throughput, verification, and cost.
- `longhaul` rewards policies that stay effective over a longer horizon instead of optimizing only the current step.
- `blackout` is the strongest stress test, with prolonged pressure and a bigger need for disciplined remediation.

## Control Loop

The environment follows a simple but operationally realistic loop:

1. Call `/reset` for a chosen `task_id`.
2. Inspect the returned observation.
3. Choose a single structured action.
4. Apply the action with `/step`.
5. Monitor state, reward, and incident progress.
6. Finish when the episode ends, the task is recovered, or a failure condition is reached.

Actions have persistent effects. That means the benchmark is not just about reacting to the current observation; it is about building a plan over multiple steps and carrying that plan through a changing incident state.

### Example Interaction

The API is designed to be easy to call from a script or agent loop:

```bash
curl -X POST "http://localhost:7860/reset?task_id=easy&seed=42"
curl -X POST "http://localhost:7860/step?task_id=easy" \
  -H "Content-Type: application/json" \
  -d '{"action_type":"get_metrics","target_service":"frontend","delta_instances":0,"fallback_service":null,"config_key":null,"config_value":null,"n_lines":0,"question":null,"note":"inspect first"}'
curl "http://localhost:7860/grade?task_id=easy"
```

The exact action schema is defined by `app/models.py`, and invalid combinations are rejected rather than silently accepted.

## Observation Space

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

This is the main reason the environment is useful: the policy has to reason across multiple signals at once. A good decision is rarely based on one field only. The observation space is structured enough for agentic control, but still incomplete enough to force inference.

## Action Space

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

The environment also exposes a `visualize` endpoint that returns a compact ASCII dashboard for quick manual inspection. That makes it easier to compare the current service state against the task description during debugging.

## Reward and Grading

Reward is dense and inspectable. It combines uptime, latency, SLA compliance, cost, recovery, MTTR bonus, burn-budget penalty, anti-panic penalty, and safety penalties.

Each task has a deterministic grader in `server.tasks` that returns a score in `[0.0, 1.0]` and rewards partial progress instead of only binary success.

The grading design matters because it gives the agent a real learning signal:

- Diagnostic progress is valuable.
- Recovery progress is valuable.
- Wasteful or contradictory behavior is discouraged.
- Fast fixes are better than slow ones, but not at the expense of correctness.

That makes the benchmark more useful for RL than a pass/fail workflow alone.

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

Useful endpoint notes:

- `/reset` initializes a task context and returns the first observation.
- `/step` applies exactly one action to the active task context.
- `/state` returns the current state without advancing the episode.
- `/grade` evaluates the current episode state with the task-specific grader.
- `/baseline` and `/benchmark_matrix` run the built-in baseline controller for quick comparison.
- `/report` combines grade, metrics, and unresolved incident context in one payload.
- `/replay` and `/evaluation_report` are useful when you want deterministic inspection artifacts rather than a single score.
- `/judge_pack` and `/showcase` are judge-facing inspection tools for fast review.

## Judge Auditability

Incident Commander is built to be auditable, not just scoreable.

- Use `/metrics?include_trace=true` to inspect step-by-step reward decomposition.
- Use `/visualize` for a compact operator dashboard view of current state.
- Use `/replay` for deterministic trajectory exports by task, seed, and policy.
- Use `/evaluation_report` for aggregate benchmarking across tasks.
- Use `/judge_pack` for a one-call evaluation snapshot.

For a fast live walkthrough, follow [DEMOSCRIPT.md](DEMOSCRIPT.md).

## Why It Is Stronger Than a Simple Scheduling Benchmark

Incident Commander is not trying to model hardware placement or resource packing. It models a human operational loop: diagnose, prioritize, remediate, verify, and explain.

That makes the benchmark stronger in a few ways:

- The policy must handle partial observability.
- The environment changes as the agent acts.
- The reward is multi-objective rather than binary.
- The tasks require multi-step planning instead of a fixed recipe.
- The inspection endpoints make it easy for judges to audit what happened.

This is a good fit for models that are meant to act as operators, not just as classifiers.

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

If you are validating a submission bundle, run the local smoke test first, then the OpenEnv validator, then the submission-specific validator script. That sequence catches most environment and API mismatch issues before deployment.

## Inference

Environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `ENV_URL`
- optional `LOCAL_IMAGE_NAME`

The inference script uses the OpenAI client and emits strict `[START]`, `[STEP]`, and `[END]` log lines. If the model call is unavailable, it falls back to a deterministic heuristic controller instead of a noop policy.

This fallback is deliberate. It keeps the script usable in restricted environments and gives judges something meaningful to compare against even when an external LLM is not available.

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

For quick sanity checks, the built-in baseline is the fastest way to confirm that task wiring, grading, and episode transitions still behave as expected after a change.

## Submission Checklist

- HF Space responds at `/health`
- `openenv.yaml` includes metadata and task definitions
- `docker build` works from the repo root
- `inference.py` runs from the repo root
- `/reset`, `/step`, `/state`, `/grade`, `/metrics`, and `/visualize` are implemented
- `replay`, `evaluation_report`, `judge_pack`, and `showcase` are available for deeper inspection

## Notes for Judges

- The project focuses on a real operational domain: incident response for an AI platform.
- The task ladder is deterministic and reproducible.
- The API is designed to be inspectable, replayable, and easy to validate.
- The benchmark exposes enough structure to compare policies meaningfully without requiring custom tooling.
- The `longhaul` task adds a stronger delayed-credit signal without making the environment brittle.
- The repository intentionally keeps the documentation surface focused so the README stays the source of truth.

## Troubleshooting

If something looks off during local testing, these checks usually find the issue quickly:

- Confirm the server is running on port `7860` before calling `/reset` or `/step`.
- Make sure you are using `task_id`, not `task_name`, when calling the API.
- If `/grade` fails, reset the episode again and replay the same task from a clean state.
- If the inference script falls back to heuristics, verify that `HF_TOKEN` and `API_BASE_URL` are set.
- If the validator complains about endpoint coverage, check `app/main.py` and `scripts/test-local.py` for the expected contract.

## Development Notes

The repository keeps the main implementation in `app/` and the grader logic in `server/tasks.py`. That separation is intentional: it keeps the environment mechanics close to the action model, while the task scoring stays easy to inspect.

If you change the environment behavior, update the README, the validator expectations, and any smoke tests that encode the public contract. That keeps the documentation, runtime behavior, and evaluation flow aligned.

## Repository Structure

```text
incident-commander/
├── Dockerfile
├── DEMOSCRIPT.md
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

## Overview

Incident Commander models the on-call workflow for a small AI platform with three coupled services: `frontend`, `auth`, and `db`. Each episode is a black-box incident response problem. The agent observes the current state, inspects logs or metrics, takes a typed action, and tries to recover service without burning budget or causing avoidable escalation.

The environment is designed to be deterministic under fixed seeds, easy to validate locally and in CI, compatible with Hugging Face Spaces and Docker, and transparent enough for judges to inspect replays, reports, and baseline comparisons.

At a practical level, each episode is a short incident drill. You reset the environment, read the observation payload, choose a single action, and repeat until the task ends. The value of the benchmark comes from the fact that the right action is not always the obvious one: sometimes you should inspect first, sometimes you should remediate immediately, and sometimes you should recover one service so that another can be fixed safely.

## Platform Model

The simulated platform is intentionally small, but the coupling between services makes the decision space realistic:

- `frontend` is the customer-facing entry point and usually where symptoms appear first.
- `auth` handles identity and token validation, so failures there can look like downstream API issues.
- `db` is the persistence layer, and problems there often surface as retries, latency spikes, or cascading errors.

The agent does not get a fully narrated answer. It gets structured telemetry, masked incident context, and recent action history. That means the policy has to infer root cause from symptoms instead of reading a label that gives the answer away.

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

The tasks are intentionally not equivalent in how they reward action:

- `easy` is mostly a contract check. It verifies that the environment, action model, and grader wiring are working.
- `medium` introduces recovery sequencing and makes it important to inspect before acting.
- `hard` adds more mixed failure modes, so the policy has to balance throughput, verification, and cost.
- `longhaul` rewards policies that keep working over a longer horizon instead of optimizing only the current observation.
- `blackout` is the strongest stress test, with prolonged pressure and a bigger need for disciplined remediation.

## Control Loop

The environment follows a simple but operationally realistic loop:

1. Call `/reset` for a chosen `task_id`.
2. Inspect the returned observation.
3. Choose a single structured action.
4. Apply the action with `/step`.
5. Monitor state, reward, and incident progress.
6. Finish when the episode ends, the task is recovered, or a failure condition is reached.

The environment is stateful across steps inside one episode. Actions can change the current service state, and the next observation reflects those changes. That is why the benchmark is useful for policy evaluation: it measures whether a controller can build on its earlier decisions instead of treating every step as isolated.

### Example Interaction

The API is designed to be easy to call from a script or agent loop:

```bash
curl -X POST "http://localhost:7860/reset?task_id=easy&seed=42"
curl -X POST "http://localhost:7860/step?task_id=easy" \
  -H "Content-Type: application/json" \
  -d '{"action_type":"get_metrics","target_service":"frontend","delta_instances":0,"fallback_service":null,"config_key":null,"config_value":null,"n_lines":0,"question":null,"note":"inspect first"}'
curl "http://localhost:7860/grade?task_id=easy"
```

The exact action schema is defined by `app/models.py`, and the environment rejects invalid combinations instead of silently accepting them.

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

The environment also exposes a `visualize` endpoint that returns a compact ASCII dashboard for quick manual inspection. That makes it easier to compare the current service state against the task description during debugging.

## Reward and Grading

Reward is dense and inspectable. It combines uptime, latency, SLA compliance, cost, recovery, MTTR bonus, burn-budget penalty, anti-panic penalty, and safety penalties.

Each task has a deterministic grader in `server.tasks` that returns a score in `[0.0, 1.0]` and rewards partial progress instead of only binary success.

The grader is intentionally shaped to reward practical incident response, not just raw throughput. A policy that restores service but does so carelessly, noisily, or wastefully will generally score worse than one that stabilizes the system with fewer unnecessary interventions.

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

Useful endpoint notes:

- `/reset` initializes a task context and returns the first observation.
- `/step` applies exactly one action to the active task context.
- `/state` returns the current state without advancing the episode.
- `/grade` evaluates the current episode state with the task-specific grader.
- `/baseline` and `/benchmark_matrix` run the built-in baseline controller for quick comparison.
- `/report` combines grade, metrics, and unresolved incident context in one payload.
- `/replay` and `/evaluation_report` are useful when you want deterministic inspection artifacts rather than a single score.

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

If you are validating a submission bundle, run the local smoke test first, then the OpenEnv validator, then the submission-specific validator script. That sequence catches most environment and API mismatch issues before you try to deploy.

## Inference

Environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `ENV_URL`
- optional `LOCAL_IMAGE_NAME`

The inference script uses the OpenAI client and emits strict `[START]`, `[STEP]`, and `[END]` log lines. If the model call is unavailable, it falls back to a deterministic heuristic controller instead of a noop policy.

This fallback is deliberate. It keeps the script usable in restricted environments and gives judges something meaningful to compare against even when an external LLM is not available.

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

For quick sanity checks, the built-in baseline is the fastest way to confirm that task wiring, grading, and episode transitions still behave as expected after a change.

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
- This repository intentionally keeps the documentation surface minimal so the README remains the source of truth.

## Troubleshooting

If something looks off during local testing, these checks usually find the issue quickly:

- Confirm the server is running on port `7860` before calling `/reset` or `/step`.
- Make sure you are using `task_id`, not `task_name`, when calling the API.
- If `/grade` fails, reset the episode again and replay the same task from a clean state.
- If the inference script falls back to heuristics, verify that `HF_TOKEN` and `API_BASE_URL` are set.
- If the validator complains about endpoint coverage, check `app/main.py` and `scripts/test-local.py` for the expected contract.

## Development Notes

The repository keeps the main implementation in `app/` and the grader logic in `server/tasks.py`. That separation is intentional: it keeps the environment mechanics close to the action model, while the task scoring stays easy to inspect.

If you change the environment behavior, update the README, the validator expectations, and any smoke tests that encode the public contract. That keeps the documentation, runtime behavior, and evaluation flow aligned.

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
