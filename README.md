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

## Why this project matters

AI platforms fail in very specific ways: deploys regress, traffic spikes overwhelm auth dependencies, database stalls cascade into frontend latency, and on-call engineers must decide whether to scale, rollback, reroute, quarantine, or escalate. Incident Commander OpenEnv turns that operational reality into a high-signal RL benchmark.

The agent does not optimize a toy score. It makes constrained, sequential decisions under partial observability and delayed consequences, with the same reliability-versus-cost trade-offs used in production SRE work.

- restore uptime before SLA breaches pile up,
- keep latency under control during traffic surges,
- stop cascading failures before they spread,
- balance recovery actions against cost and human attention.

This environment is explicitly **systemic rather than atomic**: failures are coupled across services (for example, `db` failures first poison `auth`, then surface as `frontend` 404/500 symptoms), so the agent must reason over interactions instead of isolated thresholds.

Existing coding-heavy environments mostly evaluate code synthesis. **Incident Commander instead evaluates real systems reasoning under outage pressure**: investigate, hypothesize, remediate, and verify.

The policy challenge is also **cognitive load under uncertainty**: observations can be noisy or stale, spot capacity can disappear, and priorities shift over time. High scores require robust decision-making, not brittle if-else reactions.

## Why RL and not just prompting

This environment is not a one-shot question-answer task. The agent acts in a stateful system where actions change future dynamics, and the return depends on multi-step consequences.

Why prompting alone is insufficient:

- Delayed credit assignment: beneficial actions can reduce short-term reward before improving long-horizon survival.
- Non-stationarity: phase changes (`steady`, `surge`, `silent-leak`, `thundering-herd`) require policy adaptation mid-episode.
- Partial observability: metrics can be stale or missing, so the agent must infer hidden state from secondary signals.
- Coupled failures: local fixes can worsen global behavior (for example, DB latency backpressure amplifying gateway load).

A purely reactive prompt policy can look competent on simple episodes but fails on high-pressure horizons where planning depth matters.

## Why this is a strong hackathon submission

- It models a job humans actually do: on-call incident response for an AI platform.
- The task ladder is easy to explain and hard to game.
- The reward is dense, interpretable, and aligned with production objectives.
- The project includes typed models, a validator script, a baseline script, a Dockerfile, and a Hugging Face Space deployment path.
- The action space mirrors production workflows (scale, reroute, rollback, quarantine, escalate) used in Kubernetes/Terraform-style operations.
- The agent must manage uncertainty from noisy or delayed telemetry, not just react to clean threshold signals.

## What changed in this version

- Added a stateful service graph with three interdependent services: `frontend -> auth -> db`.
- Added a scenario factory (`resource_exhaustion`, `config_drift`, `heisenbug`) instead of static scripted failures.
- Added black-box observability: root cause is hidden until the agent investigates logs, metrics, and network paths.
- Added a professional SRE toolbelt action space (exploration, remediation, verification, human collaboration).
- Added an ephemeral filesystem model where `/tmp` and runtime config are reset every episode.
- Added live incident timeline output in `/metrics` for step-by-step auditability.
- Added a multi-region failure mode (`regional_outage`) with cross-zone packet loss and replication lag.
- Added explicit `failover_database` action for resilient, production-style disaster recovery.
- Added judge-friendly endpoints `/report` and `/benchmark_matrix` for quick comparative evaluation.

---

## Environment overview

The environment simulates a microservice cluster with three core services:

- `frontend`: customer-facing API and web edge
- `auth`: identity and token service
- `db`: stateful dependency for auth/session data

The agent receives a terminal-like snapshot with noisy telemetry and user-visible symptoms. It is not told the root cause directly, and must inspect logs, metrics, and connectivity to trace failures.

### Core loop

1. The agent observes the current incident state.
2. It chooses an exploratory, remediation, verification, or communication action.
3. The environment updates service health, traffic pressure, and incidents.
4. Reward is computed from success minus latency/resource/incorrect/safety penalties.
5. The episode ends at the step budget, SLA failure threshold, or outage collapse.

---

## Task design

### Task 1: easy (30 steps)

- Objective: recover a single degraded API service under rising latency.
- What it tests: basic recovery and action selection.

### Task 2: medium / recovery calibration (40 steps)

- Objective: handle a bad deploy and traffic surge with rollback and scaling.
- What it tests: multi-step recovery under pressure.

### Task 3: hard (50 steps)

- Objective: stop a cascading failure across frontend, auth, and db services.
- What it tests: coordinated incident response and prioritization.

### Task 4: longhaul (60 steps)

- Objective: detect and mitigate a silent memory leak before "healthy" services crash.
- What it tests: proactive intervention and delayed credit assignment.

### Task 5: blackout (70 steps)

- Objective: survive a thundering-herd traffic wave where under-scaling causes DB lockups but over-scaling burns budget.
- What it tests: predictive planning from traffic rate-of-change and cost-aware scaling discipline.

---

## Observation space

Each step the agent receives a complete platform snapshot:

| Field | Type | Description |
|---|---|---|
| services | Dict[str, ServiceState] | Health, instance count, latency, error rate, and quarantine state per service |
| active_incidents | List[ActiveIncident] | Masked incident cards; deeper root cause is hidden until investigation |
| step | int | Current control cycle |
| step_budget | int | Maximum episode length |
| traffic_level | float | Aggregate demand pressure for the step |
| uptime | float | Service uptime ratio in [0, 1] |
| p95_latency | float | Current tail latency estimate |
| sla_breaches | int | Cumulative SLA violations |
| cost_per_step | float | Accumulated operating cost |
| last_action_result | str | Human-readable result of the last action |
| phase | str | Episode phase marker used for long-horizon dynamics |
| symptoms | List[str] | What customers/on-call dashboards are currently reporting |
| terminal_output | List[str] | Recent outputs from exploratory actions |
| investigation_log | List[str] | Rolling log of metrics/log/network checks |
| live_timeline | List[str] | Time-stamped incident timeline for auditability |
| available_actions | List[str] | Explicit toolbelt available to the policy |

`ServiceState` includes CPU, memory, queue depth, and noisy observed metrics (`observed_p95_latency`, `observed_error_rate`, `metric_staleness_steps`) so agents can reason under uncertainty.

## Action space

The agent can take one of the following actions:

| Action | Description |
|---|---|
| `get_metrics` | Inspect telemetry for a service |
| `list_processes` | Inspect service process/worker state |
| `read_last_n_logs` | Inspect service logs to trace failures |
| `check_network_connectivity` | Probe service-to-service network paths |
| `failover_database` | Promote DB replica across zones during regional packet-loss incidents |
| `restart_service` | Restart a service instance pool |
| `rollback_deployment` | Roll back a service to a safe release |
| `scale_up_replicas` | Add replicas to recover capacity |
| `edit_config_line` | Patch runtime config key/value |
| `run_healthcheck` | Verify remediation succeeded |
| `ask_developer` | Query a human/developer hint channel |
| `load_test` | Intentionally stress the system for repro |
| `run_command` | Execute an operational command (safety-penalized when reckless) |
| `noop` | Leave the system unchanged for the step |

### Action schema

```python
class IncidentCommanderAction(BaseModel):
  action_type: Literal["get_metrics", "list_processes", "read_last_n_logs", "check_network_connectivity", "failover_database", "restart_service", "rollback_deployment", "scale_up_replicas", "edit_config_line", "run_healthcheck", "ask_developer", "load_test", "run_command", "noop"]
    target_service: Optional[str] = None
    delta_instances: int = 0
    fallback_service: Optional[str] = None
  n_lines: int = 20
  config_key: Optional[str] = None
  config_value: Optional[str] = None
  question: Optional[str] = None
  command: Optional[str] = None
    note: Optional[str] = None
```

---

## Reward model

Per-step reward is decomposed and inspectable:

| Component | Signal | Behavior encouraged |
|---|---|---|
| Uptime | Fraction of served demand | Keep core services available |
| Latency health | Inverse p95 latency pressure | Avoid overload and queue growth |
| SLA compliance | Breach-aware term | Protect tail performance during incidents |
| Cost control | Budget ratio term | Prevent runaway spend |
| Recovery | Resolved incident ratio | Clear incidents instead of masking symptoms |
| MTTR bonus | Fast recovery bonus for incidents resolved in <=3 steps | Reward decisive, early mitigation |
| Burn-budget penalty | 99.9% SLA budget exhaustion penalty (heavier after budget is consumed) | Avoid repeated downtime bursts |
| Anti-panic penalty | Penalizes contradictory actions (for example scale up then immediate scale down) | Prefer stable engineering decisions |
| Safety penalty | Massive penalty for destructive action without prior investigation | Enforce safe SRE workflow |

Reward objective:

$$
R = Success - \text{Latency\_Penalty} - \text{Resource\_Waste} - \text{Incorrect\_Actions} - \text{Safety\_Penalty}
$$

The environment keeps a per-step reward trace and exposes it through `/metrics` for judge auditability.

The MTTR and anti-panic terms are intentional differentiators: they reward fast, stable incident management rather than noisy action spam.

### Anti-exploit safeguards

The environment includes explicit controls to reduce reward hacking and brittle policies:

- Unforced escalation penalty: paging human intervention without active high/critical pressure is penalized.
- Passive-loop penalty: repeated `noop`/`ask_developer` streaks receive an additional penalty.
- Budget failure boundary: episodes terminate early if cumulative operating cost exceeds a hard budget multiplier.

These safeguards make high scores correlate with operationally meaningful behavior instead of trivial exploit patterns.

## Grading

Each task has a deterministic grader that returns a score in the range [0.0, 1.0]. Graders are designed to reward partial progress rather than forcing binary success or failure.

- easy emphasizes restoring the first degraded service quickly.
- medium emphasizes rollback plus scaling under pressure.
- hard emphasizes stopping cascading failures before the platform collapses.
- longhaul emphasizes proactive leak mitigation before visible outage.
- blackout emphasizes thundering-herd survival with strict SLA burn-budget control.

The grader favors uptime, latency, SLA protection, cost discipline, incident recovery, and escalation discipline rather than a single brittle metric.

---

## Endpoints

- POST `/reset?task_id={easy|medium|hard|longhaul|blackout}&seed=42`
- POST `/step?task_id={easy|medium|hard|longhaul|blackout}`
- GET `/state?task_id={easy|medium|hard|longhaul|blackout}`
- GET `/grade?task_id={easy|medium|hard|longhaul|blackout}`
- GET `/tasks`
- GET `/health`
- GET `/visualize?task_id={easy|medium|hard|longhaul|blackout}`
- GET `/baseline?task_id={easy|medium|hard|longhaul|blackout}&episodes=5`
- GET `/metrics?task_id={easy|medium|hard|longhaul|blackout}`
- GET `/report?task_id={easy|medium|hard|longhaul|blackout}`
- GET `/benchmark_matrix?episodes=3`
- GET `/replay?task_id={easy|medium|hard|longhaul|blackout}&seed=42&policy={baseline|noop|reasoning}&max_steps=optional`
- GET `/evaluation_report?policy={baseline|noop|reasoning}&episodes_per_task=3&seed_start=42`

`/metrics` supports `include_trace=true|false` and returns both aggregate scores and per-step reward breakdown.

`/replay` returns a deterministic full trajectory export (state before/after each step, action, reward decomposition, and incident narrative tail) for a fixed `task_id + seed + policy`.

`/evaluation_report` returns one-call benchmark analytics across all tasks including average score, variance-derived robustness, and failure taxonomy totals.

### Replay export examples

```bash
curl "http://localhost:7860/replay?task_id=hard&seed=42&policy=baseline"
curl "http://localhost:7860/replay?task_id=blackout&seed=42&policy=reasoning&max_steps=40"
```

### One-call evaluation report examples

```bash
curl "http://localhost:7860/evaluation_report?policy=baseline&episodes_per_task=3&seed_start=42"
curl "http://localhost:7860/evaluation_report?policy=reasoning&episodes_per_task=5&seed_start=101"
```

---

## Setup

```bash
pip install -e .
python -m uvicorn app.main:app --host 0.0.0.0 --port 7860
```

## Validate

```bash
bash scripts/validate-submission.sh https://your-space.hf.space .
```

PowerShell (Windows):

```powershell
./scripts/validate-submission.ps1 -PingUrl "https://your-space.hf.space" -RepoDir "."
```

The validator checks the live Space, Docker build, and OpenEnv validation.

Use the Space app URL (`https://<space-subdomain>.hf.space`) as the ping URL, not the `huggingface.co/spaces/...` page URL.

For local API smoke testing (run server first):

```bash
python scripts/test-local.py
```

You can also run OpenEnv validation directly from the repo root:

```bash
python -m openenv.cli validate
```

## Judge quickstart (2 minutes)

1. Open `https://<space-subdomain>.hf.space/health` and verify `status=ok`.
2. POST `https://<space-subdomain>.hf.space/reset` and verify HTTP 200.
3. GET `https://<space-subdomain>.hf.space/tasks` and confirm task ids are `easy, medium, hard, longhaul, blackout`.
4. Run:

```bash
python -m openenv.cli validate
```

5. Optionally run the one-command validator script from this repo.

## Inference

Linux/macOS:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token_here"
export ENV_URL="http://localhost:7860"
python inference.py
```

PowerShell (Windows):

```powershell
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
$env:HF_TOKEN = "hf_your_token_here"
$env:ENV_URL = "http://localhost:7860"
python inference.py
```

The inference script emits strict `[START]`, `[STEP]`, and `[END]` lines.

The script reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`, and optional `LOCAL_IMAGE_NAME` from the environment.

Defaults are applied for `API_BASE_URL` and `MODEL_NAME` only. `HF_TOKEN` is intentionally left unset by default so the script can fall back to the deterministic heuristic controller when no token is available.

If model inference is unavailable (for example quota/network issues), the script falls back to a deterministic heuristic controller instead of pure `noop`, which keeps evaluation runs stable and reproducible.

## Baseline policy comparison

Measured with `python greedy_baseline.py` over fixed seeds (`42..51`):

| Task | Noop | Greedy reactive | Reasoning policy |
|---|---:|---:|---:|
| easy | 0.985 | 1.000 | 1.000 |
| medium (recovery) | 0.913 | 0.971 | 0.971 |
| hard | 0.240 | 0.324 | 0.324 |
| longhaul | 0.251 | 0.337 | 0.270 |
| blackout | 0.078 | 0.040 | 0.331 |

Interpretation:

- easy and medium (recovery) are intentionally accessible to verify API correctness, rollback/scaling semantics, and grader determinism before entering adversarial phases.
- medium is a recovery-calibration task, not the primary challenge benchmark; discriminative difficulty is concentrated in hard/longhaul/blackout.
- hard/longhaul stay below 0.50 for naive/reactive policies, demonstrating that shallow heuristics are insufficient.
- blackout shows a large gap between greedy and phase-aware reasoning due to thundering-herd dynamics and burn-budget pressure.
- In blackout, `0.331` (reasoning) vs `0.040` (greedy) is an ~8.3x gap, direct evidence that phase-aware planning outperforms reactive heuristics.

## RL evidence baseline

To reproduce the policy comparison table:

```bash
python greedy_baseline.py
```

The script prints a CSV table (`task,noop,greedy,reasoning`) across fixed seeds.

### Delayed-credit example

Concrete walkthrough from `longhaul`:

1. Step 6 (`slow-burn`): `db.memory_utilization` is 78% and still climbing; queue is stable, so scaling now looks "unnecessary".
2. Step 6 action: policy chooses `scale_up_replicas(db, +1)`. Immediate effect is higher cost and slight short-term reward drop.
3. Steps 7-14: memory creep continues; no immediate payoff yet.
4. Step 16 (`surge` transition): traffic jumps; without step-6 scale, db saturates and triggers auth/frontend cascade risk.
5. Steps 16-20: with proactive scale, latency and SLA breaches stay materially lower and outage is avoided.

The reward impact appears roughly 10 steps after the action, which is exactly the delayed credit-assignment behavior RL is meant to capture.
The reward for the `scale_up_replicas` action taken at step 6 only materialized at steps 16-20. That 10-step gap is the credit assignment problem: no rule bridges it, only learned experience can.

### Why blackout defeats greedy control

`blackout` is designed so greedy/reactive policies fail for opposite reasons:

1. Under-scale path: waiting for latency to spike before scaling lets DB lock contention form during `thundering-herd`, causing cascading SLA failures.
2. Over-scale path: aggressive immediate scale-up protects latency but rapidly exhausts burn-budget and cost constraints, triggering heavy penalties.
3. Winning path: phase-aware scaling that uses traffic rate-of-change, not just current latency, while preserving budget headroom for later outage waves.

This is why greedy can collapse to `0.040` while a reasoning policy reaches `0.331` on the same task family.

## Deployment

The repo includes a Dockerfile and OpenEnv manifest for Hugging Face Spaces deployment.

## Submission checklist

- HF Space responds at `/health`
- `openenv.yaml` includes metadata and task definitions
- `docker build` works from the repo root
- `inference.py` runs from the repo root
- 5 tasks are available through `/tasks`
- `/reset`, `/step`, `/state`, `/grade`, `/metrics`, and `/visualize` are implemented

For a complete final pass, use `PRE_SUBMISSION_CHECKLIST.md`.

---

## Design notes

- Uses a lightweight service simulation so the environment stays portable on CPU-only machines.
- Keeps the state machine small enough for fast iteration but rich enough to model real operational decisions.
- Rewards are decomposed for judge readability and debugging.

---

## Architecture deep dive

Incident Commander is organized to keep simulation logic, scoring logic, and serving logic cleanly separated.

### Runtime components

- Environment core (`app/env.py`): state transitions, scenario progression, action effects, incident generation, and step-level telemetry.
- Reward engine (`app/reward.py`): decomposed reward terms with explicit anti-exploit penalties.
- Typed contracts (`app/models.py`): action/observation/reward/task and episode result Pydantic models.
- API layer (`app/main.py`): OpenEnv-compatible endpoints, benchmark helpers, and judge-facing report surfaces.
- Task graders (`server/tasks.py`): deterministic per-task scoring from aggregated episode outcomes.

### Why this split helps judges

- It makes behavior auditable: environment transition bugs are isolated from grader bugs.
- It improves reproducibility: task definition + seed uniquely determine episode dynamics.
- It improves safety: action validation and sanitization happen before state mutation.

---

## Scenario mechanics by root cause

The same API can produce very different incident patterns depending on scenario selection. This allows broad policy evaluation without changing endpoint semantics.

### `resource_exhaustion`

- Models noisy-neighbor IO contention on DB.
- Pressure accumulates over time and can trigger frontend/auth degradation through dependency effects.
- Mitigation patterns: inspection + targeted DB stabilization + verification.

### `config_drift`

- Models stale application configuration (`db_port` mismatch).
- Symptoms are intentionally indirect at first (auth failures, elevated latency).
- Mitigation pattern: diagnose via logs, patch runtime config, verify health.

### `heisenbug`

- Models race-condition instability under elevated load.
- Can look benign under low load and fail only during surge windows.
- Mitigation pattern: rollback + controlled verification under load.

### `regional_outage`

- Models cross-zone packet loss and replication lag.
- Local service checks may appear healthy while end-to-end behavior degrades.
- Mitigation pattern: network diagnostics + DB failover.

---

## State transition timeline (single step)

For each `step(action)` call:

1. Action is sanitized and validated.
2. Unsafe or incorrect behavior signals are tracked.
3. Action effect is applied to mutable environment state.
4. Scheduled incident shocks may inject additional pressure.
5. Scenario drift is advanced.
6. Dependency-aware traffic simulation computes service load, latency, and uptime.
7. Active incidents are synchronized and resolved/incurred as needed.
8. Reward is computed from decomposed terms.
9. Step trace and timeline are appended for auditability.
10. Terminal conditions are evaluated (`max_steps`, SLA failure, collapse, budget failure, full recovery).

This sequencing is deliberate: it ensures the agent is scored against the consequences of both its chosen action and contemporaneous environment dynamics.

---

## API payload examples

### Reset

```bash
curl -X POST "http://localhost:7860/reset?task_id=hard&seed=42"
```

Example response (trimmed):

```json
{
  "task_id": "hard",
  "observation": {
    "step": 0,
    "phase": "degradation",
    "traffic_level": 0.0,
    "services": {
      "frontend": {"healthy": true, "instances": 3},
      "auth": {"healthy": true, "instances": 3},
      "db": {"healthy": true, "instances": 2}
    },
    "active_incidents": []
  }
}
```

### Step

```bash
curl -X POST "http://localhost:7860/step?task_id=hard" \
  -H "Content-Type: application/json" \
  -d '{"action_type":"read_last_n_logs","target_service":"auth","n_lines":30}'
```

Example response (trimmed):

```json
{
  "done": false,
  "reward": {
    "total": 0.41,
    "uptime": 0.86,
    "latency": 0.72,
    "sla": 0.80,
    "cost": 0.91,
    "recovery": 0.33
  },
  "info": {
    "action_result": "[30 lines] auth ...",
    "scenario": "config_drift"
  }
}
```

### Grade

```bash
curl "http://localhost:7860/grade?task_id=hard"
```

Example response (trimmed):

```json
{
  "task_id": "hard",
  "score": 0.57,
  "details": {
    "uptime_score": 0.74,
    "latency_score": 0.62,
    "sla_score": 0.59,
    "cost_score": 0.77,
    "recovery_score": 0.55
  }
}
```

---

## Reproducibility protocol

To reproduce benchmark numbers reliably:

1. Use fixed seeds (for example `42..51`) for all policy comparisons.
2. Keep `MODEL_NAME`, `API_BASE_URL`, and sampling parameters fixed for LLM runs.
3. Capture both aggregate grade and reward trace (`/metrics?include_trace=true`).
4. Report means and distribution summary (min/max/std when possible), not just a single run.
5. Compare policies on identical seed sets.

Recommended reporting table:

| Policy | Task | Seeds | Avg Score | Min | Max |
|---|---|---:|---:|---:|---:|
| noop | hard | 10 | ... | ... | ... |
| greedy | hard | 10 | ... | ... | ... |
| reasoning | hard | 10 | ... | ... | ... |

---

## Determinism and fairness guarantees

- Graders are deterministic and bounded in `[0.0, 1.0]`.
- Reset with the same `task_id` and `seed` starts from the same initial state.
- Action schema is typed and sanitized before mutation.
- Reward trace preserves per-step decomposition for post-hoc auditing.
- Episode failure modes are explicit (`ended_by_sla_failure`, `ended_by_budget_failure`).

---

## Performance profile and constraints

The environment is designed for hackathon constraints (2 vCPU / 8 GB RAM) and target runtime under 20 minutes.

### Expected behavior on constrained machines

- API boot: fast startup with CPU-only dependencies.
- Step throughput: lightweight simulation loop with bounded per-step complexity.
- Memory behavior: bounded timeline and trace buffers with rolling truncation.
- No GPU dependency: all workload simulation is synthetic and deterministic under seed.

---

## Troubleshooting matrix

| Symptom | Likely cause | Fast fix |
|---|---|---|
| `/reset` returns non-200 on Space | Space still starting / container boot issue | Check Space logs, confirm app binds `0.0.0.0:7860` |
| `openenv validate` fails task checks | Mismatch between endpoint behavior and task metadata | Re-run local `scripts/test-local.py`, verify `/tasks` and graders |
| Inference script returns no model output | Missing `HF_TOKEN` or provider issue | Set `HF_TOKEN`; script will fallback to deterministic heuristic |
| Score unexpectedly low on hard tasks | Reactive actions without early investigation | Use logs + metrics first, then remediate, then verify |
| Repeated SLA collapses in blackout | Over/under-scaling oscillation | Use phase-aware scaling and preserve budget headroom |

---

## Security and safety notes

- Destructive shell-like actions are safety-penalized when used recklessly.
- Unsafe command attempts are explicitly tracked in episode metrics.
- The environment is simulation-only and does not execute privileged host operations.
- Reward shaping discourages exploitative loops (`noop` spam, unforced escalation, contradictory remediations).

---

## Evaluator FAQ

### Does this require internet access to run locally?

No for environment execution and grading. Internet is only needed for remote LLM calls in `inference.py` when using provider-backed inference.

### Can this be evaluated without a model token?

Yes. The inference script includes a deterministic heuristic fallback path when `HF_TOKEN` is not set.

### How do I verify that all tasks are wired correctly?

Use `/tasks` for discovery and run `scripts/test-local.py` after starting the server.

### Where can judges inspect full decision traces?

Use `/metrics?task_id=<task>&include_trace=true` and `/report?task_id=<task>`.

### What demonstrates non-trivial RL behavior most clearly?

Compare `noop`, `greedy`, and `reasoning` policies on `hard`, `longhaul`, and `blackout` over shared fixed seeds.

---

## Suggested extension ideas (post-hackathon)

- Multi-agent mode: primary commander + specialist responder roles.
- Operator handoff memory: cross-episode context transfer under shift changes.
- Richer network topology: region-level service graphs with partial failover.
- Counterfactual evaluator: compare chosen action to oracle action under same state.
- Human-in-the-loop benchmark mode with explicit intervention budget.
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
