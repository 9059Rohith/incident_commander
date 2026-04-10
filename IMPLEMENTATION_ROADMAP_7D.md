# Incident Commander 7-Day Top-Tier Roadmap

This roadmap converts the high-level upgrade ideas into a strict execution plan with file-level targets, test gates, and measurable outcomes.

## Goal

Move Incident Commander from strong submission to top-tier benchmark quality through:
- stronger world realism,
- stronger anti-gaming and safety guarantees,
- stronger evidence for policy generalization,
- faster judge verification workflow.

## Day 1 (24-hour impact)

### Scope
1. Hidden track hardening and weighted scoring policy docs.
2. Baseline-ladder validation outputs in a single report.
3. Judge quick verification path (under 5 minutes).

### Files
- `app/main.py`
- `scripts/eval_baselines.py`
- `README.md`
- `openenv.yaml`

### Deliverables
1. `evaluation_report` includes explicit public-vs-hidden split and weighted aggregate.
2. Baseline report includes CI values and policy deltas.
3. `judge_quickstart` endpoint returns deterministic verification steps.

### Pass criteria
- `pytest tests -q` passes.
- `python scripts/eval_baselines.py` runs without errors.
- `/judge_quickstart` response includes all expected endpoints.

## Day 2-3 (systems depth)

### Scope
1. Multi-region graph outage model.
2. Commitment-mode penalties for policy thrashing.
3. Stronger adversarial shifts and secondary incident waves.

### Files
- `app/models.py`
- `app/env.py`
- `app/reward.py`
- `tests/test_env_contract.py`

### Deliverables
1. Region topology appears in observation and state payloads.
2. Reward decomposition includes commitment and topology outage penalties.
3. New failures appear in failure taxonomy (action spam and commitment thrash).

### Pass criteria
- Contract tests validate region and commitment fields.
- Same seed gives deterministic replay outputs.
- Baseline policy underperforms reasoning/trained policy on hard tasks.

## Day 4-5 (forensic + governance)

### Scope
1. Counterfactual forensic audit endpoint.
2. Scenario governance and anti-overfit policy doc.
3. Seed protocol and hidden rotation protocol.

### Files
- `app/main.py`
- `README.md`
- `DEMOSCRIPT.md`
- `IMPLEMENTATION_ROADMAP_7D.md`

### Deliverables
1. `/forensic_audit` returns failure timeline + recommended interventions.
2. README documents hidden policy and anti-gaming assumptions.
3. Demo script verifies all core claims in deterministic order.

### Pass criteria
- `/forensic_audit` works for all policy modes.
- Judge pack includes forensic preview.
- Demo script executes end-to-end with no invalid endpoint calls.

## Day 6 (stress + quality gates)

### Scope
1. Distribution-shift stress profiles per task.
2. Regression checks for score drift and robustness drift.
3. CI-level threshold checks.

### Files
- `app/env.py`
- `scripts/eval_baselines.py`
- `tests/test_env_contract.py`

### Deliverables
1. Stress test mode can be toggled via evaluation endpoint flags.
2. Baseline script emits robustness summary and delta thresholds.
3. Tests fail if key benchmark guarantees regress.

### Pass criteria
- Robustness metrics remain within expected corridor.
- Public+hidden weighted score remains deterministic by seed.
- No endpoint contract regressions.

## Day 7 (submission hardening)

### Scope
1. Final benchmark narrative cleanup.
2. Final OpenEnv and Docker validation run.
3. Publish judge-first summary matrix.

### Files
- `README.md`
- `openenv.yaml`
- `scripts/eval_baselines.py`

### Deliverables
1. README shows benchmark value in measurable terms, not only claims.
2. Validator and Docker build both pass cleanly.
3. Judge report includes policy ladder and confidence intervals.

### Pass criteria
- `python -m openenv.cli validate` passes.
- Docker image builds successfully.
- `judge_pack` and `evaluation_report` responses include all expected keys.

## Mandatory quality gates (every day)

1. Unit/contract tests: pass.
2. Deterministic replay: same seed -> same score and taxonomy.
3. API compatibility: no breaking changes to public endpoints.
4. Documentation parity: README reflects actual runtime behavior.
5. Baseline sanity: reasoning/trained should outperform noop on hard tasks.

## Score uplift targets

1. Task & Grader Quality: +2 to +3 from stronger hidden/forensic evidence.
2. Environment Design: +1 to +2 from graph outages and commitment penalties.
3. Creativity: +1 from multi-level control + forensic counterfactuals.
4. Judge confidence: major uplift from fast verification endpoints.

## Final shipping checklist

1. Run tests.
2. Run baseline evaluation script.
3. Run OpenEnv validator.
4. Build Docker image.
5. Verify `judge_quickstart`, `judge_pack`, `forensic_audit` endpoints.
6. Push to GitHub and Hugging Face remotes.
