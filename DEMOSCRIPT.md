# Demo Script

This script is designed for judge walkthroughs and live demos.

## 1) Start the server

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 7860
```

## 2) Probe task catalog

```bash
curl "http://127.0.0.1:7860/tasks"
```

Expected: `easy`, `medium`, `hard`, `longhaul`, `blackout`.

## 3) Run a deterministic reset

```bash
curl -X POST "http://127.0.0.1:7860/reset?task_id=blackout&seed=42"
```

Expected observation includes:
- `incident_type`
- `incident_severity`
- `civilian_risk`
- `emergency_units`
- `strategic_options`
- `tactical_options`

## 4) Execute strategic then tactical actions

Declare emergency:

```bash
curl -X POST "http://127.0.0.1:7860/step?task_id=blackout" \
  -H "Content-Type: application/json" \
  -d '{"action_type":"declare_emergency","note":"activate command mode"}'
```

Deploy drone scan:

```bash
curl -X POST "http://127.0.0.1:7860/step?task_id=blackout" \
  -H "Content-Type: application/json" \
  -d '{"action_type":"deploy_drone_scan","target_zone":"city_core","strategy_level":"tactical"}'
```

Evacuate zone:

```bash
curl -X POST "http://127.0.0.1:7860/step?task_id=blackout" \
  -H "Content-Type: application/json" \
  -d '{"action_type":"evacuate_zone","target_zone":"city_core","priority":"critical"}'
```

## 5) Show reward decomposition and trace

```bash
curl "http://127.0.0.1:7860/metrics?task_id=blackout&include_trace=true"
```

Expected trace includes:
- `civilian_risk`
- `incident_severity`
- `civilians_saved`
- decomposed reward terms

## 6) Show state visualization

```bash
curl "http://127.0.0.1:7860/visualize?task_id=blackout"
```

## 7) Produce judge-grade report bundle

```bash
curl "http://127.0.0.1:7860/report?task_id=blackout"
curl "http://127.0.0.1:7860/replay?task_id=blackout&seed=42&policy=reasoning"
curl "http://127.0.0.1:7860/evaluation_report?policy=baseline&episodes_per_task=2&seed_start=42"
curl "http://127.0.0.1:7860/judge_pack"
```

## 8) Validate contract

```bash
python scripts/test-local.py
python -m openenv.cli validate
```
