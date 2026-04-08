# Incident Commander Pre-Submission Checklist

Use this checklist right before final submission.

## 1. Hosted Space checks

- [ ] Space URL is public and reachable.
- [ ] POST `{space_url}/reset` returns HTTP 200.
- [ ] GET `{space_url}/health` returns status `ok`.
- [ ] GET `{space_url}/tasks` returns exactly 5 tasks: easy, medium, hard, longhaul, blackout.

## 2. Local quality gates

- [ ] Run tests:
  - `python -m pytest -q`
- [ ] Run OpenEnv validation:
  - `python -m openenv.cli validate`
- [ ] Run Docker build from repo root:
  - `docker build .`
- [ ] Run local endpoint smoke test (with server running):
  - `python scripts/test-local.py`

## 3. Validator scripts

Linux/macOS:

```bash
bash scripts/validate-submission.sh https://your-space.hf.space .
```

Windows PowerShell:

```powershell
./scripts/validate-submission.ps1 -PingUrl "https://your-space.hf.space" -RepoDir "."
```

## 4. Inference readiness

- [ ] `inference.py` runs without crash from repo root.
- [ ] `HF_TOKEN` is configured if remote model inference is required.
- [ ] Fallback heuristic behavior is acceptable when token/quota is unavailable.

## 5. Submission packaging

- [ ] README clearly documents setup, validate, and endpoints.
- [ ] openenv.yaml has correct metadata and task definitions.
- [ ] No secrets are committed.
- [ ] Latest working Space URL is recorded in submission form.
