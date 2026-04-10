from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

REQUIRED_ENDPOINTS = ["reset", "step", "state"]
REQUIRED_INFERENCE_MARKERS = ["log_start(", "log_step(", "log_end("]


def _run(cmd: List[str], cwd: Path) -> Tuple[bool, str]:
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
        )
        output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        return result.returncode == 0, output.strip()
    except Exception as exc:
        return False, str(exc)


def _check_required_files(repo_root: Path) -> Tuple[bool, List[str]]:
    required = [
        repo_root / "openenv.yaml",
        repo_root / "Dockerfile",
        repo_root / "inference.py",
        repo_root / "app" / "main.py",
        repo_root / "tests" / "test_env_contract.py",
    ]
    missing = [str(path.relative_to(repo_root)) for path in required if not path.exists()]
    return len(missing) == 0, missing


def _check_openenv_yaml(repo_root: Path) -> Tuple[bool, List[str]]:
    path = repo_root / "openenv.yaml"
    if not path.exists():
        return False, ["openenv.yaml not found"]

    text = path.read_text(encoding="utf-8")
    issues: List[str] = []

    for endpoint in REQUIRED_ENDPOINTS:
        marker = f"{endpoint}: /{endpoint}"
        if marker not in text:
            issues.append(f"Missing endpoint mapping: {marker}")

    for field in ["name:", "version:", "tasks:"]:
        if field not in text:
            issues.append(f"Missing required metadata field: {field}")

    return len(issues) == 0, issues


def _check_inference_contract(repo_root: Path) -> Tuple[bool, List[str]]:
    path = repo_root / "inference.py"
    if not path.exists():
        return False, ["inference.py not found"]

    text = path.read_text(encoding="utf-8")
    missing = [marker for marker in REQUIRED_INFERENCE_MARKERS if marker not in text]
    return len(missing) == 0, [f"Missing inference logger marker: {m}" for m in missing]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    checks: List[dict] = []

    files_ok, files_issues = _check_required_files(repo_root)
    checks.append(
        {
            "check": "Required files",
            "status": "pass" if files_ok else "fail",
            "details": files_issues if files_issues else ["All required files are present"],
        }
    )

    yaml_ok, yaml_issues = _check_openenv_yaml(repo_root)
    checks.append(
        {
            "check": "openenv.yaml structure",
            "status": "pass" if yaml_ok else "fail",
            "details": yaml_issues if yaml_issues else ["openenv.yaml contains required core fields"],
        }
    )

    inference_ok, inference_issues = _check_inference_contract(repo_root)
    checks.append(
        {
            "check": "inference.py logging contract",
            "status": "pass" if inference_ok else "fail",
            "details": inference_issues if inference_issues else ["inference logging markers are present"],
        }
    )

    pytest_ok, pytest_output = _run([sys.executable, "-m", "pytest", "tests", "-q"], repo_root)
    checks.append(
        {
            "check": "Test suite",
            "status": "pass" if pytest_ok else "fail",
            "details": [pytest_output.splitlines()[-1] if pytest_output else "No output"],
        }
    )

    validate_ok, validate_output = _run([sys.executable, "-m", "openenv.cli", "validate"], repo_root)
    checks.append(
        {
            "check": "OpenEnv validate",
            "status": "pass" if validate_ok else "fail",
            "details": [validate_output.splitlines()[-1] if validate_output else "No output"],
        }
    )

    passed = sum(1 for c in checks if c["status"] == "pass")
    total = len(checks)
    overall = "pass" if passed == total else "fail"

    report = {
        "project": "incident-commander",
        "overall": overall,
        "passed": passed,
        "total": total,
        "checks": checks,
    }

    print(json.dumps(report, indent=2))
    return 0 if overall == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
