from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class IncidentCommanderClient:
    base_url: str = "http://localhost:7860"
    timeout: int = 15

    def reset(self, task_id: str = "easy", seed: int = 42) -> Dict[str, Any]:
        response = requests.post(f"{self.base_url}/reset", params={"task_id": task_id, "seed": seed}, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def step(self, task_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(f"{self.base_url}/step", params={"task_id": task_id}, json=action, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def state(self, task_id: str = "easy") -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/state", params={"task_id": task_id}, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def grade(self, task_id: str = "easy") -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/grade", params={"task_id": task_id}, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
