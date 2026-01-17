from __future__ import annotations

from typing import Any, Dict
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = {
        "project": raw.get("project", {}),
        "data": raw.get("data", {}),
        "srm": raw.get("srm", {}),
        "profit": raw.get("profit", {}),
        "mes": raw.get("mes", {}),
        "bootstrap": raw.get("bootstrap", {}),
        "guardrails": raw.get("guardrails", {}),
    }

    return cfg
