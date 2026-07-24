from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_params(params_path: str | Path) -> dict[str, Any]:
    with Path(params_path).open("r", encoding="utf-8") as params_file:
        return yaml.safe_load(params_file)


def resolve_path(project_root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return project_root / path
