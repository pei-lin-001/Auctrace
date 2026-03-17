from __future__ import annotations

import os
from pathlib import Path

PROJECT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def load_project_env() -> None:
    if not PROJECT_ENV_PATH.exists():
        return
    for raw_line in PROJECT_ENV_PATH.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))
