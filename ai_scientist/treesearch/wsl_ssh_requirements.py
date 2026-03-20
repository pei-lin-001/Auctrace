from __future__ import annotations

import re
from pathlib import Path

REQUIREMENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def required_packages_from_file(requirements_path: Path) -> list[str]:
    packages: list[str] = []
    for raw_line in requirements_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        if " #" in line:
            line = line.split(" #", 1)[0].strip()
        match = REQUIREMENT_NAME_PATTERN.match(line)
        if match:
            packages.append(match.group(0))
    return packages
