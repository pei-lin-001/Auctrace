from __future__ import annotations

import re
from collections.abc import Iterable

PACKAGE_ALIASES = {
    "PIL": "Pillow",
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "Crypto": "pycryptodome",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
}

MISSING_MODULE_PATTERNS = (
    re.compile(r"No module named ['\"]([^'\"]+)['\"]"),
    re.compile(r"cannot import name ['\"][^'\"]+['\"] from ['\"]([^'\"]+)['\"]"),
)


def extract_missing_module(
    exc_type: str | None,
    exc_info: dict | None,
    term_out: Iterable[str] | None = None,
) -> str | None:
    if exc_type not in {"ModuleNotFoundError", "ImportError"}:
        return None
    if exc_info:
        for key in ("name", "msg"):
            value = exc_info.get(key)
            if isinstance(value, str):
                module_name = _search_missing_module(value)
                if module_name:
                    return module_name
    for arg in (exc_info or {}).get("args", []):
        if isinstance(arg, str):
            module_name = _search_missing_module(arg)
            if module_name:
                return module_name
    for output in term_out or []:
        module_name = _search_missing_module(output)
        if module_name:
            return module_name
    return None


def package_name_for_module(module_name: str) -> str:
    top_level = module_name.split(".", 1)[0]
    return PACKAGE_ALIASES.get(top_level, top_level.replace("_", "-"))


def _search_missing_module(text: str) -> str | None:
    for pattern in MISSING_MODULE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None
