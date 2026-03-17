from __future__ import annotations

import os

PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
NO_PROXY_KEYS = ("NO_PROXY", "no_proxy")
DISABLE_ALL_PROXY = "*"


def disable_http_proxy() -> None:
    """Disable HTTP proxy usage for the current process.

    This is intentionally one-way: Auctrace does not configure proxies anymore.
    """
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)
    for key in NO_PROXY_KEYS:
        os.environ[key] = DISABLE_ALL_PROXY
