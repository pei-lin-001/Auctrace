from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

VAST_API_ROOT = "https://console.vast.ai/api/v0"


@dataclass
class VastRuntime:
    instance_id: int
    ssh_host: str
    ssh_port: int
    num_gpus: int
    remote_root: str
    private_key_path: str | None
    offer_id: int | None = None
    machine_id: int | None = None


def cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Cannot coerce Vast setting to bool: {value!r}")


def normalize_vast_settings(exec_cfg: Any) -> dict[str, Any]:
    raw = cfg_value(exec_cfg, "vast", {}) or {}
    search = raw.get("search", {})
    return {
        "api_key_env": raw.get("api_key_env", "VAST_API_KEY"),
        "private_ssh_key_path_env": raw.get(
            "private_ssh_key_path_env", "VAST_SSH_PRIVATE_KEY_PATH"
        ),
        "public_ssh_key_path_env": raw.get(
            "public_ssh_key_path_env", "VAST_SSH_PUBLIC_KEY_PATH"
        ),
        "existing_instance_id": raw.get("existing_instance_id"),
        "offer_id": raw.get("offer_id"),
        "auto_destroy": _as_bool(raw.get("auto_destroy", True), True),
        "readiness_timeout": _as_int(raw.get("readiness_timeout", 900), 900),
        "setup_timeout": _as_int(raw.get("setup_timeout", 1800), 1800),
        "instance_poll_interval": _as_int(raw.get("instance_poll_interval", 10), 10),
        "max_provision_attempts": _as_int(raw.get("max_provision_attempts", 3), 3),
        "ssh_probe_retries": _as_int(raw.get("ssh_probe_retries", 6), 6),
        "ssh_probe_retry_delay": _as_int(raw.get("ssh_probe_retry_delay", 10), 10),
        "ssh_probe_timeout": _as_int(raw.get("ssh_probe_timeout", 30), 30),
        "ssh_probe_connect_timeout": _as_int(
            raw.get("ssh_probe_connect_timeout", 15), 15
        ),
        "remote_root": raw.get("remote_root", "/workspace/auctrace"),
        "image": raw.get("image", "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel"),
        "disk_gb": _as_int(raw.get("disk_gb", 64), 64),
        "runtype": raw.get("runtype", "ssh_direct"),
        "label_prefix": raw.get("label_prefix", "auctrace"),
        "install_project_requirements": _as_bool(
            raw.get("install_project_requirements", True), True
        ),
        "requirements_file": raw.get("requirements_file", "requirements.txt"),
        "pip_install_timeout": _as_int(raw.get("pip_install_timeout", 1800), 1800),
        "auto_install_missing_packages": _as_bool(
            raw.get("auto_install_missing_packages", True), True
        ),
        "max_auto_dependency_installs": _as_int(
            raw.get("max_auto_dependency_installs", 3), 3
        ),
        "setup_commands": raw.get("setup_commands", []),
        "search": {
            "limit": _as_int(search.get("limit", 25), 25),
            "order": search.get("order", "dph_total"),
            "type": search.get("type", "on-demand"),
            "verified": _as_bool(search.get("verified", True), True),
            "rentable": _as_bool(search.get("rentable", True), True),
            "rented": _as_bool(search.get("rented", False), False),
            "num_gpus": _as_int(search.get("num_gpus", 1), 1),
            "reliability_min": _as_float(search.get("reliability_min", 0.95), 0.95),
            "inet_up_min": _as_float(search.get("inet_up_min", 100), 100.0),
            "direct_port_count_min": _as_int(
                search.get("direct_port_count_min", 1), 1
            ),
        },
        "runtime": raw.get("runtime"),
    }


def read_text_if_set(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Configured Vast.ai SSH key path does not exist: {path}")
    return path.read_text().strip()


def runtime_payload(runtime: VastRuntime) -> dict[str, Any]:
    return {
        "instance_id": runtime.instance_id,
        "ssh_host": runtime.ssh_host,
        "ssh_port": runtime.ssh_port,
        "num_gpus": runtime.num_gpus,
        "remote_root": runtime.remote_root,
        "private_key_path": runtime.private_key_path,
        "offer_id": runtime.offer_id,
        "machine_id": runtime.machine_id,
    }


def set_runtime(exec_cfg: Any, runtime: VastRuntime) -> None:
    payload = runtime_payload(runtime)
    if isinstance(exec_cfg.vast, dict):
        exec_cfg.vast["runtime"] = payload
        return
    exec_cfg.vast.runtime = payload
