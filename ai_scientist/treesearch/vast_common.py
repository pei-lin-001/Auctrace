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
        "auto_destroy": raw.get("auto_destroy", True),
        "readiness_timeout": raw.get("readiness_timeout", 900),
        "setup_timeout": raw.get("setup_timeout", 1800),
        "instance_poll_interval": raw.get("instance_poll_interval", 10),
        "max_provision_attempts": raw.get("max_provision_attempts", 3),
        "ssh_probe_retries": raw.get("ssh_probe_retries", 6),
        "ssh_probe_retry_delay": raw.get("ssh_probe_retry_delay", 10),
        "ssh_probe_timeout": raw.get("ssh_probe_timeout", 30),
        "ssh_probe_connect_timeout": raw.get("ssh_probe_connect_timeout", 15),
        "remote_root": raw.get("remote_root", "/workspace/auctrace"),
        "image": raw.get("image", "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel"),
        "disk_gb": raw.get("disk_gb", 64),
        "runtype": raw.get("runtype", "ssh_direct"),
        "label_prefix": raw.get("label_prefix", "auctrace"),
        "install_project_requirements": raw.get("install_project_requirements", True),
        "requirements_file": raw.get("requirements_file", "requirements.txt"),
        "pip_install_timeout": raw.get("pip_install_timeout", 1800),
        "auto_install_missing_packages": raw.get("auto_install_missing_packages", True),
        "max_auto_dependency_installs": raw.get("max_auto_dependency_installs", 3),
        "setup_commands": raw.get("setup_commands", []),
        "search": {
            "limit": search.get("limit", 25),
            "order": search.get("order", "dph_total"),
            "type": search.get("type", "on-demand"),
            "verified": search.get("verified", True),
            "rentable": search.get("rentable", True),
            "rented": search.get("rented", False),
            "num_gpus": search.get("num_gpus", 1),
            "reliability_min": search.get("reliability_min", 0.95),
            "inet_up_min": search.get("inet_up_min", 100),
            "direct_port_count_min": search.get("direct_port_count_min", 1),
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
