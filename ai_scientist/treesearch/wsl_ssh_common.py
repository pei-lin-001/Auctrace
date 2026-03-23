from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from .remote_exec_common import cfg_value

DEFAULT_MAX_ARCHIVE_MB = 1024
DEFAULT_MIN_FREE_DISK_GB = 50
DEFAULT_MIN_FREE_MEMORY_GB = 4.0
DEFAULT_RECOMMENDED_WORKERS = 1


@dataclass
class WslSshRuntime:
    ssh_host: str
    ssh_port: int
    ssh_user: str
    private_key_path: str | None
    remote_root: str
    runs_root: str
    venv_path: str
    windows_stage_dir: str
    windows_stage_dir_name: str
    wsl_stage_dir: str
    distro_name: str | None
    num_gpus: int
    recommended_workers: int
    cpu_count: int
    free_disk_gb: float
    available_memory_gb: float


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
    raise ValueError(f"Cannot coerce WSL SSH setting to bool: {value!r}")


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def normalize_wsl_ssh_settings(exec_cfg: Any) -> dict[str, Any]:
    raw = cfg_value(exec_cfg, "wsl_ssh", {}) or {}
    ssh_user = raw.get("user")
    remote_root_default = f"/home/{ssh_user}/auctrace" if ssh_user else "/home/auctrace"
    remote_root = raw.get("remote_root") or remote_root_default
    venv_path = raw.get("venv_path") or f"{remote_root}/.venv"
    windows_stage_dir_name = raw.get("windows_stage_dir_name") or "auctrace-stage"
    return {
        "host": raw.get("host"),
        "host_candidates_env": raw.get(
            "host_candidates_env",
            "AI_SCIENTIST_WSL_SSH_HOST_CANDIDATES",
        ),
        "port": _as_int(raw.get("port", 22), 22),
        "user": raw.get("user"),
        "private_key_path_env": raw.get(
            "private_key_path_env", "AI_SCIENTIST_WSL_SSH_PRIVATE_KEY_PATH"
        ),
        "password_env": raw.get("password_env", "AI_SCIENTIST_WSL_SSH_PASSWORD"),
        "connect_timeout": _as_int(raw.get("connect_timeout", 30), 30),
        "command_timeout": _as_int(raw.get("command_timeout", 1800), 1800),
        "setup_timeout": _as_int(raw.get("setup_timeout", 1800), 1800),
        "readiness_timeout": _as_int(raw.get("readiness_timeout", 120), 120),
        "wsl_distro": raw.get("wsl_distro", "Ubuntu"),
        "remote_root": remote_root,
        "runs_root": f"{remote_root}/runs",
        "venv_path": venv_path,
        "windows_stage_dir_name": windows_stage_dir_name,
        "install_project_requirements": _as_bool(
            raw.get("install_project_requirements", True), True
        ),
        "requirements_file": raw.get("requirements_file", "requirements.txt"),
        "pip_install_timeout": _as_int(raw.get("pip_install_timeout", 1800), 1800),
        "cleanup_remote_workspace": _as_bool(
            raw.get("cleanup_remote_workspace", True), True
        ),
        "cleanup_stage_archive": _as_bool(
            raw.get("cleanup_stage_archive", True), True
        ),
        "max_archive_mb": _as_int(
            raw.get("max_archive_mb", DEFAULT_MAX_ARCHIVE_MB),
            DEFAULT_MAX_ARCHIVE_MB,
        ),
        "min_free_disk_gb": _as_float(
            raw.get("min_free_disk_gb", DEFAULT_MIN_FREE_DISK_GB),
            DEFAULT_MIN_FREE_DISK_GB,
        ),
        "min_free_memory_gb": _as_float(
            raw.get("min_free_memory_gb", DEFAULT_MIN_FREE_MEMORY_GB),
            DEFAULT_MIN_FREE_MEMORY_GB,
        ),
        "recommended_workers": _as_int(
            raw.get("recommended_workers", DEFAULT_RECOMMENDED_WORKERS),
            DEFAULT_RECOMMENDED_WORKERS,
        ),
        "auto_install_missing_packages": _as_bool(
            raw.get("auto_install_missing_packages", True), True
        ),
        "max_auto_dependency_installs": _as_int(
            raw.get("max_auto_dependency_installs", 3),
            3,
        ),
        "runtime": raw.get("runtime"),
    }


def resolve_private_key_path(settings: dict[str, Any]) -> str | None:
    env_name = settings["private_key_path_env"]
    if not env_name:
        return None
    from os import environ

    value = environ.get(env_name)
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Configured WSL SSH private key does not exist: {path}")
    return str(path)


def configured_host_candidates(settings: dict[str, Any]) -> list[str]:
    env_name = str(settings.get("host_candidates_env") or "").strip()
    if not env_name:
        return []
    raw_value = os.environ.get(env_name, "")
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def runtime_payload(runtime: WslSshRuntime) -> dict[str, Any]:
    return {
        "ssh_host": runtime.ssh_host,
        "ssh_port": runtime.ssh_port,
        "ssh_user": runtime.ssh_user,
        "private_key_path": runtime.private_key_path,
        "remote_root": runtime.remote_root,
        "runs_root": runtime.runs_root,
        "venv_path": runtime.venv_path,
        "windows_stage_dir": runtime.windows_stage_dir,
        "windows_stage_dir_name": runtime.windows_stage_dir_name,
        "wsl_stage_dir": runtime.wsl_stage_dir,
        "distro_name": runtime.distro_name,
        "num_gpus": runtime.num_gpus,
        "recommended_workers": runtime.recommended_workers,
        "cpu_count": runtime.cpu_count,
        "free_disk_gb": runtime.free_disk_gb,
        "available_memory_gb": runtime.available_memory_gb,
    }


def set_runtime(exec_cfg: Any, runtime: WslSshRuntime) -> None:
    payload = runtime_payload(runtime)
    if isinstance(exec_cfg.wsl_ssh, dict):
        exec_cfg.wsl_ssh["runtime"] = payload
        return
    exec_cfg.wsl_ssh.runtime = payload


def clear_runtime(exec_cfg: Any) -> None:
    if isinstance(exec_cfg.wsl_ssh, dict):
        exec_cfg.wsl_ssh["runtime"] = None
        return
    exec_cfg.wsl_ssh.runtime = None
