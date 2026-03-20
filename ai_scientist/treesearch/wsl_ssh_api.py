from __future__ import annotations

import logging
from typing import Any

from .wsl_ssh_client import auth_from_settings
from .wsl_ssh_common import (
    WslSshRuntime,
    clear_runtime,
    normalize_wsl_ssh_settings,
    resolve_private_key_path,
    set_runtime,
)
from .wsl_ssh_install import install_requirements, setup_runtime
from .wsl_ssh_probe import (
    probe_windows,
    probe_wsl,
    select_reachable_host,
    stage_paths,
    validate_resources,
    validate_settings,
)

logger = logging.getLogger("ai-scientist")


def prepare_wsl_ssh_runtime(exec_cfg: Any, exp_name: str) -> WslSshRuntime:
    del exp_name
    settings = normalize_wsl_ssh_settings(exec_cfg)
    if settings["runtime"]:
        return WslSshRuntime(**dict(settings["runtime"]))
    validate_settings(settings)
    selected_host, selection_reason = select_reachable_host(settings)
    if selected_host != settings["host"]:
        logger.warning(
            "WSL SSH primary host %s is unreachable; using %s (%s)",
            settings["host"],
            selected_host,
            selection_reason,
        )
    else:
        logger.info("WSL SSH using %s (%s)", selected_host, selection_reason)
    settings["host"] = selected_host
    settings["private_key_path"] = resolve_private_key_path(settings)
    auth = auth_from_settings(settings)
    windows_info = probe_windows(auth, settings)
    probe = probe_wsl(auth, settings)
    validate_resources(settings, probe)
    windows_stage_dir, wsl_stage_dir = stage_paths(auth, settings, windows_info)
    setup_runtime(auth, settings)
    install_requirements(auth, settings, settings["windows_stage_dir_name"], wsl_stage_dir)
    runtime = WslSshRuntime(
        ssh_host=str(settings["host"]),
        ssh_port=int(settings["port"]),
        ssh_user=str(settings["user"]),
        private_key_path=settings["private_key_path"],
        remote_root=str(settings["remote_root"]),
        runs_root=str(settings["runs_root"]),
        venv_path=str(settings["venv_path"]),
        windows_stage_dir=windows_stage_dir,
        windows_stage_dir_name=str(settings["windows_stage_dir_name"]),
        wsl_stage_dir=wsl_stage_dir,
        distro_name=str(settings["wsl_distro"]) if settings["wsl_distro"] else None,
        num_gpus=int(probe["num_gpus"]),
        recommended_workers=int(settings["recommended_workers"]),
        cpu_count=int(probe["cpu_count"]),
        free_disk_gb=float(probe["free_disk_gb"]),
        available_memory_gb=float(probe["available_memory_gb"]),
    )
    set_runtime(exec_cfg, runtime)
    return runtime


def cleanup_wsl_ssh_runtime(exec_cfg: Any) -> None:
    if not normalize_wsl_ssh_settings(exec_cfg)["runtime"]:
        return
    clear_runtime(exec_cfg)
