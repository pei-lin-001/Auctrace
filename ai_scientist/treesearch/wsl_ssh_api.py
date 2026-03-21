from __future__ import annotations

import time
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
    started = time.monotonic()
    print("[wsl_ssh] preparing runtime (first run can take a few minutes)...")
    settings = normalize_wsl_ssh_settings(exec_cfg)
    if settings["runtime"]:
        runtime = WslSshRuntime(**dict(settings["runtime"]))
        elapsed = time.monotonic() - started
        print(
            "[wsl_ssh] reusing cached runtime "
            f"({runtime.ssh_user}@{runtime.ssh_host}:{runtime.ssh_port}) in {elapsed:.1f}s"
        )
        return runtime
    validate_settings(settings)
    print(
        "[wsl_ssh] probing SSH reachability "
        f"({settings['user']}@{settings['host']}:{settings['port']}, distro={settings['wsl_distro']})..."
    )
    t0 = time.monotonic()
    selected_host, selection_reason = select_reachable_host(settings)
    print(
        f"[wsl_ssh] selected host: {selected_host} ({selection_reason}) "
        f"in {time.monotonic() - t0:.1f}s"
    )
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
    print("[wsl_ssh] probing Windows host info...")
    t0 = time.monotonic()
    windows_info = probe_windows(auth, settings)
    print(f"[wsl_ssh] Windows probe ok in {time.monotonic() - t0:.1f}s")

    print("[wsl_ssh] probing WSL resources (CPU/RAM/disk/GPU)...")
    t0 = time.monotonic()
    probe = probe_wsl(auth, settings)
    print(
        "[wsl_ssh] WSL probe ok in "
        f"{time.monotonic() - t0:.1f}s (gpus={probe.get('num_gpus')}, cpu={probe.get('cpu_count')})"
    )
    validate_resources(settings, probe)
    print("[wsl_ssh] ensuring staging directories...")
    t0 = time.monotonic()
    windows_stage_dir, wsl_stage_dir = stage_paths(auth, settings, windows_info)
    print(f"[wsl_ssh] staging ok in {time.monotonic() - t0:.1f}s")

    print("[wsl_ssh] setting up remote venv (pip upgrade included)...")
    t0 = time.monotonic()
    setup_runtime(auth, settings)
    print(f"[wsl_ssh] venv setup ok in {time.monotonic() - t0:.1f}s")

    if settings.get("install_project_requirements"):
        print(
            "[wsl_ssh] installing project requirements "
            f"(timeout={settings.get('pip_install_timeout')}s)..."
        )
    else:
        print("[wsl_ssh] skipping project requirements install (install_project_requirements=false)")
    t0 = time.monotonic()
    install_requirements(auth, settings, settings["windows_stage_dir_name"], wsl_stage_dir)
    print(f"[wsl_ssh] requirements step done in {time.monotonic() - t0:.1f}s")
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
    elapsed = time.monotonic() - started
    print(f"[wsl_ssh] runtime ready in {elapsed:.1f}s")
    return runtime


def cleanup_wsl_ssh_runtime(exec_cfg: Any) -> None:
    if not normalize_wsl_ssh_settings(exec_cfg)["runtime"]:
        return
    clear_runtime(exec_cfg)
