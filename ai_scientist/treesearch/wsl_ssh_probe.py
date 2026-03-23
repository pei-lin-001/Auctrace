from __future__ import annotations

import json
import ipaddress
import logging
import re
import shutil
import socket
import subprocess
from typing import Any

from .wsl_ssh_client import (
    ssh_base_args,
    run_command,
    WslSshAuth,
    ensure_windows_stage_dir,
    run_windows_command,
    run_wsl_command,
    windows_to_wsl_path,
)
from .wsl_ssh_common import configured_host_candidates

logger = logging.getLogger("ai-scientist")
SSH_UNREACHABLE_MARKERS = (
    "operation timed out",
    "connection timed out",
    "no route to host",
    "network is unreachable",
    "could not resolve hostname",
    "name or service not known",
    "connection refused",
)

PROBE_WINDOWS_SCRIPT = """
[Console]::OutputEncoding=[System.Text.Encoding]::UTF8
$profile = $env:USERPROFILE
$payload = @{
  username = $env:USERNAME
  userprofile = $profile
  hostname = $env:COMPUTERNAME
}
$payload | ConvertTo-Json -Compress
"""

PROBE_WSL_SCRIPT = r"""
set -euo pipefail
python3 - <<'PY'
import json
import os
import shutil
import subprocess

gpu_count = 0
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    gpu_count = len([line for line in result.stdout.splitlines() if line.strip()])
except Exception:
    gpu_count = 0

mem_available = 0.0
with open("/proc/meminfo", "r", encoding="utf-8") as handle:
    for line in handle:
        if line.startswith("MemAvailable:"):
            mem_available = int(line.split()[1]) / (1024 * 1024)
            break

payload = {
    "home": os.path.expanduser("~"),
    "cpu_count": os.cpu_count() or 1,
    "free_disk_gb": shutil.disk_usage("/").free / (1024 ** 3),
    "available_memory_gb": mem_available,
    "num_gpus": gpu_count,
}
print(json.dumps(payload))
PY
"""


def load_json_line(text: str, source: str, prefix: str) -> Any:
    for line in reversed(text.splitlines()):
        candidate = line.strip().replace("\x00", "")
        if not candidate.startswith(prefix):
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"Failed to parse {source} probe output:\n{text}")


def load_json_object(text: str, source: str) -> dict[str, Any]:
    return load_json_line(text, source, "{")


def load_json_array(text: str, source: str) -> list[str]:
    return load_json_line(text, source, "[")


def validate_settings(settings: dict[str, Any]) -> None:
    if not settings["host"]:
        raise RuntimeError("Missing WSL SSH host; set AI_SCIENTIST_WSL_SSH_HOST.")
    if not settings["user"]:
        raise RuntimeError("Missing WSL SSH user; set AI_SCIENTIST_WSL_SSH_USER.")


def _resolve_host(host: str) -> str:
    try:
        return socket.gethostbyname(host)
    except OSError as exc:
        raise RuntimeError(
            f"WSL SSH host could not be resolved: {host} ({exc})"
        ) from exc


def _ssh_probe_text(host: str, port: int, user: str, timeout: int) -> str:
    auth = WslSshAuth(
        host=host,
        port=port,
        user=user,
        private_key_path=None,
        password=None,
    )
    argv = ssh_base_args(auth, timeout) + ["exit"]
    result = run_command(argv, timeout + 2, None)
    return f"{result.stdout}\n{result.stderr}".strip()


def _build_probe_error(
    host: str,
    port: int,
    user: str,
    timeout: int,
) -> RuntimeError | None:
    ip = _resolve_host(host)
    text = _ssh_probe_text(host, port, user, timeout)
    lowered = text.lower()
    if any(marker in lowered for marker in SSH_UNREACHABLE_MARKERS):
        return RuntimeError(
            f"Resolved {host} -> {ip}, but ssh probe to {ip}:{port} failed: {text}"
        )
    try:
        socket.getaddrinfo(host, port)
        return None
    except OSError as exc:
        return RuntimeError(f"Resolved {host} -> {ip}, but address info failed: {exc}")


def _extract_tailscale_lan_host(host: str, timeout: int) -> str | None:
    if not shutil.which("tailscale"):
        return None
    command = ["tailscale", "ping", "--verbose", "-c", "1", host]
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"via (\d+\.\d+\.\d+\.\d+):\d+", result.stdout)
    if match is None:
        return None
    candidate = match.group(1)
    try:
        ip = ipaddress.ip_address(candidate)
    except ValueError:
        return None
    if not ip.is_private:
        return None
    return candidate


def _candidate_hosts(settings: dict[str, Any]) -> list[tuple[str, str]]:
    primary = str(settings["host"])
    candidates = [(primary, "configured host")]
    for item in configured_host_candidates(settings):
        if item != primary and item not in [host for host, _ in candidates]:
            candidates.append((item, "configured candidate"))
    derived = _extract_tailscale_lan_host(primary, int(settings["connect_timeout"]))
    if derived and derived not in [host for host, _ in candidates]:
        candidates.append((derived, f"derived from tailscale route for {primary}"))
    return candidates


def select_reachable_host(settings: dict[str, Any]) -> tuple[str, str]:
    port = int(settings["port"])
    user = str(settings["user"])
    timeout = int(settings["connect_timeout"])
    failures: list[str] = []
    for host, reason in _candidate_hosts(settings):
        error = _build_probe_error(host, port, user, timeout)
        if error is None:
            return host, reason
        failures.append(f"{host} ({reason}): {error}")
    raise RuntimeError(
        "WSL SSH target is not reachable on any candidate host.\n"
        + "\n".join(failures)
    )


def probe_windows(auth: WslSshAuth, settings: dict[str, Any]) -> dict[str, Any]:
    result = run_windows_command(
        auth,
        PROBE_WINDOWS_SCRIPT,
        settings["readiness_timeout"],
        settings["connect_timeout"],
    )
    if result.returncode != 0:
        raise RuntimeError(f"Windows SSH probe failed:\n{result.stdout}\n{result.stderr}")
    return load_json_object(result.stdout, "windows")


def probe_wsl(auth: WslSshAuth, settings: dict[str, Any]) -> dict[str, Any]:
    try:
        result = run_wsl_command(
            auth,
            PROBE_WSL_SCRIPT,
            settings["readiness_timeout"],
            settings["connect_timeout"],
            settings["wsl_distro"],
        )
    except TimeoutError as exc:
        distro = settings.get("wsl_distro") or "default"
        raise RuntimeError(
            "WSL probe timed out while launching the remote distro over Windows SSH. "
            f"SSH to {auth.user}@{auth.host}:{auth.port} worked, but "
            f"`wsl -d {distro}` did not return within {settings['readiness_timeout']}s. "
            "This usually means the Windows host cannot start WSL from the current non-interactive SSH session."
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(f"WSL probe failed:\n{result.stdout}\n{result.stderr}")
    return load_json_object(result.stdout, "wsl")


def stage_paths(
    auth: WslSshAuth,
    settings: dict[str, Any],
    windows_info: dict[str, Any],
) -> tuple[str, str]:
    stage = ensure_windows_stage_dir(
        auth,
        settings["windows_stage_dir_name"],
        settings["readiness_timeout"],
        settings["connect_timeout"],
    )
    if stage.returncode != 0:
        raise RuntimeError(f"Failed to create Windows staging dir:\n{stage.stdout}\n{stage.stderr}")
    windows_stage_dir = ""
    for line in stage.stdout.splitlines():
        candidate = line.replace("\x00", "").strip()
        if ":\\" in candidate:
            windows_stage_dir = candidate
    if not windows_stage_dir:
        windows_stage_dir = f"{windows_info['userprofile']}\\{settings['windows_stage_dir_name']}"
    return windows_stage_dir, windows_to_wsl_path(windows_stage_dir)


def validate_resources(settings: dict[str, Any], probe: dict[str, Any]) -> None:
    if probe["free_disk_gb"] < settings["min_free_disk_gb"]:
        raise RuntimeError(
            "WSL free disk is below the configured safety threshold: "
            f"{probe['free_disk_gb']:.1f} GiB < {settings['min_free_disk_gb']} GiB"
        )
    if probe["available_memory_gb"] < settings["min_free_memory_gb"]:
        raise RuntimeError(
            "WSL available memory is below the configured safety threshold: "
            f"{probe['available_memory_gb']:.1f} GiB < {settings['min_free_memory_gb']} GiB"
        )
