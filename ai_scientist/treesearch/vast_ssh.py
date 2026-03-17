from __future__ import annotations

import os
import re
import shlex
import socket
import subprocess
import time
from hashlib import sha256
from ipaddress import ip_address
from pathlib import Path, PurePosixPath
from typing import Any

from .vast_common import VastRuntime, read_text_if_set

DEFAULT_KEY_CANDIDATES = ("id_ed25519", "id_rsa")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_DIRNAME = ".auctrace_bootstrap"


def _public_key_path(private_path: Path) -> Path:
    return Path(f"{private_path}.pub")


def _validate_public_key_text(key_text: str) -> str:
    parts = key_text.strip().split()
    if len(parts) < 2 or not parts[0].startswith("ssh-"):
        raise RuntimeError(
            "Vast.ai remote execution requires a real SSH public key. "
            "The configured key does not look like an SSH public key."
        )
    return key_text.strip()


def _default_keypair() -> tuple[str | None, str]:
    ssh_dir = Path.home() / ".ssh"
    for name in DEFAULT_KEY_CANDIDATES:
        private_path = ssh_dir / name
        public_path = _public_key_path(private_path)
        if private_path.exists() and public_path.exists():
            return str(private_path), _validate_public_key_text(public_path.read_text())
    raise RuntimeError(
        "No usable local SSH keypair found for Vast.ai. "
        "Set VAST_SSH_PRIVATE_KEY_PATH and VAST_SSH_PUBLIC_KEY_PATH, "
        "or create ~/.ssh/id_ed25519(.pub)."
    )


def resolve_local_ssh_credentials(settings: dict[str, Any]) -> tuple[str | None, str]:
    private_env = os.environ.get(settings["private_ssh_key_path_env"])
    public_env = os.environ.get(settings["public_ssh_key_path_env"])
    if private_env:
        private_path = Path(private_env).expanduser()
        if not private_path.exists():
            raise FileNotFoundError(f"Configured Vast.ai SSH private key does not exist: {private_path}")
        public_text = read_text_if_set(public_env) or read_text_if_set(str(_public_key_path(private_path)))
        if public_text is None:
            raise RuntimeError(
                f"Missing SSH public key for {private_path}. "
                "Set VAST_SSH_PUBLIC_KEY_PATH or provide the matching .pub file."
            )
        return str(private_path), _validate_public_key_text(public_text)
    if public_env:
        public_text = read_text_if_set(public_env)
        if public_text is None:
            raise RuntimeError("Configured Vast.ai SSH public key path is empty")
        return None, _validate_public_key_text(public_text)
    return _default_keypair()


def resolve_connect_host(hostname: str) -> str:
    try:
        resolved = socket.gethostbyname(hostname)
        return resolved
    except Exception:
        pass
    dig = subprocess.run(
        ["dig", "+short", hostname, "@1.1.1.1"],
        text=True,
        capture_output=True,
        timeout=15,
    )
    if dig.returncode != 0:
        return hostname
    for line in dig.stdout.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        try:
            return str(ip_address(candidate))
        except ValueError:
            continue
    return hostname


def ssh_base_args(runtime: VastRuntime, connect_timeout: int = 30) -> list[str]:
    args = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        f"ConnectTimeout={connect_timeout}",
        "-p",
        str(runtime.ssh_port),
    ]
    if runtime.private_key_path:
        args += ["-i", runtime.private_key_path]
    args.append(f"root@{runtime.ssh_host}")
    return args


def scp_base_args(runtime: VastRuntime, connect_timeout: int = 30) -> list[str]:
    args = [
        "scp",
        "-P",
        str(runtime.ssh_port),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        f"ConnectTimeout={connect_timeout}",
    ]
    if runtime.private_key_path:
        args += ["-i", runtime.private_key_path]
    return args


def run_ssh(
    runtime: VastRuntime,
    command: str,
    timeout: int,
    connect_timeout: int = 30,
) -> subprocess.CompletedProcess:
    ssh_cmd = ssh_base_args(runtime, connect_timeout=connect_timeout) + [command]
    return subprocess.run(ssh_cmd, text=True, capture_output=True, timeout=timeout)


def run_setup_commands(runtime: VastRuntime, settings: dict[str, Any]) -> None:
    install_project_requirements(runtime, settings)
    commands = settings["setup_commands"]
    if not commands:
        return
    marker = PurePosixPath(runtime.remote_root) / ".auctrace_setup_complete"
    joined = " && ".join(commands)
    command = (
        f"mkdir -p {shlex.quote(runtime.remote_root)} && "
        f"if [ ! -f {shlex.quote(str(marker))} ]; then "
        f"cd {shlex.quote(runtime.remote_root)} && {joined} && "
        f"touch {shlex.quote(str(marker))}; fi"
    )
    result = run_ssh(runtime, command, settings["setup_timeout"])
    if result.returncode != 0:
        raise RuntimeError(f"Vast.ai remote setup failed:\n{result.stdout}\n{result.stderr}")


def install_project_requirements(runtime: VastRuntime, settings: dict[str, Any]) -> None:
    if not settings["install_project_requirements"]:
        return
    requirements_path = (PROJECT_ROOT / settings["requirements_file"]).resolve()
    if not requirements_path.exists():
        raise FileNotFoundError(
            f"Configured requirements file does not exist: {requirements_path}"
        )
    requirements_hash = sha256(requirements_path.read_bytes()).hexdigest()[:12]
    remote_dir = PurePosixPath(runtime.remote_root) / BOOTSTRAP_DIRNAME
    remote_requirements = remote_dir / f"requirements-{requirements_hash}.txt"
    marker = remote_dir / f"requirements-{requirements_hash}.done"
    mkdir_result = run_ssh(
        runtime,
        f"mkdir -p {shlex.quote(str(remote_dir))}",
        timeout=120,
    )
    if mkdir_result.returncode != 0:
        raise RuntimeError(
            "Failed to create Vast.ai bootstrap directory:\n"
            f"{mkdir_result.stdout}\n{mkdir_result.stderr}"
        )
    scp_cmd = scp_base_args(runtime) + [
        str(requirements_path),
        f"root@{runtime.ssh_host}:{remote_requirements}",
    ]
    subprocess.run(
        scp_cmd,
        check=True,
        capture_output=True,
        text=True,
        timeout=settings["pip_install_timeout"],
    )
    install_cmd = (
        f"if [ ! -f {shlex.quote(str(marker))} ]; then "
        f"python -m pip install --disable-pip-version-check --no-input "
        f"--progress-bar off -r {shlex.quote(str(remote_requirements))} && "
        f"touch {shlex.quote(str(marker))}; fi"
    )
    result = run_ssh(runtime, install_cmd, settings["pip_install_timeout"])
    if result.returncode != 0:
        raise RuntimeError(
            "Vast.ai project dependency installation failed:\n"
            f"{result.stdout}\n{result.stderr}"
        )


def resolve_ssh_endpoint(client: Any, instance_id: int, instance: dict[str, Any]) -> tuple[str, int]:
    try:
        client.ssh_url(id=instance_id)
        endpoint = (client.last_output or "").strip()
        match = re.search(r"ssh://[^@]+@([^:]+):(\d+)", endpoint)
        if match:
            return resolve_connect_host(match.group(1)), int(match.group(2))
    except Exception:
        pass
    return resolve_connect_host(str(instance["ssh_host"])), int(instance["ssh_port"])


def read_ssh_banner(host: str, port: int, timeout: int) -> str:
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        banner = sock.recv(256).decode("utf-8", errors="replace").strip()
    if not banner:
        raise RuntimeError("SSH endpoint accepted TCP but returned no SSH banner")
    if not banner.startswith("SSH-"):
        raise RuntimeError(f"Unexpected SSH banner: {banner}")
    return banner


def probe_ssh_runtime(runtime: VastRuntime, settings: dict[str, Any]) -> None:
    retries = settings["ssh_probe_retries"]
    delay = settings["ssh_probe_retry_delay"]
    connect_timeout = settings["ssh_probe_connect_timeout"]
    probe_timeout = settings["ssh_probe_timeout"]
    errors: list[str] = []
    for attempt in range(1, retries + 1):
        try:
            banner = read_ssh_banner(
                runtime.ssh_host,
                runtime.ssh_port,
                connect_timeout,
            )
            result = run_ssh(
                runtime,
                "true",
                timeout=probe_timeout,
                connect_timeout=connect_timeout,
            )
            if result.returncode != 0:
                stderr = result.stderr.strip()
                if "Permission denied" in stderr:
                    raise RuntimeError(f"SSH auth rejected (key not synced to instance): {stderr}")
                raise RuntimeError(stderr or result.stdout.strip() or "ssh probe failed")
            if banner:
                return
        except Exception as exc:
            errors.append(f"attempt {attempt}/{retries}: {exc}")
            if attempt < retries:
                wait = delay * attempt
                time.sleep(wait)
    raise RuntimeError("Vast.ai SSH health check failed:\n" + "\n".join(errors))
