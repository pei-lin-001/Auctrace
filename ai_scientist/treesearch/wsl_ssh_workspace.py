from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path, PurePosixPath

from .remote_exec_common import create_archive, extract_archive
from .wsl_ssh_client import (
    WslSshAuth,
    quote_posix_path,
    remote_windows_path,
    run_command,
    run_wsl_command,
    scp_base_args,
)
from .wsl_ssh_common import WslSshRuntime


def guard_archive_size(archive_path: Path, max_archive_mb: int) -> None:
    size_mb = archive_path.stat().st_size / (1024 * 1024)
    if size_mb <= max_archive_mb:
        return
    raise RuntimeError(
        f"Workspace archive {archive_path.name} is {size_mb:.1f} MiB, "
        f"exceeding safety limit {max_archive_mb} MiB."
    )


def push_workspace(
    auth: WslSshAuth,
    runtime: WslSshRuntime,
    settings: dict[str, object],
    local_dir: Path,
    remote_workspace: PurePosixPath,
    timeout: int,
) -> None:
    archive = create_archive(local_dir)
    try:
        guard_archive_size(archive, int(settings["max_archive_mb"]))
        remote_name = f"{archive.name}-{uuid.uuid4().hex}"
        remote_windows = remote_windows_path(runtime.windows_stage_dir_name, remote_name)
        scp_cmd = scp_base_args(auth, int(settings["connect_timeout"])) + [
            str(archive),
            f"{auth.user}@{auth.host}:{remote_windows}",
        ]
        pushed = run_command(scp_cmd, timeout, auth.password)
        if pushed.returncode != 0:
            raise RuntimeError(f"Failed to upload WSL workspace:\n{pushed.stdout}\n{pushed.stderr}")
        _unpack_workspace(auth, runtime, settings, remote_workspace, remote_name, timeout)
    finally:
        shutil.rmtree(archive.parent, ignore_errors=True)


def _unpack_workspace(
    auth: WslSshAuth,
    runtime: WslSshRuntime,
    settings: dict[str, object],
    remote_workspace: PurePosixPath,
    remote_name: str,
    timeout: int,
) -> None:
    remote_archive = PurePosixPath(runtime.wsl_stage_dir) / remote_name
    script = "\n".join(
        [
            "set -euo pipefail",
            f"mkdir -p {quote_posix_path(remote_workspace.parent)}",
            f"rm -rf {quote_posix_path(remote_workspace)}",
            (
                f"tar -xzf {quote_posix_path(remote_archive)} "
                f"-C {quote_posix_path(remote_workspace.parent)}"
            ),
            *(
                [f"rm -f {quote_posix_path(remote_archive)}"]
                if settings["cleanup_stage_archive"]
                else []
            ),
        ]
    )
    result = run_wsl_command(
        auth,
        script,
        timeout,
        int(settings["connect_timeout"]),
        runtime.distro_name,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to sync workspace to WSL:\n{result.stdout}\n{result.stderr}")


def pull_workspace(
    auth: WslSshAuth,
    runtime: WslSshRuntime,
    settings: dict[str, object],
    local_dir: Path,
    remote_workspace: PurePosixPath,
    timeout: int,
) -> None:
    remote_name = f"{local_dir.name}-{uuid.uuid4().hex}.tar.gz"
    remote_archive = PurePosixPath(runtime.wsl_stage_dir) / remote_name
    archive_dir = Path(tempfile.mkdtemp(prefix="auctrace-wsl-pull-"))
    local_archive = archive_dir / "workspace.tar.gz"
    try:
        _pack_workspace(auth, runtime, settings, remote_workspace, remote_archive, timeout)
        _download_archive(auth, runtime, settings, local_archive, remote_name, timeout)
        shutil.rmtree(local_dir, ignore_errors=True)
        extract_archive(local_archive, local_dir.parent)
    finally:
        shutil.rmtree(archive_dir, ignore_errors=True)


def _pack_workspace(
    auth: WslSshAuth,
    runtime: WslSshRuntime,
    settings: dict[str, object],
    remote_workspace: PurePosixPath,
    remote_archive: PurePosixPath,
    timeout: int,
) -> None:
    script = [
        "set -euo pipefail",
        f"cd {quote_posix_path(remote_workspace.parent)}",
        f"tar -czf {quote_posix_path(remote_archive)} {quote_posix_path(remote_workspace.name)}",
    ]
    if settings["cleanup_remote_workspace"]:
        script.extend(
            [
                f"rm -rf {quote_posix_path(remote_workspace)}",
                (
                    f"if [ -d {quote_posix_path(remote_workspace.parent)} ] && "
                    f"[ -z \"$(ls -A {quote_posix_path(remote_workspace.parent)})\" ]; then "
                    f"rmdir {quote_posix_path(remote_workspace.parent)}; fi"
                ),
            ]
        )
    result = run_wsl_command(
        auth,
        "\n".join(script),
        timeout,
        int(settings["connect_timeout"]),
        runtime.distro_name,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to pack WSL workspace:\n{result.stdout}\n{result.stderr}")


def _download_archive(
    auth: WslSshAuth,
    runtime: WslSshRuntime,
    settings: dict[str, object],
    local_archive: Path,
    remote_name: str,
    timeout: int,
) -> None:
    remote_windows = remote_windows_path(runtime.windows_stage_dir_name, remote_name)
    scp_cmd = scp_base_args(auth, int(settings["connect_timeout"])) + [
        f"{auth.user}@{auth.host}:{remote_windows}",
        str(local_archive),
    ]
    pulled = run_command(scp_cmd, timeout, auth.password)
    if pulled.returncode != 0:
        raise RuntimeError(f"Failed to download WSL workspace:\n{pulled.stdout}\n{pulled.stderr}")
    guard_archive_size(local_archive, int(settings["max_archive_mb"]))
    if not settings["cleanup_stage_archive"]:
        return
    remote_archive = PurePosixPath(runtime.wsl_stage_dir) / remote_name
    run_wsl_command(
        auth,
        f"rm -f {quote_posix_path(remote_archive)}",
        timeout,
        int(settings["connect_timeout"]),
        runtime.distro_name,
    )
