from __future__ import annotations

import base64
import os
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import PurePosixPath

import pexpect

from .wsl_ssh_common import WslSshRuntime

PASSWORD_PROMPT = r"(?i)password:"


@dataclass(frozen=True)
class WslSshAuth:
    host: str
    port: int
    user: str
    private_key_path: str | None
    password: str | None


def auth_from_settings(settings: dict[str, object]) -> WslSshAuth:
    password_env = str(settings["password_env"])
    password = os.environ.get(password_env) or None
    return WslSshAuth(
        host=str(settings["host"]),
        port=int(settings["port"]),
        user=str(settings["user"]),
        private_key_path=settings.get("private_key_path"),  # type: ignore[arg-type]
        password=password,
    )


def auth_from_runtime(runtime: WslSshRuntime, password: str | None = None) -> WslSshAuth:
    return WslSshAuth(
        host=runtime.ssh_host,
        port=runtime.ssh_port,
        user=runtime.ssh_user,
        private_key_path=runtime.private_key_path,
        password=password,
    )


def ssh_base_args(auth: WslSshAuth, connect_timeout: int) -> list[str]:
    args = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        f"ConnectTimeout={connect_timeout}",
        "-p",
        str(auth.port),
    ]
    if auth.private_key_path:
        args += ["-i", auth.private_key_path]
    elif not auth.password:
        args += ["-o", "BatchMode=yes"]
    args.append(f"{auth.user}@{auth.host}")
    return args


def scp_base_args(auth: WslSshAuth, connect_timeout: int) -> list[str]:
    args = [
        "scp",
        "-P",
        str(auth.port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        f"ConnectTimeout={connect_timeout}",
    ]
    if auth.private_key_path:
        args += ["-i", auth.private_key_path]
    elif not auth.password:
        args += ["-o", "BatchMode=yes"]
    return args


def run_command(
    argv: list[str],
    timeout: int,
    password: str | None,
    *,
    stream_output: bool = False,
    output_prefix: str | None = None,
) -> subprocess.CompletedProcess:
    if not password:
        if not stream_output:
            return subprocess.run(argv, text=True, capture_output=True, timeout=timeout)

        proc = subprocess.Popen(
            argv,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
        )

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        prefix = output_prefix or ""

        def _reader(
            pipe,
            chunks: list[str],
            out_stream,
        ) -> None:
            try:
                for line in iter(pipe.readline, ""):
                    chunks.append(line)
                    if prefix:
                        out_stream.write(prefix)
                    out_stream.write(line)
                    out_stream.flush()
            finally:
                try:
                    pipe.close()
                except Exception:
                    pass

        threads: list[threading.Thread] = []
        if proc.stdout is not None:
            t = threading.Thread(
                target=_reader,
                args=(proc.stdout, stdout_chunks, sys.stdout),
                daemon=True,
            )
            t.start()
            threads.append(t)
        if proc.stderr is not None:
            t = threading.Thread(
                target=_reader,
                args=(proc.stderr, stderr_chunks, sys.stderr),
                daemon=True,
            )
            t.start()
            threads.append(t)

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            raise TimeoutError(
                f"Timed out waiting for command: {shlex.join(argv)}"
            ) from exc
        finally:
            for t in threads:
                t.join(timeout=1)

        return subprocess.CompletedProcess(
            argv,
            int(proc.returncode or 0),
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
        )

    child = pexpect.spawn(
        shlex.join(argv),
        encoding="utf-8",
        codec_errors="ignore",
        timeout=timeout,
    )
    chunks: list[str] = []
    prefix = output_prefix or ""
    password_sent = False
    try:
        while True:
            index = child.expect([PASSWORD_PROMPT, pexpect.EOF, pexpect.TIMEOUT])
            if index == 0:
                chunks.append(child.before)
                if stream_output and child.before:
                    if prefix:
                        sys.stdout.write(prefix)
                    sys.stdout.write(child.before)
                    sys.stdout.flush()
                if password_sent:
                    raise RuntimeError("SSH password prompt repeated after password submission")
                child.sendline(password)
                password_sent = True
                continue
            if index == 1:
                chunks.append(child.before)
                if stream_output and child.before:
                    if prefix:
                        sys.stdout.write(prefix)
                    sys.stdout.write(child.before)
                    sys.stdout.flush()
                break
            raise TimeoutError(f"Timed out waiting for command: {shlex.join(argv)}")
    finally:
        child.close(force=True)
    code = child.exitstatus
    if code is None:
        code = 0 if child.signalstatus is None else int(child.signalstatus)
    return subprocess.CompletedProcess(argv, int(code), stdout="".join(chunks), stderr="")


def build_powershell_command(script: str) -> str:
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    return f"powershell -NoProfile -NonInteractive -EncodedCommand {encoded}"


def run_windows_command(
    auth: WslSshAuth,
    script: str,
    timeout: int,
    connect_timeout: int,
    *,
    stream_output: bool = False,
    output_prefix: str | None = None,
) -> subprocess.CompletedProcess:
    command = build_powershell_command("$ProgressPreference='SilentlyContinue';" + script)
    argv = ssh_base_args(auth, connect_timeout) + [command]
    return run_command(
        argv,
        timeout,
        auth.password,
        stream_output=stream_output,
        output_prefix=output_prefix,
    )


def _quote_single(text: str) -> str:
    return text.replace("'", "''")


def run_wsl_command(
    auth: WslSshAuth,
    bash_script: str,
    timeout: int,
    connect_timeout: int,
    distro_name: str | None,
    *,
    stream_output: bool = False,
    output_prefix: str | None = None,
) -> subprocess.CompletedProcess:
    payload = base64.b64encode(bash_script.encode("utf-8")).decode("ascii")
    distro = f"-d '{_quote_single(distro_name)}' " if distro_name else ""
    script = (
        "$ProgressPreference='SilentlyContinue';"
        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
        f"$payload='{payload}';"
        "$bash=[System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($payload));"
        "$bash=$bash -replace \"`r\", \"\";"
        "$tmp=Join-Path $env:TEMP ('auctrace-' + [guid]::NewGuid().ToString() + '.sh');"
        "[System.IO.File]::WriteAllText($tmp, $bash, [System.Text.UTF8Encoding]::new($false));"
        "$drive=$tmp.Substring(0,1).ToLower();"
        "$rest=$tmp.Substring(2).Replace('\\','/');"
        "$wslPath='/mnt/' + $drive + $rest;"
        f"wsl {distro}-e bash $wslPath;"
        "$code=$LASTEXITCODE;"
        "Remove-Item -Force $tmp -ErrorAction SilentlyContinue;"
        "exit $code"
    )
    return run_windows_command(
        auth,
        script,
        timeout,
        connect_timeout,
        stream_output=stream_output,
        output_prefix=output_prefix,
    )


def ensure_windows_stage_dir(
    auth: WslSshAuth,
    stage_dir_name: str,
    timeout: int,
    connect_timeout: int,
) -> subprocess.CompletedProcess:
    script = (
        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
        f"$dir=Join-Path $env:USERPROFILE '{_quote_single(stage_dir_name)}';"
        "New-Item -ItemType Directory -Force -Path $dir | Out-Null;"
        "Write-Output $dir"
    )
    return run_windows_command(auth, script, timeout, connect_timeout)


def remote_windows_path(stage_dir_name: str, archive_name: str) -> str:
    return f"{stage_dir_name}/{archive_name}"


def windows_to_wsl_path(windows_path: str) -> str:
    normalized = windows_path.replace("\\", "/")
    drive, rest = normalized.split(":/", 1)
    return f"/mnt/{drive.lower()}/{rest}"


def quote_posix_path(path: PurePosixPath | str) -> str:
    return shlex.quote(str(path))
