from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from .dependency_resolution import extract_missing_module, package_name_for_module
from .interpreter import ExecutionResult
from .remote_exec_common import (
    RESULT_END,
    RESULT_START,
    RUNNER_FILE,
    write_runner,
)
from .wsl_ssh_client import (
    auth_from_runtime,
    quote_posix_path,
    run_wsl_command,
)
from .wsl_ssh_common import WslSshRuntime, normalize_wsl_ssh_settings
from .wsl_ssh_workspace import pull_workspace, push_workspace

class WslSshRemoteInterpreter:
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int,
        agent_file_name: str,
        wsl_ssh_cfg: Any,
        run_name: str,
        gpu_id: int | None = None,
        env_vars: dict[str, str] | None = None,
    ):
        runtime_data = wsl_ssh_cfg.get("runtime") if isinstance(wsl_ssh_cfg, dict) else wsl_ssh_cfg.runtime
        if runtime_data is None:
            raise RuntimeError("WSL SSH runtime has not been prepared before interpreter creation")
        self.working_dir = Path(working_dir).resolve()
        self.timeout = timeout
        self.agent_file_name = agent_file_name
        self.runtime = WslSshRuntime(**dict(runtime_data))
        self.run_name = run_name
        self.gpu_id = gpu_id
        self.env_vars = env_vars or {}
        self.settings = normalize_wsl_ssh_settings(type("ExecProxy", (), {"wsl_ssh": wsl_ssh_cfg})())
        password = os.environ.get(self.settings["password_env"])
        self.auth = auth_from_runtime(self.runtime, password=password)
        self.auto_install_missing_packages = self.settings["auto_install_missing_packages"]
        self.max_auto_dependency_installs = self.settings["max_auto_dependency_installs"]
        self.pip_install_timeout = self.settings["pip_install_timeout"]

    def cleanup_session(self) -> None:
        return None

    def run(self, code: str, reset_session: bool = True) -> ExecutionResult:
        del reset_session
        self.working_dir.mkdir(parents=True, exist_ok=True)
        (self.working_dir / self.agent_file_name).write_text(code)
        write_runner(self.working_dir, self.agent_file_name, self.timeout)
        remote_workspace = PurePosixPath(self.runtime.runs_root) / self.run_name / self.working_dir.name
        install_logs: list[str] = []
        attempted_packages: set[str] = set()
        try:
            self.sync_to_remote(remote_workspace)
            result = self._run_with_dependency_recovery(
                remote_workspace,
                install_logs,
                attempted_packages,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult([str(exc)], self.timeout, "TimeoutError")
        except Exception as exc:
            return ExecutionResult([str(exc)], 0.0, exc.__class__.__name__, {"args": [str(exc)]})
        finally:
            try:
                self.sync_from_remote(remote_workspace)
            except Exception:
                pass
        result.term_out.extend(install_logs)
        return result

    def _run_with_dependency_recovery(
        self,
        remote_workspace: PurePosixPath,
        install_logs: list[str],
        attempted_packages: set[str],
    ) -> ExecutionResult:
        while True:
            result = self._run_remote(remote_workspace)
            module_name = extract_missing_module(
                result.exc_type,
                result.exc_info,
                result.term_out,
            )
            if not self._should_auto_install(module_name, attempted_packages):
                return result
            package_name = package_name_for_module(module_name)
            install_logs.append(
                "[AUCTRACE_AUTO_INSTALL] "
                f"Detected missing Python module '{module_name}'. "
                f"Installing package '{package_name}' on WSL SSH backend and retrying."
            )
            try:
                self._install_missing_package(remote_workspace, package_name)
            except Exception as exc:
                result.term_out.extend(install_logs)
                result.term_out.append(
                    "[AUCTRACE_AUTO_INSTALL] "
                    f"Failed to install '{package_name}': {exc}"
                )
                result.exc_type = "DependencyInstallError"
                result.exc_info = {
                    "missing_module": module_name,
                    "package_name": package_name,
                    "install_error": str(exc),
                }
                return result
            attempted_packages.add(package_name)

    def _should_auto_install(
        self,
        module_name: str | None,
        attempted_packages: set[str],
    ) -> bool:
        if not self.auto_install_missing_packages or module_name is None:
            return False
        if len(attempted_packages) >= self.max_auto_dependency_installs:
            return False
        package_name = package_name_for_module(module_name)
        return package_name not in attempted_packages

    def _install_missing_package(
        self,
        remote_workspace: PurePosixPath,
        package_name: str,
    ) -> None:
        script = "\n".join(
            [
                "set -euo pipefail",
                f"source {quote_posix_path(self.runtime.venv_path)}/bin/activate",
                f"cd {quote_posix_path(remote_workspace)}",
                (
                    "python -m pip install --disable-pip-version-check --no-input "
                    f"--progress-bar off {quote_posix_path(package_name)}"
                ),
            ]
        )
        result = run_wsl_command(
            self.auth,
            script,
            self.pip_install_timeout,
            self.settings["connect_timeout"],
            self.runtime.distro_name,
        )
        if result.returncode != 0:
            raise RuntimeError(f"{result.stdout}\n{result.stderr}".strip())

    def _run_remote(self, remote_workspace: PurePosixPath) -> ExecutionResult:
        env_lines = []
        if self.gpu_id is not None:
            env_lines.append(f"export CUDA_VISIBLE_DEVICES={self.gpu_id}")
        for key, value in self.env_vars.items():
            env_lines.append(f"export {key}={quote_posix_path(value)}")
        script = [
            "set -euo pipefail",
            f"source {quote_posix_path(self.runtime.venv_path)}/bin/activate",
            *env_lines,
            f"cd {quote_posix_path(remote_workspace)}",
            f"python {RUNNER_FILE}",
        ]
        completed = run_wsl_command(
            self.auth,
            "\n".join(script),
            self.timeout + 60,
            self.settings["connect_timeout"],
            self.runtime.distro_name,
        )
        return self.parse_result(completed)

    def parse_result(self, completed: subprocess.CompletedProcess) -> ExecutionResult:
        stdout = completed.stdout.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
        stderr = completed.stderr.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
        match = re.search(
            rf"{RESULT_START}\n(.*?)\n{RESULT_END}",
            stdout,
            flags=re.DOTALL,
        )
        clean_stdout = re.sub(
            rf"\n?{RESULT_START}\n.*?\n{RESULT_END}\n?",
            "",
            stdout,
            flags=re.DOTALL,
        )
        if not match:
            return ExecutionResult(
                [clean_stdout, stderr],
                0.0,
                "RemoteCommandError",
                {"returncode": completed.returncode},
            )
        payload = json.loads(match.group(1))
        exc_stack = payload.get("exc_stack")
        if exc_stack is not None:
            exc_stack = [tuple(item) for item in exc_stack]
        return ExecutionResult(
            [clean_stdout, stderr],
            float(payload.get("exec_time", 0.0)),
            payload.get("exc_type"),
            payload.get("exc_info"),
            exc_stack,
        )

    def sync_to_remote(self, remote_workspace: PurePosixPath) -> None:
        push_workspace(
            self.auth,
            self.runtime,
            self.settings,
            self.working_dir,
            remote_workspace,
            self.timeout,
        )

    def sync_from_remote(self, remote_workspace: PurePosixPath) -> None:
        pull_workspace(
            self.auth,
            self.runtime,
            self.settings,
            self.working_dir,
            remote_workspace,
            self.timeout,
        )
