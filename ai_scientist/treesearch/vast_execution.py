from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from .dependency_resolution import extract_missing_module, package_name_for_module
from .interpreter import ExecutionResult
from .vast_common import VastRuntime, cfg_value
from .vast_ssh import run_ssh, scp_base_args

RESULT_START = "__AUCTRACE_VAST_RESULT_START__"
RESULT_END = "__AUCTRACE_VAST_RESULT_END__"
RUNNER_FILE = "__auctrace_vast_exec.py"


def create_archive(local_workspace: Path) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="auctrace-vast-"))
    archive_path = tmp_dir / f"{local_workspace.name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(local_workspace, arcname=local_workspace.name)
    return archive_path


def extract_archive(archive_path: Path, target_parent: Path) -> None:
    target_parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(target_parent)


def write_runner(local_workspace: Path, agent_file_name: str, timeout: int) -> None:
    runner_code = f"""import json, runpy, signal, time, traceback
START=time.time()
TIMEOUT={timeout}
def _timeout_handler(signum, frame):
    raise TimeoutError("Remote execution timed out")
signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(TIMEOUT)
payload={{"exc_type": None, "exc_info": None, "exc_stack": None, "exec_time": None}}
try:
    runpy.run_path("{agent_file_name}", run_name="__main__")
except BaseException as exc:
    payload["exc_type"] = exc.__class__.__name__
    payload["exc_info"] = {{"args": [str(arg) for arg in getattr(exc, "args", [])]}}
    for attr in ("name", "msg", "obj"):
        if hasattr(exc, attr):
            payload["exc_info"][attr] = str(getattr(exc, attr))
    payload["exc_stack"] = [[t.filename, t.lineno, t.name, t.line] for t in traceback.extract_tb(exc.__traceback__)]
    traceback.print_exc()
finally:
    signal.alarm(0)
    payload["exec_time"] = time.time() - START
    print("{RESULT_START}")
    print(json.dumps(payload))
    print("{RESULT_END}")
"""
    (local_workspace / RUNNER_FILE).write_text(runner_code)


class VastRemoteInterpreter:
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int,
        agent_file_name: str,
        vast_cfg: Any,
        run_name: str,
        gpu_id: int | None = None,
        env_vars: dict[str, str] | None = None,
    ):
        runtime_data = cfg_value(vast_cfg, "runtime")
        if runtime_data is None:
            raise RuntimeError("Vast.ai runtime has not been prepared before interpreter creation")
        self.working_dir = Path(working_dir).resolve()
        self.timeout = timeout
        self.agent_file_name = agent_file_name
        self.runtime = VastRuntime(**dict(runtime_data))
        self.run_name = run_name
        self.gpu_id = gpu_id
        self.env_vars = env_vars or {}
        self.auto_install_missing_packages = cfg_value(
            vast_cfg, "auto_install_missing_packages", True
        )
        self.max_auto_dependency_installs = cfg_value(
            vast_cfg, "max_auto_dependency_installs", 3
        )
        self.pip_install_timeout = cfg_value(vast_cfg, "pip_install_timeout", 1800)

    def cleanup_session(self) -> None:
        return None

    def run(self, code: str, reset_session: bool = True) -> ExecutionResult:
        del reset_session
        self.working_dir.mkdir(parents=True, exist_ok=True)
        (self.working_dir / self.agent_file_name).write_text(code)
        write_runner(self.working_dir, self.agent_file_name, self.timeout)
        remote_workspace = (
            PurePosixPath(self.runtime.remote_root) / self.run_name / self.working_dir.name
        )
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
            return ExecutionResult(term_out=[str(exc)], exec_time=self.timeout, exc_type="TimeoutError")
        except Exception as exc:
            return ExecutionResult(
                term_out=[str(exc)],
                exec_time=0.0,
                exc_type=exc.__class__.__name__,
                exc_info={"args": [str(arg) for arg in getattr(exc, "args", [])]},
            )
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
            completed = self.run_remote(remote_workspace)
            result = self.parse_result(completed)
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
                f"Installing package '{package_name}' on Vast and retrying."
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
        command = (
            f"cd {shlex.quote(str(remote_workspace))} && "
            "python -m pip install --disable-pip-version-check "
            f"--no-input --progress-bar off {shlex.quote(package_name)}"
        )
        completed = run_ssh(self.runtime, command, self.pip_install_timeout)
        if completed.returncode != 0:
            raise RuntimeError(f"{completed.stdout}\n{completed.stderr}".strip())

    def run_remote(self, remote_workspace: PurePosixPath) -> subprocess.CompletedProcess:
        env_parts = [f"CUDA_VISIBLE_DEVICES={self.gpu_id}"] if self.gpu_id is not None else []
        env_parts.extend(f"{key}={shlex.quote(value)}" for key, value in self.env_vars.items())
        env_prefix = " ".join(env_parts)
        command = (
            f"mkdir -p {shlex.quote(str(remote_workspace))} && "
            f"cd {shlex.quote(str(remote_workspace))} && "
            f"{env_prefix} python {RUNNER_FILE}"
        )
        return run_ssh(self.runtime, command, self.timeout + 60)

    def parse_result(self, completed: subprocess.CompletedProcess) -> ExecutionResult:
        match = re.search(
            rf"{RESULT_START}\n(.*?)\n{RESULT_END}",
            completed.stdout,
            flags=re.DOTALL,
        )
        clean_stdout = re.sub(
            rf"\n?{RESULT_START}\n.*?\n{RESULT_END}\n?",
            "",
            completed.stdout,
            flags=re.DOTALL,
        )
        if not match:
            return ExecutionResult(
                term_out=[clean_stdout, completed.stderr],
                exec_time=0.0,
                exc_type="RemoteCommandError",
                exc_info={"returncode": completed.returncode},
            )
        payload = json.loads(match.group(1))
        exc_stack = payload.get("exc_stack")
        if exc_stack is not None:
            exc_stack = [tuple(item) for item in exc_stack]
        return ExecutionResult(
            term_out=[clean_stdout, completed.stderr],
            exec_time=float(payload.get("exec_time", 0.0)),
            exc_type=payload.get("exc_type"),
            exc_info=payload.get("exc_info"),
            exc_stack=exc_stack,
        )

    def sync_to_remote(self, remote_workspace: PurePosixPath) -> None:
        archive = create_archive(self.working_dir)
        remote_archive = f"/tmp/{archive.name}-{uuid.uuid4().hex}"
        remote_parent = str(remote_workspace.parent)
        scp_cmd = scp_base_args(self.runtime) + [
            str(archive),
            f"root@{self.runtime.ssh_host}:{remote_archive}",
        ]
        subprocess.run(scp_cmd, check=True, capture_output=True, text=True, timeout=self.timeout)
        command = (
            f"rm -rf {shlex.quote(str(remote_workspace))} && "
            f"mkdir -p {shlex.quote(remote_parent)} && "
            f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(remote_parent)} && "
            f"rm -f {shlex.quote(remote_archive)}"
        )
        result = run_ssh(self.runtime, command, self.timeout)
        shutil.rmtree(archive.parent, ignore_errors=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to sync workspace to Vast.ai:\n{result.stdout}\n{result.stderr}")

    def sync_from_remote(self, remote_workspace: PurePosixPath) -> None:
        remote_archive = f"/tmp/{self.working_dir.name}-{uuid.uuid4().hex}.tar.gz"
        archive_dir = Path(tempfile.mkdtemp(prefix="auctrace-vast-pull-"))
        local_archive = archive_dir / "workspace.tar.gz"
        pack_cmd = (
            f"cd {shlex.quote(str(remote_workspace.parent))} && "
            f"tar -czf {shlex.quote(remote_archive)} {shlex.quote(remote_workspace.name)}"
        )
        packed = run_ssh(self.runtime, pack_cmd, self.timeout)
        if packed.returncode != 0:
            raise RuntimeError(f"Failed to pack Vast.ai workspace:\n{packed.stdout}\n{packed.stderr}")
        scp_cmd = scp_base_args(self.runtime) + [
            f"root@{self.runtime.ssh_host}:{remote_archive}",
            str(local_archive),
        ]
        subprocess.run(scp_cmd, check=True, capture_output=True, text=True, timeout=self.timeout)
        run_ssh(self.runtime, f"rm -f {shlex.quote(remote_archive)}", self.timeout)
        shutil.rmtree(self.working_dir, ignore_errors=True)
        extract_archive(local_archive, self.working_dir.parent)
        shutil.rmtree(archive_dir, ignore_errors=True)
