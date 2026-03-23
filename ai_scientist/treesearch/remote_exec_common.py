"""Shared utilities for remote execution backends."""
from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any


def cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    """Get a value from a config object (dict or object with attributes)."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# Result markers for remote execution
RESULT_START = "__AUCTRACE_REMOTE_RESULT_START__"
RESULT_END = "__AUCTRACE_REMOTE_RESULT_END__"
RUNNER_FILE = "__auctrace_remote_exec.py"


def create_archive(local_workspace: Path) -> Path:
    """Create a tar.gz archive of the local workspace."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="auctrace-archive-"))
    archive_path = tmp_dir / f"{local_workspace.name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(local_workspace, arcname=local_workspace.name)
    return archive_path


def extract_archive(archive_path: Path, target_parent: Path) -> None:
    """Extract a tar.gz archive to the target parent directory."""
    target_parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(target_parent)


def write_runner(local_workspace: Path, agent_file_name: str, timeout: int) -> None:
    """Write the remote execution runner script."""
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


def cleanup_archive(archive_path: Path) -> None:
    """Remove the archive and its temp directory."""
    if archive_path.exists():
        parent = archive_path.parent
        shutil.rmtree(parent, ignore_errors=True)
