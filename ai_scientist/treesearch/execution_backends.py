from __future__ import annotations

from pathlib import Path
from typing import Any

from .interpreter import Interpreter
from .wsl_ssh_execution import WslSshRemoteInterpreter


def create_interpreter(
    cfg: Any,
    working_dir: Path | str,
    gpu_id: int | None = None,
    env_vars: dict[str, str] | None = None,
):
    backend = getattr(cfg.exec, "backend", "local")
    common_args = {
        "working_dir": working_dir,
        "timeout": cfg.exec.timeout,
        "agent_file_name": cfg.exec.agent_file_name,
        "env_vars": env_vars or {},
    }
    if backend == "wsl_ssh":
        return WslSshRemoteInterpreter(
            wsl_ssh_cfg=cfg.exec.wsl_ssh,
            run_name=cfg.exp_name,
            gpu_id=gpu_id,
            **common_args,
        )
    return Interpreter(
        format_tb_ipython=cfg.exec.format_tb_ipython,
        **common_args,
    )
