from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from ai_scientist.project_env import load_project_env
from ai_scientist.treesearch.wsl_ssh_api import prepare_wsl_ssh_runtime
from ai_scientist.treesearch.wsl_ssh_execution import WslSshRemoteInterpreter

CONFIG_PATH = Path(__file__).resolve().parents[2] / "bfts_config.yaml"
VERIFY_RUN_NAME = "verify-wsl-ssh"
VERIFY_MARKER = "AUCTRACE_WSL_SSH_OK"


def _load_cfg():
    load_project_env()
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.exec.backend = "wsl_ssh"
    return cfg


def _build_code() -> str:
    return f"""
import json
import os
import torch
print("{VERIFY_MARKER}")
print(json.dumps({{"cwd": os.getcwd(), "files": sorted(os.listdir('.'))[:5], "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available()}}))
""".strip()


def _extract_remote_payload(text: str) -> dict[str, object]:
    for line in reversed(text.splitlines()):
        candidate = line.strip().replace("\x00", "")
        if not candidate.startswith("{"):
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"Remote verification payload missing:\n{text}")


def main() -> None:
    cfg = _load_cfg()
    runtime = prepare_wsl_ssh_runtime(cfg.exec, VERIFY_RUN_NAME)
    temp_root = Path(tempfile.mkdtemp(prefix="auctrace-wsl-verify-"))
    working_dir = temp_root / "workspace"
    working_dir.mkdir(parents=True, exist_ok=True)
    try:
        interpreter = WslSshRemoteInterpreter(
            working_dir=working_dir,
            timeout=300,
            agent_file_name="runfile.py",
            wsl_ssh_cfg=cfg.exec.wsl_ssh,
            run_name=VERIFY_RUN_NAME,
            gpu_id=0 if runtime.num_gpus > 0 else None,
            env_vars={},
        )
        result = interpreter.run(_build_code())
        text = "\n".join(result.term_out)
        if result.exc_type is not None:
            raise RuntimeError(f"Remote verification failed: {result.exc_type}\n{text}")
        if VERIFY_MARKER not in text:
            raise RuntimeError(f"Remote verification marker missing:\n{text}")
        remote_payload = _extract_remote_payload(text)
        summary = {
            "status": "ok",
            "ssh_host": runtime.ssh_host,
            "ssh_user": runtime.ssh_user,
            "remote_root": runtime.remote_root,
            "venv_path": runtime.venv_path,
            "windows_stage_dir": runtime.windows_stage_dir,
            "wsl_stage_dir": runtime.wsl_stage_dir,
            "num_gpus": runtime.num_gpus,
            "recommended_workers": runtime.recommended_workers,
            "torch_version": remote_payload.get("torch_version"),
            "cuda_available": remote_payload.get("cuda_available"),
        }
        print(json.dumps(summary, ensure_ascii=False))
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
