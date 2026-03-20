from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

from .wsl_ssh_client import (
    WslSshAuth,
    remote_windows_path,
    run_command,
    run_wsl_command,
    scp_base_args,
)
from .wsl_ssh_probe import load_json_array
from .wsl_ssh_requirements import required_packages_from_file


def setup_runtime(auth: WslSshAuth, settings: dict[str, Any]) -> None:
    requirements_path = (Path(__file__).resolve().parents[2] / settings["requirements_file"]).resolve()
    if settings["install_project_requirements"] and not requirements_path.exists():
        raise FileNotFoundError(f"Configured requirements file does not exist: {requirements_path}")
    script = "\n".join(
        [
            "set -euo pipefail",
            f"mkdir -p {settings['remote_root']} {settings['runs_root']}",
            f"if [ ! -d {settings['venv_path']} ]; then python3 -m venv {settings['venv_path']}; fi",
            f"source {settings['venv_path']}/bin/activate",
            "python -m pip install --disable-pip-version-check --no-input --progress-bar off -U pip",
        ]
    )
    result = run_wsl_command(
        auth,
        script,
        settings["setup_timeout"],
        settings["connect_timeout"],
        settings["wsl_distro"],
    )
    if result.returncode != 0:
        raise RuntimeError(f"WSL runtime setup failed:\n{result.stdout}\n{result.stderr}")


def _missing_packages_script(settings: dict[str, Any], packages: list[str]) -> str:
    package_list = ", ".join(repr(package) for package in packages)
    return "\n".join(
        [
            "set -euo pipefail",
            f"source {settings['venv_path']}/bin/activate",
            "python - <<'PY'",
            "import json",
            "import subprocess",
            f"packages = [{package_list}]",
            "missing = []",
            "for package in packages:",
            "    result = subprocess.run(['python', '-m', 'pip', 'show', package], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)",
            "    if result.returncode != 0:",
            "        missing.append(package)",
            "print(json.dumps(missing))",
            "PY",
        ]
    )


def missing_packages(
    auth: WslSshAuth,
    settings: dict[str, Any],
    packages: list[str],
) -> list[str]:
    if not packages:
        return []
    result = run_wsl_command(
        auth,
        _missing_packages_script(settings, packages),
        settings["pip_install_timeout"],
        settings["connect_timeout"],
        settings["wsl_distro"],
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to probe installed WSL packages:\n{result.stdout}\n{result.stderr}")
    return load_json_array(result.stdout, "missing-packages")


def _install_requirements_file(
    auth: WslSshAuth,
    settings: dict[str, Any],
    remote_requirements: str,
    marker: str,
) -> None:
    script = "\n".join(
        [
            "set -euo pipefail",
            f"source {settings['venv_path']}/bin/activate",
            (
                "python -m pip install --disable-pip-version-check --no-input --progress-bar off "
                f"-r {remote_requirements}"
            ),
            f"touch {marker}",
        ]
    )
    result = run_wsl_command(
        auth,
        script,
        settings["pip_install_timeout"],
        settings["connect_timeout"],
        settings["wsl_distro"],
    )
    if result.returncode != 0:
        raise RuntimeError(
            "WSL project dependency installation failed:\n"
            f"{result.stdout}\n{result.stderr}"
        )


def _requirements_paths(settings: dict[str, Any]) -> tuple[Path, str, str, str]:
    requirements_path = (Path(__file__).resolve().parents[2] / settings["requirements_file"]).resolve()
    requirements_hash = sha256(requirements_path.read_bytes()).hexdigest()[:12]
    remote_name = f"requirements-{requirements_hash}.txt"
    bootstrap_root = f"{settings['remote_root']}/.auctrace_bootstrap"
    return requirements_path, remote_name, bootstrap_root, f"{bootstrap_root}/{remote_name}"


def _upload_requirements(
    auth: WslSshAuth,
    settings: dict[str, Any],
    requirements_path: Path,
    windows_stage_dir_name: str,
    remote_name: str,
) -> None:
    remote_windows = remote_windows_path(windows_stage_dir_name, remote_name)
    scp_cmd = scp_base_args(auth, settings["connect_timeout"]) + [
        str(requirements_path),
        f"{auth.user}@{auth.host}:{remote_windows}",
    ]
    pushed = run_command(scp_cmd, settings["pip_install_timeout"], auth.password)
    if pushed.returncode != 0:
        raise RuntimeError(f"Failed to upload WSL requirements:\n{pushed.stdout}\n{pushed.stderr}")


def _prepare_bootstrap_file(
    auth: WslSshAuth,
    settings: dict[str, Any],
    bootstrap_root: str,
    remote_name: str,
    remote_requirements: str,
    wsl_stage_dir: str,
) -> None:
    prep_script = "\n".join(
        [
            "set -euo pipefail",
            f"mkdir -p {bootstrap_root}",
            f"cp {wsl_stage_dir}/{remote_name} {remote_requirements}",
            *(
                [f"rm -f {wsl_stage_dir}/{remote_name}"]
                if settings["cleanup_stage_archive"]
                else []
            ),
        ]
    )
    prepared = run_wsl_command(
        auth,
        prep_script,
        settings["pip_install_timeout"],
        settings["connect_timeout"],
        settings["wsl_distro"],
    )
    if prepared.returncode != 0:
        raise RuntimeError(f"Failed to prepare WSL requirements bootstrap:\n{prepared.stdout}\n{prepared.stderr}")


def _marker_exists(
    auth: WslSshAuth,
    settings: dict[str, Any],
    marker: str,
) -> bool:
    result = run_wsl_command(
        auth,
        f"test -f {marker}",
        settings["connect_timeout"],
        settings["connect_timeout"],
        settings["wsl_distro"],
    )
    return result.returncode == 0


def _refresh_requirements(
    auth: WslSshAuth,
    settings: dict[str, Any],
    marker: str,
    remote_requirements: str,
) -> None:
    run_wsl_command(
        auth,
        f"rm -f {marker}",
        settings["connect_timeout"],
        settings["connect_timeout"],
        settings["wsl_distro"],
    )
    _install_requirements_file(auth, settings, remote_requirements, marker)


def install_requirements(
    auth: WslSshAuth,
    settings: dict[str, Any],
    windows_stage_dir_name: str,
    wsl_stage_dir: str,
) -> None:
    if not settings["install_project_requirements"]:
        return
    requirements_path, remote_name, bootstrap_root, remote_requirements = _requirements_paths(settings)
    marker = f"{bootstrap_root}/{remote_name}.done"
    packages = required_packages_from_file(requirements_path)
    _upload_requirements(auth, settings, requirements_path, windows_stage_dir_name, remote_name)
    _prepare_bootstrap_file(
        auth,
        settings,
        bootstrap_root,
        remote_name,
        remote_requirements,
        wsl_stage_dir,
    )
    marker_present = _marker_exists(auth, settings, marker)
    current_missing = missing_packages(auth, settings, packages)
    if (not marker_present) or current_missing:
        _refresh_requirements(auth, settings, marker, remote_requirements)
        current_missing = missing_packages(auth, settings, packages)
    if current_missing:
        raise RuntimeError(
            "WSL project dependency installation left required packages missing: "
            + ", ".join(current_missing)
        )
