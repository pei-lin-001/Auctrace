from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ai_scientist.project_env import load_project_env

from .vast_common import VAST_API_ROOT, VastRuntime, normalize_vast_settings, set_runtime
from .vast_ssh import (
    probe_ssh_runtime,
    resolve_local_ssh_credentials,
    resolve_ssh_endpoint,
    run_setup_commands,
)

for vast_dir in (Path.home() / ".config" / "vastai", Path.home() / ".cache" / "vastai"):
    vast_dir.mkdir(parents=True, exist_ok=True)

try:
    from vastai_sdk import VastAI
except ImportError:  # pragma: no cover - dependency checked at runtime
    VastAI = None

INSTANCE_FAILURE_MARKERS = (
    "error pulling image",
    "failed to resolve reference",
    "tls:",
    "no host in request url",
)
load_project_env()


def build_vast_client(api_key: str) -> VastAI:
    if VastAI is None:
        raise RuntimeError(
            "Missing official Vast.ai SDK. Reinstall dependencies so `vastai-sdk` is available."
        )
    return VastAI(api_key=api_key, raw=True, quiet=True)


def api_json_request(
    method: str, path: str, api_key: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{VAST_API_ROOT}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Vast.ai API {method} {path} failed: {exc.code} {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Vast.ai API {method} {path} failed: {exc.reason}") from exc


def search_query(settings: dict[str, Any]) -> str:
    search = settings["search"]
    terms = [
        f"verified={'true' if search['verified'] else 'false'}",
        f"rentable={'true' if search['rentable'] else 'false'}",
        f"rented={'true' if search['rented'] else 'false'}",
        f"num_gpus>={search['num_gpus']}",
        f"reliability>={search['reliability_min']}",
        f"inet_up>={search['inet_up_min']}",
    ]
    if "direct" in settings["runtype"]:
        terms.append(f"direct_port_count>={search['direct_port_count_min']}")
    return " ".join(terms)


def coerce_offers(response: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(response, list):
        return response
    offers = response.get("offers", [])
    if isinstance(offers, list):
        return offers
    if isinstance(offers, dict):
        return [offers]
    raise RuntimeError(f"Unexpected Vast.ai offers payload: {type(offers)}")


def instance_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("instances"), dict):
        return payload["instances"]
    if isinstance(payload.get("instance"), dict):
        return payload["instance"]
    return payload


def show_instance(api_key: str, instance_id: int) -> dict[str, Any]:
    return instance_from_payload(api_json_request("GET", f"/instances/{instance_id}/", api_key))


def wait_for_instance(api_key: str, instance_id: int, settings: dict[str, Any]) -> dict[str, Any]:
    deadline = time.time() + settings["readiness_timeout"]
    sshd_grace = 15
    while time.time() < deadline:
        instance = show_instance(api_key, instance_id)
        status_msg = (instance.get("status_msg") or "").lower()
        if any(marker in status_msg for marker in INSTANCE_FAILURE_MARKERS):
            raise RuntimeError(
                f"Vast.ai instance {instance_id} failed during startup: {instance.get('status_msg')}"
            )
        if instance.get("actual_status") in {"offline", "exited"}:
            raise RuntimeError(f"Vast.ai instance {instance_id} became {instance.get('actual_status')}")
        ssh_ready = instance.get("ssh_host") and instance.get("ssh_port")
        if instance.get("cur_state") == "running" and instance.get("actual_status") == "running" and ssh_ready:
            time.sleep(sshd_grace)
            return instance
        time.sleep(settings["instance_poll_interval"])
    raise TimeoutError(f"Timed out waiting for Vast.ai instance {instance_id} to become ready")


def create_instance(api_key: str, offer_id: int, settings: dict[str, Any], exp_name: str) -> int:
    payload = {
        "client_id": "me",
        "image": settings["image"],
        "disk": settings["disk_gb"],
        "label": f"{settings['label_prefix']}-{exp_name}",
        "runtype": "ssh_direct" if "direct" in settings["runtype"] else "ssh",
        "cancel_unavail": True,
    }
    response = api_json_request("PUT", f"/asks/{offer_id}/", api_key, payload)
    new_contract = response.get("new_contract")
    if new_contract is None:
        raise RuntimeError(f"Vast.ai create-instance response missing new_contract: {response}")
    return int(new_contract)


def destroy_instance(api_key: str, instance_id: int) -> None:
    api_json_request("DELETE", f"/instances/{instance_id}/", api_key)


def offer_candidates(client: VastAI, settings: dict[str, Any]) -> list[dict[str, Any]]:
    if settings["offer_id"] is not None:
        return [{"offer_id": int(settings["offer_id"]), "machine_id": None}]
    query = search_query(settings)
    offers = coerce_offers(
        client.search_offers(
            type=settings["search"]["type"],
            no_default=True,
            limit=max(settings["search"]["limit"], settings["max_provision_attempts"]),
            order=settings["search"]["order"],
            query=query,
        )
    )
    candidates = []
    for offer in offers:
        candidates.append(
            {
                "offer_id": int(offer.get("ask_contract_id") or offer.get("id")),
                "machine_id": offer.get("machine_id"),
            }
        )
    if not candidates:
        raise RuntimeError(
            "No Vast.ai offers matched the configured search filters.\n"
            f"query={query}\n"
            f"search={settings.get('search')}\n"
            "If this is unexpected, check network/proxy settings. "
            "The launcher can disable proxies via --disable-http-proxy."
        )
    return candidates


def _attach_ssh_with_retry(client: VastAI, instance_id: int, public_key_text: str, max_attempts: int = 3) -> None:
    for attempt in range(1, max_attempts + 1):
        client.attach_ssh(instance_id=instance_id, ssh_key=public_key_text)
        if attempt < max_attempts:
            time.sleep(5)
    time.sleep(10)


def _format_failure(
    offer_id: int,
    machine_id: int | None,
    instance_id: int | None,
    exc: Exception,
) -> str:
    location = f"offer={offer_id}"
    if machine_id is not None:
        location += f", machine={machine_id}"
    if instance_id is not None:
        location += f", instance={instance_id}"
    return f"{location}: {exc}"


def _runtime_from_instance(
    client: VastAI,
    instance: dict[str, Any],
    instance_id: int,
    offer_id: int | None,
    private_key_path: str | None,
    settings: dict[str, Any],
) -> VastRuntime:
    ssh_host, ssh_port = resolve_ssh_endpoint(client, instance_id, instance)
    return VastRuntime(
        instance_id=instance_id,
        ssh_host=ssh_host,
        ssh_port=ssh_port,
        num_gpus=int(instance.get("num_gpus", settings["search"]["num_gpus"])),
        remote_root=settings["remote_root"],
        private_key_path=private_key_path,
        offer_id=offer_id,
        machine_id=instance.get("machine_id"),
    )


def _provision_existing_instance(
    api_key: str,
    client: VastAI,
    settings: dict[str, Any],
    public_key_text: str,
    private_key_path: str | None,
    exp_name: str,
) -> VastRuntime:
    instance_id = int(settings["existing_instance_id"])
    api_json_request(
        "PUT",
        f"/instances/{instance_id}/",
        api_key,
        {"state": "running", "label": f"{settings['label_prefix']}-{exp_name}"},
    )
    client.attach_ssh(instance_id=instance_id, ssh_key=public_key_text)
    instance = wait_for_instance(api_key, instance_id, settings)
    runtime = _runtime_from_instance(
        client, instance, instance_id, settings["offer_id"], private_key_path, settings
    )
    _attach_ssh_with_retry(client, instance_id, public_key_text)
    probe_ssh_runtime(runtime, settings)
    run_setup_commands(runtime, settings)
    return runtime


def prepare_vast_runtime(exec_cfg: Any, exp_name: str) -> VastRuntime:
    settings = normalize_vast_settings(exec_cfg)
    if settings["runtime"]:
        return VastRuntime(**dict(settings["runtime"]))
    api_key = os.environ.get(settings["api_key_env"])
    if not api_key:
        raise RuntimeError(f"Missing Vast.ai API key in ${settings['api_key_env']}")
    private_key_path, public_key_text = resolve_local_ssh_credentials(settings)
    client = build_vast_client(api_key)
    if settings["existing_instance_id"] is not None:
        runtime = _provision_existing_instance(
            api_key, client, settings, public_key_text, private_key_path, exp_name
        )
        set_runtime(exec_cfg, runtime)
        return runtime

    failures: list[str] = []
    bad_machines: set[int] = set()
    for candidate in offer_candidates(client, settings):
        offer_id = candidate["offer_id"]
        machine_id = candidate["machine_id"]
        if machine_id in bad_machines:
            continue
        instance_id = None
        try:
            instance_id = create_instance(api_key, offer_id, settings, exp_name)
            client.attach_ssh(instance_id=instance_id, ssh_key=public_key_text)
            instance = wait_for_instance(api_key, instance_id, settings)
            runtime = _runtime_from_instance(
                client, instance, instance_id, offer_id, private_key_path, settings
            )
            _attach_ssh_with_retry(client, instance_id, public_key_text)
            probe_ssh_runtime(runtime, settings)
            run_setup_commands(runtime, settings)
            set_runtime(exec_cfg, runtime)
            return runtime
        except Exception as exc:
            current_machine = machine_id
            if instance_id is not None:
                try:
                    current_machine = show_instance(api_key, instance_id).get("machine_id", machine_id)
                except Exception:
                    pass
            if current_machine is not None:
                bad_machines.add(int(current_machine))
            failures.append(_format_failure(offer_id, current_machine, instance_id, exc))
            if instance_id is not None:
                try:
                    destroy_instance(api_key, instance_id)
                except Exception as destroy_exc:
                    failures.append(
                        _format_failure(offer_id, current_machine, instance_id, destroy_exc)
                    )
            if len(failures) >= settings["max_provision_attempts"]:
                break
    failure_text = "\n".join(f"- {item}" for item in failures)
    raise RuntimeError(
        "Failed to provision a healthy Vast.ai instance after trying the lowest-cost usable offers:\n"
        f"{failure_text}"
    )


def cleanup_vast_runtime(exec_cfg: Any) -> None:
    settings = normalize_vast_settings(exec_cfg)
    runtime = settings["runtime"]
    if not runtime or not settings["auto_destroy"]:
        return
    api_key = os.environ.get(settings["api_key_env"])
    if not api_key:
        return
    destroy_instance(api_key, int(runtime["instance_id"]))
