from __future__ import annotations

import json
import os
import os.path as osp
import re
from typing import Any, Mapping, Sequence

from .fact_extraction import normalize_key_part
from .facts import FactStore

ARTIFACT_MANIFEST_SCHEMA = "auctrace.artifact_manifest.v1"
_TABLE_LABEL_RE = re.compile(r"\\label\{(tab:[^}]+)\}")


def _normalize_artifact_path(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _relpath(base_folder: str, path: str) -> str:
    try:
        return osp.relpath(path, base_folder)
    except ValueError:
        return path


def _description_from_name(name: str) -> str:
    stem = osp.splitext(name)[0]
    return stem.replace("_", " ").replace("-", " ").strip()


def _figure_entries(base_folder: str) -> list[dict[str, Any]]:
    figures_dir = osp.join(base_folder, "figures")
    if not osp.isdir(figures_dir):
        return []
    out: list[dict[str, Any]] = []
    for name in sorted(os.listdir(figures_dir)):
        path = osp.join(figures_dir, name)
        if not osp.isfile(path):
            continue
        out.append(
            {
                "artifact_id": f"figure.{normalize_key_part(osp.splitext(name)[0])}",
                "artifact_type": "figure",
                "path": _relpath(base_folder, path),
                "description": _description_from_name(name),
                "source_stage": None,
                "source_node_id": None,
                "source_exp_results_dir": None,
                "derived_from_facts": [],
            }
        )
    return out


def _raw_result_entries(base_folder: str, store: FactStore | None) -> list[dict[str, Any]]:
    if store is None:
        return []
    by_dir: dict[str, list[str]] = {}
    by_stage: dict[str, str | None] = {}
    by_node: dict[str, str | None] = {}
    for key, rec in store.facts.items():
        prov = rec.provenance or {}
        exp_dir = str(prov.get("exp_results_dir") or "").strip()
        if not exp_dir:
            continue
        by_dir.setdefault(exp_dir, []).append(key)
        by_stage.setdefault(exp_dir, prov.get("stage_name"))
        by_node.setdefault(exp_dir, prov.get("node_id"))

    out: list[dict[str, Any]] = []
    for exp_dir in sorted(by_dir):
        out.append(
            {
                "artifact_id": f"raw_result.{normalize_key_part(exp_dir)}",
                "artifact_type": "raw_result",
                "path": _relpath(base_folder, exp_dir),
                "description": f"Raw experiment results for {by_stage.get(exp_dir) or 'unknown stage'}",
                "source_stage": by_stage.get(exp_dir),
                "source_node_id": by_node.get(exp_dir),
                "source_exp_results_dir": _relpath(base_folder, exp_dir),
                "derived_from_facts": sorted(by_dir[exp_dir]),
            }
        )
    return out


def _table_entries(base_folder: str, tex_path: str | None) -> list[dict[str, Any]]:
    if not tex_path or not osp.exists(tex_path):
        return []
    with open(tex_path, "r", encoding="utf-8") as f:
        tex = f.read()
    labels = sorted(set(_TABLE_LABEL_RE.findall(tex)))
    out: list[dict[str, Any]] = []
    for label in labels:
        out.append(
            {
                "artifact_id": f"table.{normalize_key_part(label)}",
                "artifact_type": "table",
                "path": f"{_relpath(base_folder, tex_path)}#{label}",
                "description": label,
                "source_stage": None,
                "source_node_id": None,
                "source_exp_results_dir": None,
                "derived_from_facts": [],
            }
        )
    return out


def _dedupe_entries(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for entry in entries:
        artifact_id = str(entry.get("artifact_id") or "").strip()
        if artifact_id:
            deduped[artifact_id] = dict(entry)
    return [deduped[key] for key in sorted(deduped)]


def build_artifact_manifest(
    *,
    base_folder: str,
    store: FactStore | None = None,
    tex_path: str | None = None,
) -> dict[str, Any]:
    entries = []
    entries.extend(_figure_entries(base_folder))
    entries.extend(_raw_result_entries(base_folder, store))
    entries.extend(_table_entries(base_folder, tex_path))
    return {"schema": ARTIFACT_MANIFEST_SCHEMA, "artifacts": _dedupe_entries(entries)}


def save_artifact_manifest(path: str, payload: Mapping[str, Any]) -> None:
    parent = osp.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, ensure_ascii=True)


def ensure_artifact_manifest(
    *,
    base_folder: str,
    store: FactStore | None = None,
    tex_path: str | None = None,
) -> dict[str, Any]:
    payload = build_artifact_manifest(base_folder=base_folder, store=store, tex_path=tex_path)
    save_artifact_manifest(osp.join(base_folder, "logs", "0-run", "artifact_manifest.json"), payload)
    return payload


def artifact_manifest_summary(
    payload: Mapping[str, Any],
    *,
    max_entries: int = 32,
) -> dict[str, Any]:
    artifacts = payload.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, list) else []
    trimmed: list[dict[str, Any]] = []
    for item in artifacts[:max_entries]:
        if not isinstance(item, Mapping):
            continue
        trimmed.append(
            {
                "artifact_id": item.get("artifact_id"),
                "artifact_type": item.get("artifact_type"),
                "path": item.get("path"),
                "description": item.get("description"),
                "source_stage": item.get("source_stage"),
                "derived_from_facts": item.get("derived_from_facts"),
            }
        )
    return {
        "schema": ARTIFACT_MANIFEST_SCHEMA,
        "artifact_count": len(artifacts),
        "artifacts": trimmed,
        "includegraphics_targets": figure_includegraphics_targets(payload),
    }


def artifact_id_set(payload: Mapping[str, Any]) -> set[str]:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        return set()
    out: set[str] = set()
    for item in artifacts:
        if not isinstance(item, Mapping):
            continue
        artifact_id = str(item.get("artifact_id") or "").strip()
        if artifact_id:
            out.add(artifact_id)
    return out


def figure_includegraphics_targets(payload: Mapping[str, Any]) -> list[str]:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        return []
    allowed: set[str] = set()
    for item in artifacts:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("artifact_type") or "").strip() != "figure":
            continue
        path = _normalize_artifact_path(str(item.get("path") or ""))
        if not path:
            continue
        basename = osp.basename(path)
        stem, _ = osp.splitext(basename)
        path_stem, _ = osp.splitext(path)
        allowed.update(
            {
                path,
                basename,
                stem,
                path_stem,
                f"../{path}",
                f"../{path_stem}",
            }
        )
    return sorted(allowed)
