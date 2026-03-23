from __future__ import annotations

import os.path as osp
import re
from typing import Any, Iterable

from ai_scientist.treesearch.journal import Journal, Node

from .fact_extraction import (
    NodeMetricProvenance,
    build_ablation_fact_prefix,
    extract_facts_from_metric_payload,
)
from .facts import FactStore


_MAIN_STAGE_RE = re.compile(r"^(\d+)_")


def _parse_main_stage_number(stage_name: str) -> int:
    match = _MAIN_STAGE_RE.match(stage_name.strip())
    if not match:
        raise ValueError(f"Unexpected stage name format: {stage_name!r}")
    return int(match.group(1))


def _read_best_node_id(stage_dir: str) -> str:
    best_id_path = osp.join(stage_dir, "best_node_id.txt")
    if not osp.exists(best_id_path):
        return ""
    with open(best_id_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _find_node_by_id(journal: Journal, node_id: str) -> Node | None:
    if not node_id:
        return None
    for node in journal.nodes:
        if getattr(node, "id", None) == node_id:
            return node
    return None


def _metric_payload_from_node(node: Node) -> dict[str, Any]:
    metric = getattr(node, "metric", None)
    if metric is None:
        return {}
    return {
        "value": getattr(metric, "value", None),
        "maximize": getattr(metric, "maximize", None),
        "name": getattr(metric, "name", None),
        "description": getattr(metric, "description", None),
    }


def _delete_prefix(store: FactStore, prefix: str) -> None:
    drop = [k for k in store.facts.keys() if k.startswith(prefix)]
    for k in drop:
        del store.facts[k]


def _extract_best_node_facts(
    *,
    prefix: str,
    stage_name: str,
    stage_dir: str,
    journal: Journal,
) -> list:
    best_id = _read_best_node_id(stage_dir)
    best = _find_node_by_id(journal, best_id)
    if best is None:
        return []

    prov = NodeMetricProvenance(
        stage_name=stage_name,
        node_id=str(best.id),
        exp_results_dir=(
            str(best.exp_results_dir)
            if getattr(best, "exp_results_dir", None) is not None
            else None
        ),
        source=osp.join(stage_dir, "journal.json"),
    )
    metric_payload = _metric_payload_from_node(best)
    return extract_facts_from_metric_payload(
        prefix=prefix,
        metric_payload=metric_payload,
        provenance=prov,
    )


def _iter_ablation_nodes(journal: Journal) -> Iterable[Node]:
    for node in journal.nodes:
        ablation_name = getattr(node, "ablation_name", None)
        if not ablation_name:
            continue
        metric = getattr(node, "metric", None)
        if metric is None or getattr(metric, "value", None) is None:
            continue
        if bool(getattr(node, "is_buggy", False)) or bool(getattr(node, "is_buggy_plots", False)):
            continue
        yield node


def _pick_best_per_ablation(nodes: Iterable[Node]) -> dict[str, Node]:
    best: dict[str, Node] = {}
    for node in nodes:
        name = str(getattr(node, "ablation_name", "")).strip()
        if not name:
            continue
        prev = best.get(name)
        if prev is None or (getattr(node, "metric", None) > getattr(prev, "metric", None)):
            best[name] = node
    return best


def update_fact_store_for_stage(
    *,
    fact_store_path: str,
    stage_name: str,
    stage_dir: str,
    journal: Journal,
) -> int:
    """Update fact_store.json incrementally based on the latest stage artifacts.

    This is safe to call repeatedly (e.g., after each step_callback). It is deterministic
    and does not require extra LLM calls: it uses the stage's recorded best_node_id.txt
    and the in-memory journal nodes/metrics.
    """

    store = FactStore.load_json(fact_store_path) if osp.exists(fact_store_path) else FactStore()
    main_stage = _parse_main_stage_number(stage_name)

    if main_stage == 2:
        _delete_prefix(store, "stage2.best.")
        for rec in _extract_best_node_facts(
            prefix="stage2.best", stage_name=stage_name, stage_dir=stage_dir, journal=journal
        ):
            store.add(rec, allow_update=True)

    elif main_stage == 3:
        _delete_prefix(store, "stage3.best.")
        for rec in _extract_best_node_facts(
            prefix="stage3.best", stage_name=stage_name, stage_dir=stage_dir, journal=journal
        ):
            store.add(rec, allow_update=True)

    elif main_stage == 4:
        best_by_ablation = _pick_best_per_ablation(_iter_ablation_nodes(journal))
        for ablation_name, node in best_by_ablation.items():
            prefix = build_ablation_fact_prefix(ablation_name)
            metric_payload = _metric_payload_from_node(node)
            prov = NodeMetricProvenance(
                stage_name=stage_name,
                node_id=str(node.id),
                exp_results_dir=(
                    str(node.exp_results_dir)
                    if getattr(node, "exp_results_dir", None) is not None
                    else None
                ),
                source=osp.join(stage_dir, "journal.json"),
            )
            for rec in extract_facts_from_metric_payload(
                prefix=prefix,
                metric_payload=metric_payload,
                provenance=prov,
            ):
                store.add(rec, allow_update=True)

    store.save_json(fact_store_path)
    return len(store.facts)
