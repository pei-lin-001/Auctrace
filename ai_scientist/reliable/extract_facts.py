from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from typing import Any, Mapping

from ai_scientist.treesearch.utils.metric import MetricValue, WorstMetricValue

from .facts import FactRecord, FactStore
from .fact_extraction import (
    NodeMetricProvenance,
    build_ablation_fact_prefix,
    extract_facts_from_metric_payload,
)


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_best_node_from_stage_dir(stage_dir: str) -> dict[str, Any]:
    best_id_path = osp.join(stage_dir, "best_node_id.txt")
    journal_path = osp.join(stage_dir, "journal.json")
    best_id = ""
    if osp.exists(best_id_path):
        best_id = open(best_id_path, "r", encoding="utf-8").read().strip()
    if not best_id:
        raise ValueError(f"Missing best node id at: {best_id_path}")

    journal = _load_json(journal_path)
    nodes = journal.get("nodes") if isinstance(journal, Mapping) else None
    if not isinstance(nodes, list):
        raise ValueError(f"journal.json missing nodes list: {journal_path}")

    node = next((n for n in nodes if isinstance(n, Mapping) and n.get("id") == best_id), None)
    if node is None:
        raise ValueError(f"Best node id {best_id!r} not found in {journal_path}")
    return dict(node)


def _add_best_node_facts(
    *,
    store: FactStore,
    log_dir: str,
    stage_name: str,
    prefix: str,
) -> None:
    stage_dir = osp.join(log_dir, f"stage_{stage_name}")
    node = _read_best_node_from_stage_dir(stage_dir)
    prov = NodeMetricProvenance(
        stage_name=stage_name,
        node_id=str(node.get("id")) if node.get("id") else None,
        exp_results_dir=str(node.get("exp_results_dir")) if node.get("exp_results_dir") else None,
        source=f"{stage_dir}/journal.json",
    )
    for rec in extract_facts_from_metric_payload(
        prefix=prefix,
        metric_payload=node.get("metric") or {},
        provenance=prov,
    ):
        store.add(rec)


def _list_stage_dirs(log_dir: str, *, main_stage: int) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if not osp.isdir(log_dir):
        return out
    prefix = f"{main_stage}_"
    for entry in os.listdir(log_dir):
        if not entry.startswith("stage_"):
            continue
        stage_name = entry[len("stage_") :].strip()
        if not stage_name.startswith(prefix):
            continue
        stage_dir = osp.join(log_dir, entry)
        if not osp.isdir(stage_dir):
            continue
        if not osp.exists(osp.join(stage_dir, "journal.json")):
            continue
        out.append((stage_name, stage_dir))
    out.sort(key=lambda x: x[0])
    return out


def _metric_value_from_payload(payload: Mapping[str, Any]) -> MetricValue:
    value = payload.get("value")
    if value is None:
        return WorstMetricValue()
    return MetricValue(
        value=value,
        maximize=payload.get("maximize"),
        name=payload.get("name"),
        description=payload.get("description"),
    )


def _pick_best_per_ablation(nodes: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    best: dict[str, tuple[MetricValue, Mapping[str, Any]]] = {}
    for node in nodes:
        ablation_name = str(node.get("ablation_name") or "").strip()
        if not ablation_name:
            continue
        if bool(node.get("is_buggy")) or bool(node.get("is_buggy_plots")):
            continue
        metric_payload = node.get("metric") or {}
        if not isinstance(metric_payload, Mapping):
            continue
        mv = _metric_value_from_payload(metric_payload)
        if mv.value is None:
            continue
        prev = best.get(ablation_name)
        if prev is None or mv > prev[0]:
            best[ablation_name] = (mv, node)
    return {name: pair[1] for name, pair in best.items()}


def _add_ablation_facts(*, store: FactStore, log_dir: str) -> None:
    for stage_name, stage_dir in _list_stage_dirs(log_dir, main_stage=4):
        journal = _load_json(osp.join(stage_dir, "journal.json"))
        nodes = journal.get("nodes") if isinstance(journal, Mapping) else None
        if not isinstance(nodes, list):
            continue
        best_by_ablation = _pick_best_per_ablation(
            [n for n in nodes if isinstance(n, Mapping)]
        )
        for ablation_name, node in best_by_ablation.items():
            prefix = build_ablation_fact_prefix(ablation_name)
            prov = NodeMetricProvenance(
                stage_name=stage_name,
                node_id=str(node.get("id")) if node.get("id") else None,
                exp_results_dir=str(node.get("exp_results_dir")) if node.get("exp_results_dir") else None,
                source=f"{stage_dir}/journal.json",
            )
            for rec in extract_facts_from_metric_payload(
                prefix=prefix,
                metric_payload=node.get("metric") or {},
                provenance=prov,
            ):
                store.add(rec, allow_update=True)


def build_fact_store_for_run(base_folder: str) -> FactStore:
    """Build a FactStore for one experiments/<run>/ folder.

    This is intentionally minimal and deterministic:
    - It reads stage summary files to locate stage names (stage_2/stage_3).
    - It then reads best_node_id.txt + journal.json to fetch the best node.
    - It extracts structured metrics from node.metric.value.metric_names[*].data[*].
    """

    log_dir = osp.join(base_folder, "logs", "0-run")
    baseline_summary_path = osp.join(log_dir, "baseline_summary.json")
    research_summary_path = osp.join(log_dir, "research_summary.json")

    store = FactStore()

    if osp.exists(baseline_summary_path):
        baseline = _load_json(baseline_summary_path)
        stage_name = str(baseline.get("stage_name", "2_baseline")).strip()
        _add_best_node_facts(store=store, log_dir=log_dir, stage_name=stage_name, prefix="stage2.best")

    if osp.exists(research_summary_path):
        research = _load_json(research_summary_path)
        stage_name = str(research.get("stage_name", "3_research")).strip()
        _add_best_node_facts(store=store, log_dir=log_dir, stage_name=stage_name, prefix="stage3.best")

    _add_ablation_facts(store=store, log_dir=log_dir)

    return store


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract deterministic fact variables from an experiments/<run>/ folder.")
    parser.add_argument("--base-folder", required=True, help="experiments/<run>/ folder.")
    parser.add_argument("--out", default="", help="Output fact_store.json path. Defaults to <base>/logs/0-run/fact_store.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    store = build_fact_store_for_run(args.base_folder)
    out = args.out.strip() or osp.join(args.base_folder, "logs", "0-run", "fact_store.json")
    store.save_json(out)
    print(f"[facts] wrote {len(store.facts)} facts to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
