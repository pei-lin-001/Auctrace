from __future__ import annotations

import json
import os.path as osp
from typing import Any

from .extract_facts import build_fact_store_for_run
from .facts import FactStore
from .writeup_summary import (
    build_writeup_summary_symbolic,
    save_symbolic_summary_artifacts,
    validate_writeup_summary_symbolic,
)


def _load_json_or_default(path: str, default: Any) -> Any:
    if not osp.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_raw_summary_inputs(log_dir: str) -> dict[str, Any]:
    return {
        "draft_summary": _load_json_or_default(osp.join(log_dir, "draft_summary.json"), {}),
        "baseline_summary": _load_json_or_default(osp.join(log_dir, "baseline_summary.json"), {}),
        "research_summary": _load_json_or_default(osp.join(log_dir, "research_summary.json"), {}),
        "ablation_summary": _load_json_or_default(osp.join(log_dir, "ablation_summary.json"), []),
    }


def ensure_symbolic_writeup_inputs(base_folder: str) -> tuple[FactStore, dict[str, Any]]:
    log_dir = osp.join(base_folder, "logs", "0-run")
    fact_store_path = osp.join(log_dir, "fact_store.json")
    symbolic_summary_path = osp.join(log_dir, "writeup_summary_symbolic.json")

    store = (
        FactStore.load_json(fact_store_path)
        if osp.exists(fact_store_path)
        else build_fact_store_for_run(base_folder)
    )
    if not osp.exists(fact_store_path):
        store.save_json(fact_store_path)

    if osp.exists(symbolic_summary_path):
        with open(symbolic_summary_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        validate_writeup_summary_symbolic(payload, store)
        return store, payload

    raw_inputs = _load_raw_summary_inputs(log_dir)
    payload = build_writeup_summary_symbolic(store, **raw_inputs)
    validate_writeup_summary_symbolic(payload, store)
    save_symbolic_summary_artifacts(log_dir, payload)
    return store, payload
