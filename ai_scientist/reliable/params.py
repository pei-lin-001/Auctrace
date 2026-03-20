from __future__ import annotations

import ast
import json
import os
import os.path as osp
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, MutableMapping

from .errors import InvalidParamKeyError, ParamStoreFormatError, UnknownParamKeyError

_PARAM_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")
_STAGE_NUMBER_RE = re.compile(r"stage_(\d+)_")
_CONFIG_VALUE_TYPES = (int, float)
_KNOWN_DATASET_CLASS_COUNTS = {
    "ag_news": 4,
    "imdb": 2,
    "yelp_polarity": 2,
    "synthetic": 2,
}
_SHARED_PARAM_RECORDS = (
    ("classification.random_baseline.binary", 0.5, "float:2", "Random-guess baseline for binary classification."),
    ("classification.random_baseline.four_class", 0.25, "float:2", "Random-guess baseline for four-class classification."),
)


def validate_param_key(key: str) -> None:
    if not isinstance(key, str) or not key:
        raise InvalidParamKeyError(f"Param key must be a non-empty string, got: {key!r}")
    if not _PARAM_KEY_PATTERN.match(key):
        raise InvalidParamKeyError(
            "Invalid param key (allowed: [A-Za-z0-9_.:-], max 200 chars, "
            f"must start with alnum): {key!r}"
        )


@dataclass(frozen=True)
class ParamRecord:
    key: str
    meaning: str
    value: Any
    format: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "meaning": self.meaning,
            "value": self.value,
            "format": self.format,
            "provenance": dict(self.provenance),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "ParamRecord":
        if "key" not in data or "meaning" not in data or "value" not in data:
            raise ParamStoreFormatError(
                "ParamRecord requires keys: 'key', 'meaning', 'value'. "
                f"Got keys: {sorted(data.keys())}"
            )
        key = str(data["key"])
        validate_param_key(key)
        fmt = data.get("format")
        provenance = data.get("provenance") or {}
        if not isinstance(provenance, Mapping):
            raise ParamStoreFormatError(
                f"ParamRecord.provenance must be a mapping, got: {type(provenance)}"
            )
        return ParamRecord(
            key=key,
            meaning=str(data["meaning"]),
            value=data["value"],
            format=None if fmt is None else str(fmt),
            provenance=dict(provenance),
        )


@dataclass
class ParamStore:
    params: MutableMapping[str, ParamRecord] = field(default_factory=dict)

    def add(self, record: ParamRecord, *, allow_update: bool = True) -> None:
        validate_param_key(record.key)
        if record.key in self.params and not allow_update:
            raise ParamStoreFormatError(f"Duplicate param key not allowed: {record.key!r}")
        self.params[record.key] = record

    def get(self, key: str) -> ParamRecord:
        validate_param_key(key)
        record = self.params.get(key)
        if record is None:
            raise UnknownParamKeyError(f"Unknown param key: {key!r}")
        return record

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "auctrace.param_store.v1",
            "params": [rec.to_dict() for rec in self.params.values()],
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "ParamStore":
        items = data.get("params")
        if not isinstance(items, list):
            raise ParamStoreFormatError(
                "ParamStore JSON must have 'params' as a list. "
                f"Got type: {type(items)}"
            )
        store = ParamStore()
        for item in items:
            if not isinstance(item, Mapping):
                raise ParamStoreFormatError(
                    f"ParamStore params must be list[dict], got: {type(item)}"
                )
            store.add(ParamRecord.from_dict(item))
        return store

    def save_json(self, path: str) -> None:
        parent = osp.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=True)

    @staticmethod
    def load_json(path: str) -> "ParamStore":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, Mapping):
            raise ParamStoreFormatError(
                f"ParamStore JSON must be an object, got: {type(data)}"
            )
        return ParamStore.from_dict(data)


def format_param_value_for_latex(record: ParamRecord) -> str:
    rendered = str(record.value) if record.format is None else _apply_simple_format(
        record.value,
        record.format,
    )
    return _escape_latex_text(rendered)


def params_index_for_prompt(store: ParamStore) -> str:
    lines: list[str] = []
    for key in sorted(store.params):
        rec = store.params[key]
        lines.append(f"- {key}: {rec.meaning.strip().replace(chr(10), ' ')}")
    return "\n".join(lines)


def ensure_param_store_for_run(base_folder: str) -> ParamStore:
    store_path = osp.join(base_folder, "logs", "0-run", "param_store.json")
    if osp.exists(store_path):
        return ParamStore.load_json(store_path)
    store = build_param_store_for_run(base_folder)
    store.save_json(store_path)
    return store


def build_param_store_for_run(base_folder: str) -> ParamStore:
    store = ParamStore()
    _add_shared_params(store)
    best_solution_paths = sorted(
        _iter_best_solution_paths(base_folder),
        key=_stage_sort_key,
    )
    latest_prefix = ""
    latest_config: dict[str, ParamRecord] = {}
    for path in best_solution_paths:
        prefix = _stage_prefix(path)
        if not prefix:
            continue
        source = _read_text(path)
        tree = ast.parse(source, filename=path)
        latest_prefix = prefix
        latest_config = _add_config_params(store, tree, prefix, path)
        _add_dataset_params(store, tree, source, prefix, path)
    if latest_prefix and latest_config:
        for name, record in latest_config.items():
            store.add(
                ParamRecord(
                    key=f"current_best.config.{name}",
                    meaning=record.meaning.replace(latest_prefix, "current_best"),
                    value=record.value,
                    format=record.format,
                    provenance=dict(record.provenance),
                )
            )
    return store


def _iter_best_solution_paths(base_folder: str) -> Iterable[str]:
    root = osp.join(base_folder, "logs", "0-run")
    for name in os.listdir(root):
        if not name.startswith("stage_"):
            continue
        stage_dir = osp.join(root, name)
        if not osp.isdir(stage_dir):
            continue
        for item in os.listdir(stage_dir):
            if item.startswith("best_solution_") and item.endswith(".py"):
                yield osp.join(stage_dir, item)


def _stage_sort_key(path: str) -> tuple[int, str]:
    stage_name = osp.basename(osp.dirname(path))
    match = _STAGE_NUMBER_RE.search(stage_name)
    stage_num = int(match.group(1)) if match else 999
    return stage_num, stage_name


def _stage_prefix(path: str) -> str:
    stage_name = osp.basename(osp.dirname(path))
    match = _STAGE_NUMBER_RE.search(stage_name)
    if match is None:
        return ""
    return f"stage{int(match.group(1))}.best"


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _add_shared_params(store: ParamStore) -> None:
    for key, value, fmt, meaning in _SHARED_PARAM_RECORDS:
        store.add(
            ParamRecord(
                key=key,
                meaning=meaning,
                value=value,
                format=fmt,
                provenance={"source": "deterministic_constant"},
            )
        )


def _add_config_params(
    store: ParamStore,
    tree: ast.AST,
    prefix: str,
    source_path: str,
) -> dict[str, ParamRecord]:
    records: dict[str, ParamRecord] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "Config":
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                continue
            value = _literal_numeric_value(stmt.value)
            if value is None:
                continue
            record = ParamRecord(
                key=f"{prefix}.config.{target.id}",
                meaning=f"{prefix} config value for '{target.id}'",
                value=value,
                provenance={"source_file": source_path, "kind": "config"},
            )
            store.add(record)
            records[target.id] = record
        break
    return records


def _literal_numeric_value(node: ast.AST) -> int | float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, _CONFIG_VALUE_TYPES):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _literal_numeric_value(node.operand)
        return -inner if inner is not None else None
    return None


def _add_dataset_params(
    store: ParamStore,
    tree: ast.AST,
    source: str,
    prefix: str,
    source_path: str,
) -> None:
    synthetic_classes = _extract_make_classification_classes(tree)
    if synthetic_classes:
        _add_baseline_params(store, prefix, "synthetic", synthetic_classes, source_path)
    for dataset_name, class_count in _KNOWN_DATASET_CLASS_COUNTS.items():
        if dataset_name == "synthetic":
            continue
        if dataset_name in source:
            _add_baseline_params(store, prefix, dataset_name, class_count, source_path)


def _extract_make_classification_classes(tree: ast.AST) -> int | None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Name) or func.id != "make_classification":
            continue
        for keyword in node.keywords:
            if keyword.arg != "n_classes":
                continue
            value = _literal_numeric_value(keyword.value)
            if isinstance(value, int) and value > 0:
                return value
    return None


def _add_baseline_params(
    store: ParamStore,
    prefix: str,
    dataset_name: str,
    class_count: int,
    source_path: str,
) -> None:
    baseline = 1.0 / float(class_count)
    provenance = {
        "source_file": source_path,
        "dataset": dataset_name,
        "class_count": class_count,
        "kind": "derived_random_baseline",
    }
    store.add(
        ParamRecord(
            key=f"{prefix}.dataset.{dataset_name}.class_count",
            meaning=f"{prefix} class count for dataset '{dataset_name}'",
            value=class_count,
            provenance=provenance,
        )
    )
    store.add(
        ParamRecord(
            key=f"{prefix}.dataset.{dataset_name}.random_baseline",
            meaning=f"{prefix} random-guess baseline for dataset '{dataset_name}'",
            value=baseline,
            format="float:2",
            provenance=provenance,
        )
    )


def _apply_simple_format(value: Any, fmt: str) -> str:
    if fmt.startswith("float:"):
        decimals = int(fmt.split(":", 1)[1])
        return f"{float(value):.{decimals}f}"
    if fmt == "int":
        return str(int(value))
    if fmt.startswith("percent:"):
        decimals = int(fmt.split(":", 1)[1])
        pct = 100.0 * float(value)
        return f"{pct:.{decimals}f}%"
    raise ParamStoreFormatError(f"Unsupported param format: {fmt!r}")


def _escape_latex_text(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("_", "\\_")
    )
