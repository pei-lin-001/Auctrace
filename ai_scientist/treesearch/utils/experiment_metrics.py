from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

GENERIC_PATH_PARTS = {"experiment_results", "metrics", "losses"}
SKIP_SCALAR_KEYS = {
    "timestamps",
    "predictions",
    "ground_truth",
    "labels",
    "hyperparams",
    "hyperparams_tested",
    "best_hyperparam",
    "best_val_loss",
    "budget_consumed",
    "total_budget",
    "informative_budget",
    "uninformative_budget",
    "retries",
    "sci_stops",
}
POSITIVE_HINTS = {
    "accuracy",
    "f1",
    "precision",
    "recall",
    "auc",
    "score",
    "rate",
    "efficiency",
    "return",
    "yield",
    "success",
}
NEGATIVE_HINTS = {
    "loss",
    "error",
    "waste",
    "ratio",
    "mse",
    "rmse",
    "mae",
    "perplexity",
}
SPECIAL_METRIC_NAMES = {
    "err": "effective research return (ERR)",
    "ery": "effective research yield (ERY)",
    "avg_metric": "average metric",
    "success_rate": "success rate",
    "waste_ratio": "waste ratio",
    "budget_waste_ratio": "budget waste ratio",
    "research_efficiency": "research efficiency",
}


def extract_metrics_payload(data_path: Path) -> tuple[list[dict[str, Any]], str]:
    data = np.load(data_path, allow_pickle=True).item()
    if not isinstance(data, dict):
        raise ValueError("experiment_data.npy does not contain a dictionary payload")

    collector: dict[str, dict[str, Any]] = {}
    _walk_mapping(data, (), collector)
    metric_names = sorted(collector.values(), key=_metric_sort_key)
    if not metric_names:
        raise ValueError("No deterministic metrics could be extracted from experiment_data.npy")
    for metric in metric_names:
        metric["data"].sort(key=lambda item: item["dataset_name"])
    return metric_names, _build_summary(metric_names)


def _walk_mapping(
    value: dict[str, Any], path: tuple[str, ...], collector: dict[str, dict[str, Any]]
) -> None:
    _extract_from_container(value, path, collector)
    for key, child in value.items():
        if key in {"metrics", "losses"}:
            continue
        if isinstance(child, dict):
            _walk_mapping(child, path + (str(key),), collector)


def _extract_from_container(
    value: dict[str, Any], path: tuple[str, ...], collector: dict[str, dict[str, Any]]
) -> None:
    dataset_name = _build_dataset_label(path)
    metrics_block = value.get("metrics")
    losses_block = value.get("losses")
    emitted: set[str] = set()

    if isinstance(metrics_block, dict):
        emitted |= _emit_metric_block(
            metrics_block, dataset_name, collector, "metrics", bool(losses_block)
        )
    elif isinstance(losses_block, dict):
        emitted |= _emit_metric_block(losses_block, dataset_name, collector, "losses", False)

    for key, child in value.items():
        if key in {"metrics", "losses"} or key in SKIP_SCALAR_KEYS:
            continue
        if str(key) in emitted or not _looks_metric_key(str(key)):
            continue
        series = _numeric_series(child)
        if series is None:
            continue
        metric_name = _display_metric_name(str(key), "scalar", False)
        _register_metric(metric_name, dataset_name, series, collector)


def _emit_metric_block(
    block: dict[str, Any],
    dataset_name: str,
    collector: dict[str, dict[str, Any]],
    block_name: str,
    has_loss_block: bool,
) -> set[str]:
    emitted: set[str] = set()
    for key, child in block.items():
        if str(key).strip().lower() in SKIP_SCALAR_KEYS:
            continue
        series = _numeric_series(child)
        if series is None:
            continue
        metric_name = _display_metric_name(str(key), block_name, has_loss_block)
        _register_metric(metric_name, dataset_name, series, collector)
        emitted.add(str(key))
    return emitted


def _register_metric(
    metric_name: str,
    dataset_name: str,
    series: list[float],
    collector: dict[str, dict[str, Any]],
) -> None:
    lower_is_better = _infer_lower_is_better(metric_name)
    metric = collector.setdefault(
        metric_name,
        {
            "metric_name": metric_name,
            "lower_is_better": lower_is_better,
            "description": f"{metric_name} extracted deterministically from experiment_data.npy.",
            "data": [],
        },
    )
    metric["data"].append(
        {
            "dataset_name": dataset_name,
            "final_value": _last_finite(series),
            "best_value": _best_finite(series, lower_is_better),
        }
    )


def _build_dataset_label(path: tuple[str, ...]) -> str:
    parts = [part for part in path if part not in GENERIC_PATH_PARTS]
    if not parts:
        return "overall"
    base = parts[0]
    if len(parts) == 1:
        return base
    return f"{base} ({', '.join(parts[1:])})"


def _display_metric_name(key: str, block_name: str, has_loss_block: bool) -> str:
    lowered = key.strip().lower()
    if lowered in SPECIAL_METRIC_NAMES:
        return SPECIAL_METRIC_NAMES[lowered]
    if block_name == "losses" and lowered in {"train", "val", "validation", "test"}:
        return _loss_alias(lowered)
    if block_name == "metrics" and has_loss_block and lowered in {"train", "val", "validation", "test"}:
        return _loss_alias(lowered)
    if lowered == "train":
        return "training metric"
    if lowered in {"val", "validation"}:
        return "validation metric"
    if lowered == "test":
        return "test metric"
    return lowered.replace("_", " ")


def _loss_alias(key: str) -> str:
    if key == "train":
        return "training loss"
    if key in {"val", "validation"}:
        return "validation loss"
    return "test loss"


def _infer_lower_is_better(metric_name: str) -> bool:
    lowered = metric_name.lower()
    if any(token in lowered for token in NEGATIVE_HINTS):
        return True
    if any(token in lowered for token in POSITIVE_HINTS):
        return False
    return False


def _looks_metric_key(key: str) -> bool:
    lowered = key.strip().lower()
    if lowered in SKIP_SCALAR_KEYS:
        return False
    if lowered in SPECIAL_METRIC_NAMES:
        return True
    return any(token in lowered for token in POSITIVE_HINTS | NEGATIVE_HINTS)


def _numeric_series(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0 or not np.issubdtype(value.dtype, np.number):
            return None
        flat = value.astype(float).reshape(-1).tolist()
        filtered = [v for v in flat if np.isfinite(v)]
        return filtered or None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        if all(isinstance(item, (int, float, np.number)) for item in value):
            filtered = [float(item) for item in value if np.isfinite(float(item))]
            return filtered or None
        return None
    if isinstance(value, (int, float, np.number)):
        scalar = float(value)
        if np.isfinite(scalar):
            return [scalar]
    return None


def _last_finite(series: list[float]) -> float:
    return float(series[-1])


def _best_finite(series: list[float], lower_is_better: bool) -> float:
    chooser = min if lower_is_better else max
    return float(chooser(series))


def _build_summary(metric_names: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for metric in metric_names:
        lines.append(f"Metric: {metric['metric_name']}")
        for point in metric["data"]:
            lines.append(
                "  "
                + f"{point['dataset_name']}: "
                + f"final={point['final_value']}, best={point['best_value']}"
            )
    return "\n".join(lines)


def _metric_sort_key(metric: dict[str, Any]) -> tuple[int, str]:
    name = str(metric["metric_name"]).lower()
    if "validation loss" in name:
        return (0, name)
    if any(token in name for token in {"accuracy", "f1", "auc", "success rate"}):
        return (1, name)
    if any(token in name for token in {"research efficiency", "return", "yield", "average metric"}):
        return (2, name)
    if "training loss" in name:
        return (3, name)
    if any(token in name for token in {"loss", "error", "waste ratio", "ratio"}):
        return (4, name)
    return (5, name)
