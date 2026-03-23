from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

from .facts import FactRecord, FactStore

_KEY_PART_CLEAN_RE = re.compile(r"[^a-z0-9_.:-]+")
_UNDERSCORE_RE = re.compile(r"_+")
_HASH_LEN = 10
_MAX_FACT_KEY_LEN = 200
_ABLATION_PART_MAX_LEN = 48
_DATASET_PART_MAX_LEN = 56
_METRIC_PART_MAX_LEN = 48


def normalize_key_part(text: str) -> str:
    out = text.strip().lower()
    out = out.replace(" ", "_")
    out = _KEY_PART_CLEAN_RE.sub("_", out)
    out = _UNDERSCORE_RE.sub("_", out)
    out = out.strip("_.:-")
    return out or "unknown"


def shorten_key_part(text: str, *, max_len: int) -> str:
    normalized = normalize_key_part(text)
    if len(normalized) <= max_len:
        return normalized

    if max_len <= _HASH_LEN + 2:
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        return f"k{digest}"[:max_len]

    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:_HASH_LEN]
    head_len = max_len - _HASH_LEN - 1
    head = normalized[:head_len].rstrip("_.:-")
    if not head:
        head = normalized[:1]
    return f"{head}_{digest}"


def build_ablation_fact_prefix(ablation_name: str) -> str:
    ablation_part = shorten_key_part(
        ablation_name,
        max_len=_ABLATION_PART_MAX_LEN,
    )
    return f"stage4.ablation.{ablation_part}"


def _metric_fact_key(prefix: str, dataset_name: str, metric_name: str, value_kind: str) -> str:
    dataset_part = shorten_key_part(dataset_name, max_len=_DATASET_PART_MAX_LEN)
    metric_part = shorten_key_part(metric_name, max_len=_METRIC_PART_MAX_LEN)
    key = f"{prefix}.{dataset_part}.{metric_part}.{value_kind}"
    if len(key) <= _MAX_FACT_KEY_LEN:
        return key

    fixed_len = len(prefix) + len(value_kind) + 3
    remaining = max(32, _MAX_FACT_KEY_LEN - fixed_len)
    dataset_budget = max(16, remaining // 2)
    metric_budget = max(16, remaining - dataset_budget)
    dataset_part = shorten_key_part(dataset_name, max_len=dataset_budget)
    metric_part = shorten_key_part(metric_name, max_len=metric_budget)
    key = f"{prefix}.{dataset_part}.{metric_part}.{value_kind}"
    if len(key) <= _MAX_FACT_KEY_LEN:
        return key

    available = max(24, _MAX_FACT_KEY_LEN - fixed_len)
    dataset_budget = max(12, available // 2)
    metric_budget = max(12, available - dataset_budget)
    dataset_part = shorten_key_part(dataset_name, max_len=dataset_budget)
    metric_part = shorten_key_part(metric_name, max_len=metric_budget)
    return f"{prefix}.{dataset_part}.{metric_part}.{value_kind}"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _mean(values: Sequence[float | None]) -> float | None:
    vs = [v for v in values if v is not None]
    if not vs:
        return None
    return float(sum(vs) / len(vs))


def _stdev(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)


def _ci95_bounds(values: Sequence[float]) -> tuple[float | None, float | None]:
    """Return (lower, upper) 95% CI using 1.96 * std / sqrt(n). Returns (None, None) for n < 2."""
    n = len(values)
    if n < 2:
        return None, None
    mean = sum(values) / n
    se = _stdev(values) / math.sqrt(n)
    margin = 1.96 * se
    return mean - margin, mean + margin


@dataclass(frozen=True)
class NodeMetricProvenance:
    stage_name: str
    node_id: str | None
    exp_results_dir: str | None
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "node_id": self.node_id,
            "exp_results_dir": self.exp_results_dir,
            "source": self.source,
        }


def add_delta_facts(
    store: FactStore,
    *,
    stage2_prefix: str,
    stage3_prefix: str,
    delta_prefix: str = "delta",
) -> list[FactRecord]:
    """Compute delta facts: stage3_prefix.* minus stage2_prefix.* for matching numeric keys.

    Returned records are NOT added to the store; the caller decides whether to
    call store.add() on them. This keeps the function pure and testable.
    """
    delta_records: list[FactRecord] = []
    for key3, rec3 in store.facts.items():
        if not key3.startswith(stage3_prefix + "."):
            continue
        suffix = key3[len(stage3_prefix) + 1:]
        key2 = stage2_prefix + "." + suffix
        if key2 not in store.facts:
            continue
        rec2 = store.facts[key2]
        v3 = _safe_float(rec3.value)
        v2 = _safe_float(rec2.value)
        if v3 is None or v2 is None:
            continue
        delta = v3 - v2
        delta_key = delta_prefix + "." + suffix
        delta_records.append(
            FactRecord(
                key=delta_key,
                meaning=f"Delta ({stage3_prefix} minus {stage2_prefix}) for {suffix}",
                value=delta,
                format=rec3.format,
                provenance={
                    "source": "add_delta_facts",
                    "stage3_key": key3,
                    "stage2_key": key2,
                    "stage3_value": v3,
                    "stage2_value": v2,
                },
            )
        )
    return delta_records


def extract_facts_from_metric_payload(
    *,
    prefix: str,
    metric_payload: Mapping[str, Any],
    provenance: NodeMetricProvenance,
) -> list[FactRecord]:
    value = metric_payload.get("value")
    if not isinstance(value, Mapping):
        return []

    metric_names = value.get("metric_names")
    if not isinstance(metric_names, list):
        return []

    records: list[FactRecord] = []
    for metric_entry in metric_names:
        if not isinstance(metric_entry, Mapping):
            continue
        metric_name = str(metric_entry.get("metric_name", "metric")).strip()
        metric_part = shorten_key_part(metric_name, max_len=_METRIC_PART_MAX_LEN)
        data = metric_entry.get("data") or []
        if not isinstance(data, list):
            continue

        # Group points by dataset_name so multiple entries for the same dataset
        # are treated as independent seeds and get a proper 95% CI.
        by_dataset: dict[str, list[Mapping]] = defaultdict(list)
        for point in data:
            if not isinstance(point, Mapping):
                continue
            ds = str(point.get("dataset_name", "dataset")).strip()
            by_dataset[ds].append(point)

        all_best: list[float] = []
        all_final: list[float] = []

        for dataset_name, points in by_dataset.items():
            best_seeds = [v for v in (_safe_float(p.get("best_value")) for p in points) if v is not None]
            final_seeds = [v for v in (_safe_float(p.get("final_value")) for p in points) if v is not None]

            mean_best_ds = _mean(best_seeds)
            mean_final_ds = _mean(final_seeds)
            ci_lo_best, ci_hi_best = _ci95_bounds(best_seeds)
            ci_lo_final, ci_hi_final = _ci95_bounds(final_seeds)

            ds_part = shorten_key_part(dataset_name, max_len=16)
            m_part = shorten_key_part(metric_name, max_len=16)

            if mean_best_ds is not None:
                all_best.append(mean_best_ds)
                alias = f"{m_part}.{ds_part}.best"
                records.append(
                    FactRecord(
                        key=_metric_fact_key(prefix, dataset_name, metric_name, "best"),
                        meaning=(
                            f"{prefix} best value for metric '{metric_name}' "
                            f"on dataset '{dataset_name}'"
                        ),
                        value=mean_best_ds,
                        format="float:4",
                        provenance={
                            **provenance.to_dict(),
                            "metric_name": metric_name,
                            "dataset_name": dataset_name,
                            "value_kind": "best_value",
                            "n_seeds": len(best_seeds),
                        },
                        ci_lower=ci_lo_best,
                        ci_upper=ci_hi_best,
                        short_alias=alias if len(alias) <= 48 else None,
                    )
                )

            if mean_final_ds is not None:
                all_final.append(mean_final_ds)
                alias = f"{m_part}.{ds_part}.final"
                records.append(
                    FactRecord(
                        key=_metric_fact_key(prefix, dataset_name, metric_name, "final"),
                        meaning=(
                            f"{prefix} final value for metric '{metric_name}' "
                            f"on dataset '{dataset_name}'"
                        ),
                        value=mean_final_ds,
                        format="float:4",
                        provenance={
                            **provenance.to_dict(),
                            "metric_name": metric_name,
                            "dataset_name": dataset_name,
                            "value_kind": "final_value",
                            "n_seeds": len(final_seeds),
                        },
                        ci_lower=ci_lo_final,
                        ci_upper=ci_hi_final,
                        short_alias=alias if len(alias) <= 48 else None,
                    )
                )

        mean_best = _mean(all_best)
        mean_final = _mean(all_final)
        if mean_best is not None:
            records.append(
                FactRecord(
                    key=f"{prefix}.mean.{metric_part}.best",
                    meaning=f"{prefix} mean best value for metric '{metric_name}' across datasets",
                    value=mean_best,
                    format="float:4",
                    provenance={
                        **provenance.to_dict(),
                        "metric_name": metric_name,
                        "dataset_name": "mean",
                        "value_kind": "best_value",
                    },
                )
            )
        if mean_final is not None:
            records.append(
                FactRecord(
                    key=f"{prefix}.mean.{metric_part}.final",
                    meaning=f"{prefix} mean final value for metric '{metric_name}' across datasets",
                    value=mean_final,
                    format="float:4",
                    provenance={
                        **provenance.to_dict(),
                        "metric_name": metric_name,
                        "dataset_name": "mean",
                        "value_kind": "final_value",
                    },
                )
            )
    return records
