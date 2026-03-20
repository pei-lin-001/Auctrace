from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

from .facts import FactRecord

_KEY_PART_CLEAN_RE = re.compile(r"[^a-z0-9_.:-]+")
_UNDERSCORE_RE = re.compile(r"_+")


def normalize_key_part(text: str) -> str:
    out = text.strip().lower()
    out = out.replace(" ", "_")
    out = _KEY_PART_CLEAN_RE.sub("_", out)
    out = _UNDERSCORE_RE.sub("_", out)
    out = out.strip("_.:-")
    return out or "unknown"


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
        metric_part = normalize_key_part(metric_name)
        data = metric_entry.get("data") or []
        if not isinstance(data, list):
            continue

        best_vals: list[float | None] = []
        final_vals: list[float | None] = []
        for point in data:
            if not isinstance(point, Mapping):
                continue
            dataset_name = str(point.get("dataset_name", "dataset")).strip()
            dataset_part = normalize_key_part(dataset_name)

            best_val = _safe_float(point.get("best_value"))
            final_val = _safe_float(point.get("final_value"))
            best_vals.append(best_val)
            final_vals.append(final_val)

            if best_val is not None:
                records.append(
                    FactRecord(
                        key=f"{prefix}.{dataset_part}.{metric_part}.best",
                        meaning=(
                            f"{prefix} best value for metric '{metric_name}' "
                            f"on dataset '{dataset_name}'"
                        ),
                        value=best_val,
                        format="float:4",
                        provenance={
                            **provenance.to_dict(),
                            "metric_name": metric_name,
                            "dataset_name": dataset_name,
                            "value_kind": "best_value",
                        },
                    )
                )

            if final_val is not None:
                records.append(
                    FactRecord(
                        key=f"{prefix}.{dataset_part}.{metric_part}.final",
                        meaning=(
                            f"{prefix} final value for metric '{metric_name}' "
                            f"on dataset '{dataset_name}'"
                        ),
                        value=final_val,
                        format="float:4",
                        provenance={
                            **provenance.to_dict(),
                            "metric_name": metric_name,
                            "dataset_name": dataset_name,
                            "value_kind": "final_value",
                        },
                    )
                )

        mean_best = _mean(best_vals)
        mean_final = _mean(final_vals)
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

