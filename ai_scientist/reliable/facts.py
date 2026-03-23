from __future__ import annotations

import json
import os.path as osp
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, MutableMapping

from .errors import FactStoreFormatError, InvalidFactKeyError, UnknownFactKeyError

_FACT_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")


def validate_fact_key(key: str) -> None:
    if not isinstance(key, str) or not key:
        raise InvalidFactKeyError(f"Fact key must be a non-empty string, got: {key!r}")
    if not _FACT_KEY_PATTERN.match(key):
        raise InvalidFactKeyError(
            "Invalid fact key (allowed: [A-Za-z0-9_.:-], max 200 chars, "
            f"must start with alnum): {key!r}"
        )


JsonValue = Any


@dataclass(frozen=True)
class FactRecord:
    key: str
    meaning: str
    value: JsonValue
    format: str | None = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    unit: str | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    short_alias: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "key": self.key,
            "meaning": self.meaning,
            "value": self.value,
            "format": self.format,
            "provenance": dict(self.provenance),
        }
        if self.unit is not None:
            d["unit"] = self.unit
        if self.ci_lower is not None:
            d["ci_lower"] = self.ci_lower
        if self.ci_upper is not None:
            d["ci_upper"] = self.ci_upper
        if self.short_alias is not None:
            d["short_alias"] = self.short_alias
        return d

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "FactRecord":
        if "key" not in data or "meaning" not in data or "value" not in data:
            raise FactStoreFormatError(
                "FactRecord requires keys: 'key', 'meaning', 'value'. "
                f"Got keys: {sorted(data.keys())}"
            )
        key = str(data["key"])
        validate_fact_key(key)
        meaning = str(data["meaning"])
        value = data["value"]
        fmt = data.get("format")
        fmt = None if fmt is None else str(fmt)
        provenance = data.get("provenance") or {}
        if not isinstance(provenance, Mapping):
            raise FactStoreFormatError(
                f"FactRecord.provenance must be a mapping, got: {type(provenance)}"
            )
        unit = data.get("unit")
        unit = None if unit is None else str(unit)
        ci_lower_raw = data.get("ci_lower")
        ci_lower = float(ci_lower_raw) if ci_lower_raw is not None else None
        ci_upper_raw = data.get("ci_upper")
        ci_upper = float(ci_upper_raw) if ci_upper_raw is not None else None
        short_alias = data.get("short_alias")
        short_alias = None if short_alias is None else str(short_alias)
        return FactRecord(
            key=key,
            meaning=meaning,
            value=value,
            format=fmt,
            provenance=dict(provenance),
            unit=unit,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            short_alias=short_alias,
        )


@dataclass
class FactStore:
    facts: MutableMapping[str, FactRecord] = field(default_factory=dict)

    def add(self, record: FactRecord, *, allow_update: bool = True) -> None:
        validate_fact_key(record.key)
        if record.key in self.facts and not allow_update:
            raise FactStoreFormatError(f"Duplicate fact key not allowed: {record.key!r}")
        self.facts[record.key] = record

    def get(self, key: str) -> FactRecord:
        validate_fact_key(key)
        record = self.facts.get(key)
        if record is None:
            raise UnknownFactKeyError(f"Unknown fact key: {key!r}")
        return record

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "auctrace.fact_store.v1",
            "facts": [rec.to_dict() for rec in self.facts.values()],
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "FactStore":
        facts = data.get("facts")
        if not isinstance(facts, list):
            raise FactStoreFormatError(
                "FactStore JSON must have 'facts' as a list. "
                f"Got type: {type(facts)}"
            )
        store = FactStore()
        for item in facts:
            if not isinstance(item, Mapping):
                raise FactStoreFormatError(
                    f"FactStore facts must be list[dict], got: {type(item)}"
                )
            store.add(FactRecord.from_dict(item))
        return store

    def save_json(self, path: str) -> None:
        parent = osp.dirname(path)
        if parent:
            # Do not create deep trees; writeup dirs already exist.
            # This is a best-effort convenience.
            try:
                import os

                os.makedirs(parent, exist_ok=True)
            except Exception:
                pass
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=True)

    @staticmethod
    def load_json(path: str) -> "FactStore":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, Mapping):
            raise FactStoreFormatError(
                f"FactStore JSON must be an object, got: {type(data)}"
            )
        return FactStore.from_dict(data)


def format_fact_value_for_latex(record: FactRecord) -> str:
    """Format a fact value for insertion into LaTeX.

    This is intentionally deterministic and conservative. For the first pass,
    we focus on numeric/string primitives and provide a minimal formatting
    surface. More sophisticated formatting (CI, units, etc.) can be layered
    later without changing the placeholder contract.
    """

    value = record.value
    fmt = record.format

    rendered: str
    if fmt is None:
        rendered = str(value)
    else:
        rendered = _apply_simple_format(value, fmt)

    return _escape_latex_text(rendered)


def format_fact_ci_for_latex(record: FactRecord) -> str:
    """Render a 95% CI as '(lo, hi)' for \\factci{KEY}. Returns '' if no CI stored."""
    if record.ci_lower is None or record.ci_upper is None:
        return ""
    lo = _apply_simple_format(record.ci_lower, record.format) if record.format else f"{record.ci_lower:.4f}"
    hi = _apply_simple_format(record.ci_upper, record.format) if record.format else f"{record.ci_upper:.4f}"
    return _escape_latex_text(f"({lo}, {hi})")


def format_fact_with_unit_for_latex(record: FactRecord) -> str:
    """Render value with unit appended for \\factunit{KEY}."""
    base = format_fact_value_for_latex(record)
    if record.unit:
        return base + "~" + _escape_latex_text(record.unit)
    return base


def _apply_simple_format(value: JsonValue, fmt: str) -> str:
    # Supported:
    # - "float:N" -> N decimal places
    # - "int" -> integer cast
    # - "percent:N" -> value interpreted as fraction [0,1], rendered as 100*value with N decimals and \%
    if fmt.startswith("float:"):
        decimals = int(fmt.split(":", 1)[1])
        return f"{float(value):.{decimals}f}"
    if fmt == "int":
        return str(int(value))
    if fmt.startswith("percent:"):
        decimals = int(fmt.split(":", 1)[1])
        pct = 100.0 * float(value)
        return f"{pct:.{decimals}f}%"
    raise FactStoreFormatError(f"Unsupported fact format: {fmt!r}")


def _escape_latex_text(text: str) -> str:
    # Keep this minimal; most facts are numeric.
    # - Escape percent because it starts LaTeX comments.
    # - Escape ampersand because it breaks tables.
    # - Escape underscore because it breaks text mode.
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("_", "\\_")
    )


def facts_index_for_prompt(store: FactStore) -> str:
    """Produce a compact index of fact keys for LLM prompting.

    Values are intentionally omitted to prevent the LLM from copying numbers.
    Alias, unit, and CI hints are shown so the LLM knows which macros to use.
    """

    lines: list[str] = []
    for key in sorted(store.facts.keys()):
        rec = store.facts[key]
        meaning = rec.meaning.strip().replace("\n", " ")
        line = f"- {key}: {meaning}"
        hints: list[str] = []
        if rec.short_alias:
            hints.append(f"alias={rec.short_alias}")
        if rec.unit:
            hints.append(f"unit={rec.unit}")
        if rec.ci_lower is not None and rec.ci_upper is not None:
            hints.append(r"has_ci → \factci{" + key + "}")
        if hints:
            line += f"  [{', '.join(hints)}]"
        lines.append(line)
    return "\n".join(lines)


def build_store(records: Iterable[FactRecord]) -> FactStore:
    store = FactStore()
    for rec in records:
        store.add(rec)
    return store

