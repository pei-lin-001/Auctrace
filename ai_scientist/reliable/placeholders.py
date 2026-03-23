from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import InvalidFactKeyError, InvalidParamKeyError
from .facts import validate_fact_key
from .params import validate_param_key

_FACT_MACRO_RE = re.compile(r"\\fact\{([^{}]+)\}")
_PARAM_MACRO_RE = re.compile(r"\\param\{([^{}]+)\}")
_FACTCI_MACRO_RE = re.compile(r"\\factci\{([^{}]+)\}")
_FACTUNIT_MACRO_RE = re.compile(r"\\factunit\{([^{}]+)\}")


@dataclass(frozen=True)
class FactPlaceholder:
    key: str
    start: int
    end: int


@dataclass(frozen=True)
class ParamPlaceholder:
    key: str
    start: int
    end: int


@dataclass(frozen=True)
class FactCIPlaceholder:
    key: str
    start: int
    end: int


@dataclass(frozen=True)
class FactUnitPlaceholder:
    key: str
    start: int
    end: int


def iter_fact_placeholders(tex: str) -> list[FactPlaceholder]:
    placeholders: list[FactPlaceholder] = []
    for match in _FACT_MACRO_RE.finditer(tex):
        key = match.group(1).strip()
        try:
            validate_fact_key(key)
        except InvalidFactKeyError as e:
            raise InvalidFactKeyError(
                f"Invalid \\fact{{...}} key at [{match.start()}:{match.end()}]: {e}"
            ) from e
        placeholders.append(FactPlaceholder(key=key, start=match.start(), end=match.end()))
    return placeholders


def list_fact_keys_in_order(tex: str) -> list[str]:
    seen: set[str] = set()
    keys: list[str] = []
    for ph in iter_fact_placeholders(tex):
        if ph.key in seen:
            continue
        seen.add(ph.key)
        keys.append(ph.key)
    return keys


def iter_param_placeholders(tex: str) -> list[ParamPlaceholder]:
    placeholders: list[ParamPlaceholder] = []
    for match in _PARAM_MACRO_RE.finditer(tex):
        key = match.group(1).strip()
        try:
            validate_param_key(key)
        except InvalidParamKeyError as e:
            raise InvalidParamKeyError(
                f"Invalid \\param{{...}} key at [{match.start()}:{match.end()}]: {e}"
            ) from e
        placeholders.append(ParamPlaceholder(key=key, start=match.start(), end=match.end()))
    return placeholders


def list_param_keys_in_order(tex: str) -> list[str]:
    seen: set[str] = set()
    keys: list[str] = []
    for ph in iter_param_placeholders(tex):
        if ph.key in seen:
            continue
        seen.add(ph.key)
        keys.append(ph.key)
    return keys


def iter_factci_placeholders(tex: str) -> list[FactCIPlaceholder]:
    placeholders: list[FactCIPlaceholder] = []
    for match in _FACTCI_MACRO_RE.finditer(tex):
        key = match.group(1).strip()
        try:
            validate_fact_key(key)
        except InvalidFactKeyError as e:
            raise InvalidFactKeyError(
                f"Invalid \\factci{{...}} key at [{match.start()}:{match.end()}]: {e}"
            ) from e
        placeholders.append(FactCIPlaceholder(key=key, start=match.start(), end=match.end()))
    return placeholders


def iter_factunit_placeholders(tex: str) -> list[FactUnitPlaceholder]:
    placeholders: list[FactUnitPlaceholder] = []
    for match in _FACTUNIT_MACRO_RE.finditer(tex):
        key = match.group(1).strip()
        try:
            validate_fact_key(key)
        except InvalidFactKeyError as e:
            raise InvalidFactKeyError(
                f"Invalid \\factunit{{...}} key at [{match.start()}:{match.end()}]: {e}"
            ) from e
        placeholders.append(FactUnitPlaceholder(key=key, start=match.start(), end=match.end()))
    return placeholders
