from __future__ import annotations

import difflib
import os.path as osp
import re
from typing import Any, Mapping

from .artifact_manifest import figure_includegraphics_targets
from .errors import (
    InvalidFigurePathError,
    SymbolicLatexError,
    UnknownFactKeyError,
    UnknownParamKeyError,
)
from .facts import FactStore
from .params import ParamStore
from .placeholders import list_fact_keys_in_order, list_param_keys_in_order

_INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?{([^}]+)}")


def require_fact_store(path: str) -> FactStore:
    if not osp.exists(path):
        raise FileNotFoundError(
            f"Missing fact_store.json at {path!r}. "
            "Run fact extraction first (or disable symbolic facts)."
        )
    store = FactStore.load_json(path)
    if not store.facts:
        raise SymbolicLatexError(
            f"FactStore is empty at {path!r}. Cannot run symbolic-facts writeup."
        )
    return store


def require_symbolic_latex_uses_facts(symbolic_tex: str, *, min_unique: int = 1) -> list[str]:
    keys = list_fact_keys_in_order(symbolic_tex)
    if len(keys) < min_unique:
        raise SymbolicLatexError(
            "Symbolic-facts mode requires fact placeholders in LaTeX. "
            f"Expected at least {min_unique} unique \\\\fact{{key}} usage, got {len(keys)}."
        )
    return keys


def require_no_fact_placeholders(tex: str) -> None:
    if "\\fact{" not in tex:
        return
    keys = list_fact_keys_in_order(tex)
    raise SymbolicLatexError(
        "Non-symbolic writeup mode forbids \\\\fact{KEY} placeholders, "
        f"but found {len(keys)} unique fact keys (example={keys[:3]}). "
        "Disable symbolic facts or output numeric values directly."
    )


def require_no_param_placeholders(tex: str) -> None:
    if "\\param{" not in tex:
        return
    keys = list_param_keys_in_order(tex)
    raise SymbolicLatexError(
        "Non-symbolic writeup mode forbids \\\\param{KEY} placeholders, "
        f"but found {len(keys)} unique param keys (example={keys[:3]}). "
        "Disable symbolic facts or output literal setup values directly."
    )


def require_symbolic_tex_uses_known_facts(
    symbolic_tex: str, store: FactStore, *, min_unique: int = 1
) -> list[str]:
    keys = require_symbolic_latex_uses_facts(symbolic_tex, min_unique=min_unique)
    unknown = [key for key in keys if key not in store.facts]
    if not unknown:
        return keys
    details: list[str] = []
    for key in unknown[:5]:
        matches = difflib.get_close_matches(key, store.facts.keys(), n=3, cutoff=0.55)
        if matches:
            details.append(f"{key!r} (did you mean {matches!r}?)")
        else:
            details.append(repr(key))
    raise UnknownFactKeyError(
        "Symbolic-facts mode only allows exact fact keys from fact_store.json, "
        f"but found {len(unknown)} unknown key(s): {', '.join(details)}. "
        "Use an existing key exactly as listed, or rewrite the sentence qualitatively."
    )


def require_symbolic_tex_file_uses_facts(path: str, *, min_unique: int = 1) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        tex = f.read()
    return require_symbolic_latex_uses_facts(tex, min_unique=min_unique)


def require_symbolic_tex_uses_known_params(
    symbolic_tex: str,
    store: ParamStore,
) -> list[str]:
    keys = list_param_keys_in_order(symbolic_tex)
    unknown = [key for key in keys if key not in store.params]
    if not unknown:
        return keys
    details: list[str] = []
    for key in unknown[:5]:
        matches = difflib.get_close_matches(key, store.params.keys(), n=3, cutoff=0.55)
        if matches:
            details.append(f"{key!r} (did you mean {matches!r}?)")
        else:
            details.append(repr(key))
    raise UnknownParamKeyError(
        "Symbolic-facts mode only allows exact param keys from param_store.json, "
        f"but found {len(unknown)} unknown key(s): {', '.join(details)}. "
        "Use an existing key exactly as listed, or rewrite the sentence qualitatively."
    )


def _normalize_graphics_target(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def require_includegraphics_paths_from_manifest(
    tex: str,
    artifact_manifest: Mapping[str, Any],
) -> list[str]:
    raw_targets = _INCLUDEGRAPHICS_RE.findall(tex)
    targets = [_normalize_graphics_target(item) for item in raw_targets if item.strip()]
    if not targets:
        return []
    allowed = set(figure_includegraphics_targets(artifact_manifest))
    invalid = [target for target in targets if target not in allowed]
    if not invalid:
        return targets
    figure_choices = [item for item in sorted(allowed) if "/" not in item or item.startswith("../figures/")]
    details: list[str] = []
    for target in invalid[:5]:
        matches = difflib.get_close_matches(target, figure_choices, n=3, cutoff=0.45)
        if matches:
            details.append(f"{target!r} (did you mean {matches!r}?)")
        else:
            details.append(repr(target))
    raise InvalidFigurePathError(
        "LaTeX figure references must come from the current artifact manifest, "
        f"but found {len(invalid)} invalid includegraphics target(s): {', '.join(details)}. "
        "Use only exact strings from artifact_manifest_summary.includegraphics_targets."
    )
