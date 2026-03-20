from __future__ import annotations

from typing import Any, Mapping

from .facts import FactStore
from .gates import (
    require_includegraphics_paths_from_manifest,
    require_no_fact_placeholders,
    require_no_param_placeholders,
    require_symbolic_tex_uses_known_facts,
    require_symbolic_tex_uses_known_params,
)
from .numeric_lint import require_no_unanchored_numeric_literals
from .params import ParamStore


def validate_generated_writeup(
    tex: str,
    *,
    symbolic_facts: bool,
    store: FactStore | None = None,
    param_store: ParamStore | None = None,
    artifact_manifest: Mapping[str, Any] | None = None,
    min_unique_fact_keys: int = 1,
) -> None:
    if not symbolic_facts:
        require_no_fact_placeholders(tex)
        require_no_param_placeholders(tex)
        return
    if store is None:
        raise ValueError("symbolic writeup validation requires a FactStore")
    if artifact_manifest is None:
        raise ValueError("symbolic writeup validation requires an artifact manifest")
    require_symbolic_tex_uses_known_facts(
        tex,
        store,
        min_unique=min_unique_fact_keys,
    )
    if "\\param{" in tex and param_store is None:
        raise ValueError("symbolic writeup validation requires a ParamStore when \\param{...} is used")
    if param_store is not None:
        require_symbolic_tex_uses_known_params(tex, param_store)
    require_includegraphics_paths_from_manifest(tex, artifact_manifest)
    require_no_unanchored_numeric_literals(tex)
