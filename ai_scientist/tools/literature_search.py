import os
from typing import Any, Dict, List, Optional

from ai_scientist.tools import openalex, semantic_scholar


SUPPORTED_LITERATURE_BACKENDS = ("semantic_scholar", "openalex")


def get_literature_backend() -> str:
    backend = os.getenv("AI_SCIENTIST_LITERATURE_BACKEND", "semantic_scholar")
    return backend.strip().lower()


def search_for_papers(query: str, result_limit: int = 10) -> Optional[List[Dict[str, Any]]]:
    backend = get_literature_backend()
    if backend in {"semantic_scholar", "s2"}:
        return semantic_scholar.search_for_papers(query, result_limit=result_limit)
    if backend == "openalex":
        return openalex.search_for_papers(query, result_limit=result_limit)
    raise ValueError(
        "Unknown literature backend "
        f"{backend!r}. Supported backends: {', '.join(SUPPORTED_LITERATURE_BACKENDS)}. "
        "Set env AI_SCIENTIST_LITERATURE_BACKEND to choose."
    )

