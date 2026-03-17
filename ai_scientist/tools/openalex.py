import os
import time
import warnings
from typing import Any, Dict, List, Optional

import backoff
import requests

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.tools.openalex_formatting import (
    abstract_from_inverted_index,
    extract_venue,
    work_to_bibtex,
)


OPENALEX_API_BASE_URL = "https://api.openalex.org"

# OpenAlex `select` only supports top-level fields (not nested). Keep this small
# to reduce response size and parsing cost.
OPENALEX_WORK_SELECT_FIELDS = ",".join(
    [
        "id",
        "doi",
        "ids",
        "title",
        "display_name",
        "publication_year",
        "publication_date",
        "type",
        "authorships",
        "primary_location",
        "biblio",
        "cited_by_count",
        "abstract_inverted_index",
    ]
)

_WARNED_MISSING_OPENALEX_API_KEY = False


def _on_backoff(details: Dict[str, Any]) -> None:
    wait_s = details.get("wait")
    tries = details.get("tries")
    target = details.get("target")
    fn_name = getattr(target, "__name__", str(target))
    print(
        f"[openalex] backing off {wait_s:0.1f}s after {tries} tries "
        f"calling {fn_name} at {time.strftime('%X')}"
    )


def _is_retryable_http_error(exc: BaseException) -> bool:
    if not isinstance(exc, requests.exceptions.HTTPError):
        return False
    rsp = getattr(exc, "response", None)
    if rsp is None:
        return False
    return rsp.status_code in {429, 500, 502, 503, 504}


def _giveup_on_exception(exc: BaseException) -> bool:
    # Give up immediately for non-retryable HTTP status codes.
    if isinstance(exc, requests.exceptions.HTTPError):
        return not _is_retryable_http_error(exc)
    return False


def _openalex_request_params(result_limit: int) -> Dict[str, Any]:
    global _WARNED_MISSING_OPENALEX_API_KEY
    params: Dict[str, Any] = {
        "per_page": max(1, min(int(result_limit), 100)),
        "select": OPENALEX_WORK_SELECT_FIELDS,
    }
    api_key = os.getenv("OPENALEX_API_KEY")
    if api_key:
        params["api_key"] = api_key
    else:
        if not _WARNED_MISSING_OPENALEX_API_KEY:
            _WARNED_MISSING_OPENALEX_API_KEY = True
            warnings.warn(
                "No OpenAlex API key found (env OPENALEX_API_KEY). "
                "Some endpoints may be rate-limited or unavailable without a key."
            )

    mailto = os.getenv("OPENALEX_MAILTO")
    if mailto:
        params["mailto"] = mailto
    return params


@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.HTTPError, requests.exceptions.ConnectionError, requests.exceptions.Timeout),
    on_backoff=_on_backoff,
    giveup=_giveup_on_exception,
    max_tries=6,
)
def search_for_papers(query: str, result_limit: int = 10) -> Optional[List[Dict[str, Any]]]:
    if not query:
        return None

    params = _openalex_request_params(result_limit=result_limit)
    params["search"] = query

    rsp = requests.get(
        f"{OPENALEX_API_BASE_URL}/works",
        params=params,
        timeout=30,
    )
    if rsp.status_code != 200:
        print(f"[openalex] status={rsp.status_code} body={rsp.text[:500]}")
    rsp.raise_for_status()

    payload = rsp.json()
    works = payload.get("results") or []
    if not works:
        return None

    papers: List[Dict[str, Any]] = []
    for work in works:
        ids = work.get("ids") or {}
        abstract = abstract_from_inverted_index(work.get("abstract_inverted_index"))
        paper = {
            "title": work.get("display_name") or "",
            "authors": [
                {"name": (a.get("author") or {}).get("display_name") or ""}
                for a in (work.get("authorships") or [])
                if (a.get("author") or {}).get("display_name")
            ],
            "venue": extract_venue(work),
            "year": work.get("publication_year") or "",
            "abstract": abstract,
            "citationCount": work.get("cited_by_count") or 0,
            "citationStyles": {"bibtex": work_to_bibtex(work)},
            "openalex_id": work.get("id") or "",
            "doi": ids.get("doi") or work.get("doi") or "",
        }
        papers.append(paper)

    # Match the Semantic Scholar behavior: citation-heavy results first.
    papers.sort(key=lambda x: x.get("citationCount", 0), reverse=True)
    return papers[: max(1, int(result_limit))]


class OpenAlexSearchTool(BaseTool):
    def __init__(
        self,
        name: str = "SearchOpenAlex",
        description: str = (
            "Search for relevant literature using OpenAlex. "
            "Provide a search query to find relevant papers."
        ),
        max_results: int = 10,
    ):
        parameters = [
            {
                "name": "query",
                "type": "str",
                "description": "The search query to find relevant papers.",
            }
        ]
        super().__init__(name, description, parameters)
        self.max_results = max_results

    def use_tool(self, query: str) -> Optional[str]:
        papers = search_for_papers(query, result_limit=self.max_results)
        if not papers:
            return "No papers found."

        lines: List[str] = []
        for i, paper in enumerate(papers):
            authors = ", ".join([a.get("name", "Unknown") for a in paper.get("authors", [])])
            lines.append(
                f"{i + 1}: {paper.get('title', 'Unknown Title')}. "
                f"{authors}. {paper.get('venue', 'Unknown Venue')}, {paper.get('year', 'Unknown Year')}.\n"
                f"Number of citations: {paper.get('citationCount', 'N/A')}\n"
                f"Abstract: {paper.get('abstract', 'No abstract available.')}"
            )
        return "\n\n".join(lines)
