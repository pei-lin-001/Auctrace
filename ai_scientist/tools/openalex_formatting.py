import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


def strip_to_ascii(text: str) -> str:
    nfkd_form = unicodedata.normalize("NFKD", text)
    return nfkd_form.encode("ASCII", "ignore").decode("ascii")


def latex_escape(text: str) -> str:
    # Minimal LaTeX escaping to keep BibTeX robust with pdflatex.
    replacements = {
        "\\": r"\\",
        "{": r"\{",
        "}": r"\}",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "_": r"\_",
        "$": r"\$",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out: List[str] = []
    for ch in text:
        out.append(replacements.get(ch, ch))
    return "".join(out)


def abstract_from_inverted_index(inv: Optional[Dict[str, List[int]]]) -> str:
    if not inv:
        return ""

    max_pos = -1
    for positions in inv.values():
        if not positions:
            continue
        pos_max = max(positions)
        if pos_max > max_pos:
            max_pos = pos_max
    if max_pos < 0:
        return ""

    words: List[str] = [""] * (max_pos + 1)
    for token, positions in inv.items():
        for pos in positions:
            if 0 <= pos < len(words) and not words[pos]:
                words[pos] = token
    return " ".join([w for w in words if w])


def extract_venue(work: Dict[str, Any]) -> str:
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    venue = source.get("display_name") or ""
    if venue:
        return venue

    raw_source = (primary_location.get("raw_source_name") or "").strip()
    if raw_source:
        return raw_source

    # Fallback: if `locations` is present (not always selected), search for the
    # first location with a `source.display_name`.
    for loc in work.get("locations") or []:
        loc_source = (loc or {}).get("source") or {}
        loc_venue = (loc_source.get("display_name") or "").strip()
        if loc_venue:
            return loc_venue
    return ""


def _first_author_lastname(work: Dict[str, Any]) -> str:
    authorships = work.get("authorships") or []
    if not authorships:
        return "unknown"

    first = authorships[0] or {}
    author = first.get("author") or {}
    name = author.get("display_name") or author.get("name") or ""
    name = name.strip()
    if not name:
        return "unknown"
    parts = re.split(r"\\s+", name)
    last = parts[-1] if parts else "unknown"
    last_ascii = strip_to_ascii(last).lower()
    last_ascii = re.sub(r"[^a-z0-9]+", "", last_ascii)
    return last_ascii or "unknown"


def _work_id_suffix(work: Dict[str, Any]) -> str:
    raw_id = (work.get("id") or "").strip()
    if not raw_id:
        return "unknown"
    # Typical OpenAlex IDs look like: "https://openalex.org/W2741809807"
    tail = raw_id.rstrip("/").split("/")[-1]
    tail_ascii = strip_to_ascii(tail).lower()
    tail_ascii = re.sub(r"[^a-z0-9]+", "", tail_ascii)
    return tail_ascii or "unknown"


def _title_keyword(work: Dict[str, Any]) -> str:
    title = (work.get("display_name") or "").strip()
    if not title:
        return "paper"
    title_ascii = strip_to_ascii(title).lower()
    words = re.findall(r"[a-z0-9]+", title_ascii)
    return words[0] if words else "paper"


def make_cite_key(work: Dict[str, Any]) -> str:
    year = work.get("publication_year") or ""
    year_str = str(year) if year else "nodate"
    return f"{_first_author_lastname(work)}{year_str}{_title_keyword(work)}_{_work_id_suffix(work)}"


def _entry_type_and_venue_field(work: Dict[str, Any]) -> Tuple[str, str]:
    work_type = (work.get("type") or "").lower()
    if "proceedings" in work_type:
        return "inproceedings", "booktitle"
    if "journal" in work_type:
        return "article", "journal"
    return "misc", "howpublished"


def _authors_bibtex(work: Dict[str, Any]) -> str:
    authorships = work.get("authorships") or []
    names: List[str] = []
    for auth in authorships:
        author = (auth or {}).get("author") or {}
        name = author.get("display_name") or author.get("name") or ""
        name = name.strip()
        if name:
            names.append(strip_to_ascii(name))
    return " and ".join([latex_escape(n) for n in names])


def _doi_bare(doi: str) -> str:
    doi = doi.strip()
    if doi.startswith("https://doi.org/"):
        return doi[len("https://doi.org/") :]
    if doi.startswith("http://doi.org/"):
        return doi[len("http://doi.org/") :]
    return doi


def work_to_bibtex(work: Dict[str, Any]) -> str:
    entry_type, venue_field = _entry_type_and_venue_field(work)
    key = make_cite_key(work)

    title = strip_to_ascii((work.get("display_name") or "").strip())
    venue = strip_to_ascii(extract_venue(work).strip())
    year = work.get("publication_year")
    ids = work.get("ids") or {}
    doi = ids.get("doi") or work.get("doi") or ""
    doi = _doi_bare(doi) if isinstance(doi, str) else ""

    fields: Dict[str, str] = {}
    if title:
        fields["title"] = f"{{{latex_escape(title)}}}"
    authors = _authors_bibtex(work)
    if authors:
        fields["author"] = f"{{{authors}}}"
    if venue:
        fields[venue_field] = f"{{{latex_escape(venue)}}}"
    if year:
        fields["year"] = f"{{{year}}}"
    if doi:
        fields["doi"] = f"{{{latex_escape(doi)}}}"
        fields["url"] = f"{{https://doi.org/{latex_escape(doi)}}}"
    else:
        work_id = (work.get("id") or "").strip()
        if work_id:
            fields["url"] = f"{{{latex_escape(strip_to_ascii(work_id))}}}"

    lines = [f"@{entry_type}{{{key},"]
    for field_name, value in fields.items():
        lines.append(f"  {field_name} = {value},")
    lines.append("}")
    return "\n".join(lines)
