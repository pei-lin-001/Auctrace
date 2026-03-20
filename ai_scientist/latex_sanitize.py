from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable


UNICODE_REPLACEMENTS: dict[str, str] = {
    "\u00A0": " ",  # NO-BREAK SPACE
    "\u202F": " ",  # NARROW NO-BREAK SPACE
    "\u2010": "-",  # HYPHEN
    "\u2011": "-",  # NON-BREAKING HYPHEN
    "\u2013": "--",  # EN DASH
    "\u2014": "---",  # EM DASH
    "\u2212": "-",  # MINUS SIGN
    "\u2248": "\\ensuremath{\\approx}",  # ALMOST EQUAL TO
    "\u2264": "\\ensuremath{\\leq}",  # LESS-THAN OR EQUAL TO
    "\u2265": "\\ensuremath{\\geq}",  # GREATER-THAN OR EQUAL TO
    "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK
    "\u201C": "\"",  # LEFT DOUBLE QUOTATION MARK
    "\u201D": "\"",  # RIGHT DOUBLE QUOTATION MARK
}


@dataclass(frozen=True)
class SanitizeReport:
    changed: bool
    replacements: Dict[str, int]
    remaining_non_ascii: Dict[str, int]


def _count_occurrences(text: str, chars: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for ch in chars:
        n = text.count(ch)
        if n:
            counts[ch] = n
    return counts


def _count_remaining_non_ascii(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for ch in text:
        if ord(ch) <= 127:
            continue
        counts[ch] = counts.get(ch, 0) + 1
    return counts


def sanitize_tex_file_for_pdflatex(tex_path: str | Path) -> SanitizeReport:
    """Replace a small set of Unicode characters that commonly break pdflatex.

    This is intentionally narrow: it targets known problematic code points
    observed in generated manuscripts (e.g., U+202F, U+2248) without attempting
    a lossy full-Unicode-to-ASCII rewrite.
    """
    path = Path(tex_path)
    text = path.read_text(encoding="utf-8", errors="strict")

    replacements = _count_occurrences(text, UNICODE_REPLACEMENTS.keys())
    if not replacements:
        return SanitizeReport(
            changed=False,
            replacements={},
            remaining_non_ascii=_count_remaining_non_ascii(text),
        )

    new_text = text
    for ch, repl in UNICODE_REPLACEMENTS.items():
        if ch in replacements:
            new_text = new_text.replace(ch, repl)

    path.write_text(new_text, encoding="utf-8")
    return SanitizeReport(
        changed=True,
        replacements=replacements,
        remaining_non_ascii=_count_remaining_non_ascii(new_text),
    )


def ensure_tex_has_end_document(tex_path: str | Path) -> bool:
    """Ensure a LaTeX file contains a terminating \\end{document}.

    Some LLM-generated manuscripts accidentally omit the final terminator, which
    causes pdflatex to abort with 'no legal \\end found'. This fix is narrow and
    explicit: if we detect a \\begin{document} but no \\end{document}, we append
    it.
    """
    path = Path(tex_path)
    text = path.read_text(encoding="utf-8", errors="strict")
    if "\\end{document}" in text:
        return False
    if "\\begin{document}" not in text:
        raise ValueError("template.tex is missing \\\\begin{document}")
    suffix = "\n\\end{document}\n"
    if not text.endswith("\n"):
        suffix = "\n" + suffix
    path.write_text(text + suffix, encoding="utf-8")
    return True


def ensure_tex_uses_references_bibliography(tex_path: str | Path) -> bool:
    """Normalize bibliography database to 'references' when present.

    The ICLR template ships with 'iclr2025.bib' as an example database, but our
    pipeline collects citations into a generated references.bib. If the model
    outputs \\bibliography{iclr2025}, citations will show as '??' despite being
    present in references.bib.
    """
    path = Path(tex_path)
    text = path.read_text(encoding="utf-8", errors="strict")
    needle = "\\bibliography{iclr2025}"
    if needle not in text:
        return False
    text = text.replace(needle, "\\bibliography{references}")
    path.write_text(text, encoding="utf-8")
    return True


_BIBTEX_ESCAPE_RE = re.compile(r"(?<!\\)([_&])")
_BIBTEX_DOUBLE_ESCAPE_RE = re.compile(r"\\\\([_&])")
_BIBTEX_ESCAPE_MAP = {
    "_": r"\_",
    "&": r"\&",
}


def sanitize_bibtex_text_for_pdflatex(bibtex_text: str) -> tuple[str, Dict[str, int]]:
    """Escape a small set of LaTeX-special characters inside BibTeX entries.

    Semantic Scholar (and similar sources) sometimes return BibTeX with raw
    underscores like `RNN_Bert_Based` in titles. BibTeX will forward those
    tokens into the generated `.bbl`, which makes pdflatex fail with
    `Missing $ inserted` / `Extra }`.

    This sanitizer is intentionally narrow:
    - Escapes `_` and `&` when they are not already escaped.
    - Skips the entry header line that contains the citation key (starts with `@`)
      so citation keys remain stable for `\\cite{...}`.
    """

    counts: Dict[str, int] = {}
    out_lines: list[str] = []
    for raw_line in bibtex_text.splitlines(keepends=True):
        if raw_line.lstrip().startswith("@"):
            out_lines.append(raw_line)
            continue

        # Normalize legacy over-escaping like `\\_` -> `\_` inside bib fields.
        normalized_line = _BIBTEX_DOUBLE_ESCAPE_RE.sub(r"\\\1", raw_line)

        def _repl(match: re.Match[str]) -> str:
            ch = match.group(1)
            counts[ch] = counts.get(ch, 0) + 1
            return _BIBTEX_ESCAPE_MAP[ch]

        out_lines.append(_BIBTEX_ESCAPE_RE.sub(_repl, normalized_line))

    return "".join(out_lines), counts
