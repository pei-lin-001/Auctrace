from __future__ import annotations

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
