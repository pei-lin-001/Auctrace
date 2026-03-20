from __future__ import annotations

import re

from .errors import SymbolicLatexError

_FILECONTENTS_BEGIN = r"\begin{filecontents}{references.bib}"
_FILECONTENTS_END = r"\end{filecontents}"
_CLEANUP_MAP = (
    ("</end", r"\end"),
    ("</begin", r"\begin"),
    ("’", "'"),
    (r"\elliniwidth", r"\linewidth"),
)
_EMBEDDED_LABEL_PATTERNS = (
    re.compile(
        r"\\includegraphics(?P<opts>\[[^\]]*\])?\{(?P<prefix>%\s*)?\\label\{(?P<label>[^}]+)\}\s*(?P<target>[^{}]+?)\}",
        re.DOTALL,
    ),
    re.compile(
        r"\\includegraphics(?P<opts>\[[^\]]*\])?\{(?P<target>[^{}]+?)\s*\\label\{(?P<label>[^}]+)\}\}",
        re.DOTALL,
    ),
)
_HTMLISH_IMG_RE = re.compile(
    r"(?m)^(?P<indent>\s*)<img\s+width=(?P<width>[^\]]+)\]\{(?P<target>[^}]+)\}\s*$"
)


def normalize_generated_latex_draft(
    generated_tex: str,
    *,
    scaffold_tex: str,
) -> tuple[str, dict[str, int | bool]]:
    cleaned = generated_tex.strip()
    for bad_str, repl_str in _CLEANUP_MAP:
        cleaned = cleaned.replace(bad_str, repl_str)
    cleaned = re.sub(r"(\d+(?:\.\d+)?)%", r"\1\\%", cleaned)
    cleaned = _repair_htmlish_img_tags(cleaned)
    cleaned = _repair_missing_includegraphics_backslash(cleaned)
    cleaned, embedded_label_fixes = _repair_embedded_graphics_labels(cleaned)
    normalized = preserve_canonical_scaffold(cleaned, scaffold_tex=scaffold_tex)
    return normalized, {
        "changed": normalized != generated_tex,
        "scaffold_preserved": normalized != cleaned,
        "embedded_label_fixes": embedded_label_fixes,
    }


def preserve_canonical_scaffold(generated_tex: str, *, scaffold_tex: str) -> str:
    scaffold_prefix = _scaffold_prefix(scaffold_tex)
    generated_suffix = _editable_suffix(generated_tex)
    return scaffold_prefix + generated_suffix


def scaffold_prefix(tex: str) -> str:
    """Return the immutable LaTeX prefix up to \\end{filecontents}.

    This prefix contains the bibliography scaffold and package imports. In
    reliable mode, we treat it as immutable and only allow edits after it.
    """

    return _scaffold_prefix(tex)


def editable_suffix(tex: str) -> str:
    """Return the editable LaTeX suffix after \\end{filecontents}."""

    return _editable_suffix(tex)


def assemble_from_suffix(*, scaffold_tex: str, suffix_tex: str) -> str:
    """Assemble a full LaTeX document from a canonical scaffold + editable suffix."""

    prefix = _scaffold_prefix(scaffold_tex)
    suffix = suffix_tex
    if not suffix.strip():
        raise SymbolicLatexError(
            "Editable LaTeX suffix is empty after \\end{filecontents}."
        )
    if not suffix.startswith("\n"):
        suffix = "\n" + suffix
    return prefix + suffix


def _scaffold_prefix(tex: str) -> str:
    _, end = _filecontents_bounds(tex)
    return tex[:end]


def _editable_suffix(tex: str) -> str:
    _, end = _filecontents_bounds(tex)
    suffix = tex[end:]
    if not suffix.strip():
        raise SymbolicLatexError(
            "Generated LaTeX is missing manuscript content after \\end{filecontents}."
        )
    return suffix


def _filecontents_bounds(tex: str) -> tuple[int, int]:
    begin = tex.find(_FILECONTENTS_BEGIN)
    if begin < 0:
        raise SymbolicLatexError(
            "Generated LaTeX must preserve \\begin{filecontents}{references.bib}."
        )
    end = tex.find(_FILECONTENTS_END, begin)
    if end < 0:
        raise SymbolicLatexError(
            "Generated LaTeX must preserve \\end{filecontents} for the bibliography scaffold."
        )
    return begin, end + len(_FILECONTENTS_END)


def _repair_embedded_graphics_labels(tex: str) -> tuple[str, int]:
    repaired = tex
    total_replacements = 0
    for pattern in _EMBEDDED_LABEL_PATTERNS:
        repaired, replacements = pattern.subn(_graphics_label_replacement, repaired)
        total_replacements += replacements
    return repaired, total_replacements


def _repair_missing_includegraphics_backslash(tex: str) -> str:
    repaired = re.sub(
        r"(?m)^(?P<indent>\s*)includegraphics(?=\s*(?:\[|\{))",
        r"\g<indent>\\includegraphics",
        tex,
    )
    repaired = re.sub(
        r"(?m)(\\centering\s+)includegraphics(?=\s*(?:\[|\{))",
        r"\1\\includegraphics",
        repaired,
    )
    return repaired


def _repair_htmlish_img_tags(tex: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        indent = match.group("indent") or ""
        width = match.group("width").strip()
        target = match.group("target").strip()
        return f"{indent}\\includegraphics[width={width}]{{{target}}}"

    return _HTMLISH_IMG_RE.sub(_repl, tex)


def _graphics_label_replacement(match: re.Match[str]) -> str:
    label = match.group("label").strip()
    opts = match.group("opts") or ""
    target = _clean_graphics_target(match.group("target"))
    return f"\\label{{{label}}}\n\\includegraphics{opts}{{{target}}}"


def _clean_graphics_target(raw_target: str) -> str:
    lines = []
    for line in raw_target.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        lines.append(stripped)
    target = "".join(lines).strip()
    if not target:
        raise SymbolicLatexError(
            "Malformed \\includegraphics argument lost its figure target during normalization."
        )
    return target
