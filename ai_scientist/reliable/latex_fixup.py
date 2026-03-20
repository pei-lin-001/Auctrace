from __future__ import annotations

from dataclasses import dataclass

from .latex_scaffold import normalize_generated_latex_draft


@dataclass(frozen=True)
class LatexFixupReport:
    changed: bool
    scaffold_preserved: bool
    embedded_label_fixes: int
    bibliography_fixed: bool
    end_document_appended: bool


def fixup_latex_before_validation(
    latex_text: str,
    *,
    scaffold_tex: str,
) -> tuple[str, LatexFixupReport]:
    """Apply deterministic LaTeX normalization before gates/compilation.

    This is *not* a fallback that hides failures. It only performs safe,
    mechanical edits (scaffold preservation, minor syntax cleanup) and the
    pipeline still runs strict gates + strict compilation afterwards.
    """

    normalized, report = normalize_generated_latex_draft(
        latex_text,
        scaffold_tex=scaffold_tex,
    )
    fixed_bibliography = False
    out = normalized
    if "\\bibliography{iclr2025}" in out:
        out = out.replace("\\bibliography{iclr2025}", "\\bibliography{references}")
        fixed_bibliography = True

    appended_end_document = False
    if "\\begin{document}" in out and "\\end{document}" not in out:
        out = out.rstrip() + "\n\\end{document}\n"
        appended_end_document = True

    return out, LatexFixupReport(
        changed=bool(report.get("changed")) or fixed_bibliography or appended_end_document,
        scaffold_preserved=bool(report.get("scaffold_preserved")),
        embedded_label_fixes=int(report.get("embedded_label_fixes") or 0),
        bibliography_fixed=fixed_bibliography,
        end_document_appended=appended_end_document,
    )

