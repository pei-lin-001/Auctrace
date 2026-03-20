from __future__ import annotations


def build_writeup_mode_instructions(symbolic_facts: bool) -> str:
    if not symbolic_facts:
        return (
            "NON-SYMBOLIC MODE:\n"
            "- Do NOT use any \\\\fact{...} or \\\\param{...} placeholders.\n"
            "- Write numeric results directly in the LaTeX.\n"
        )
    return (
        "FACT VARIABLES MODE (symbolic writeup):\n"
        "- Do NOT write any numeric literals directly (including experimental values, literature stats, or rhetorical percentages).\n"
        "  Only LaTeX layout dimensions like 0.5\\textwidth are allowed outside placeholders.\n"
        "- Use \\\\fact{KEY} only for experimental values from writeup_context_pack.facts_index.\n"
        "- Use \\\\param{KEY} only for method/setup/constants from writeup_context_pack.params_index.\n"
        "- Prefer exact keys already surfaced in writeup_context_pack.preferred_fact_refs and WRITEUP_SYMBOLIC_SUMMARY.*.fact_refs before scanning the full facts_index.\n"
        "- Keep placeholders in the LaTeX; a renderer will fill values before compilation.\n"
        "- Use only exact keys listed in the provided facts_index and params_index.\n"
        "- Never invent, rename, or change stage/dataset suffixes in a key.\n"
        "- If no exact fact or param key exists, rewrite that sentence qualitatively.\n"
        "- The manuscript must still contain at least one valid \\\\fact{KEY} placeholder, and it should appear in the main text rather than only in comments.\n"
        "- Do not write approximate relative reductions or raw percentage ranges such as 'over 30\\\\%' or '15-25\\\\%'.\n"
        "- When including figures, use only exact strings from writeup_context_pack.artifact_manifest_summary.includegraphics_targets.\n"
        "- Do NOT reference stale experiment_results/*.png paths from older drafts or prior runs.\n"
        "- Keep the LaTeX scaffold before \\\\end{filecontents} structurally unchanged: package imports, theorem declarations, \\\\graphicspath, and the references.bib scaffold are fixed.\n"
    )


def build_reflection_guard_block(
    *,
    symbolic_facts: bool,
    remediation_instructions: str,
) -> str:
    mode_rules = build_writeup_mode_instructions(symbolic_facts)
    return (
        f"{mode_rules}\n"
        f"{remediation_instructions}\n"
        "Do not repeat the previous invalid draft. Keep all figure paths and symbolic placeholders valid.\n"
        "Do not rewrite the static LaTeX scaffold before \\end{filecontents}; only revise the manuscript content that follows it.\n"
    )
