import argparse
import json
import os
import os.path as osp
import re
import shutil
import subprocess
import traceback
import unicodedata
import uuid
import tempfile
from typing import Any

from ai_scientist.llm import (
    get_response_from_llm,
    extract_json_between_markers,
    create_client,
)

from ai_scientist.utils.token_tracker import track_token_usage

from ai_scientist.tools.literature_search import search_for_papers

from ai_scientist.perform_vlm_review import (
    generate_vlm_img_review,
    perform_imgs_cap_ref_review,
    perform_imgs_cap_ref_review_selection,
    detect_duplicate_figures,
)
from ai_scientist.vlm import create_client as create_vlm_client
from ai_scientist.latex_sanitize import (
    ensure_tex_has_end_document,
    ensure_tex_uses_references_bibliography,
    sanitize_bibtex_text_for_pdflatex,
    sanitize_tex_file_for_pdflatex,
)

from ai_scientist.reliable.facts import facts_index_for_prompt
from ai_scientist.reliable.latex_compile import (
    compile_symbolic_latex_project,
    ensure_latex_template_assets,
)
from ai_scientist.reliable.latex_scaffold import editable_suffix, normalize_generated_latex_draft
from ai_scientist.reliable.latex_fixup import fixup_latex_before_validation
from ai_scientist.reliable.latex_patch import apply_latex_patch_ops
from ai_scientist.reliable.params import (
    ensure_param_store_for_run,
    params_index_for_prompt,
)
from ai_scientist.reliable.writeup_inputs import ensure_symbolic_writeup_inputs
from ai_scientist.reliable.writeup_validation import validate_generated_writeup
from ai_scientist.reliable.artifact_manifest import (
    artifact_manifest_summary,
    ensure_artifact_manifest,
)
from ai_scientist.reliable.context_packs import (
    build_claim_context_pack,
    build_writeup_context_pack,
    save_context_pack,
)
from ai_scientist.reliable.remediation import (
    build_remediation_prompt_block,
    classify_remediation_failure,
    print_remediation_report,
    save_remediation_report,
    remediation_retry_target,
    should_reuse_symbolic_writeup_artifacts,
)
from ai_scientist.reliable.errors import SymbolicLatexError
from ai_scientist.reliable.writeup_retry import (
    has_validated_writeup,
    restore_validated_writeup,
    save_validated_writeup,
)
from ai_scientist.reliable.writeup_rules import (
    build_reflection_guard_block,
    build_writeup_mode_instructions,
)
from ai_scientist.reliable.claim_ledger import (
    generate_claim_ledger,
    save_claim_ledger,
    validate_claim_ledger,
)
from ai_scientist.reliable.claim_traceability import (
    build_claim_trace_index,
    save_claim_trace_index,
    validate_used_fact_traceability,
)
from ai_scientist.reliable.outsider_audit import (
    build_outsider_audit_context_pack,
    build_outsider_audit_inputs,
    generate_outsider_audit,
    save_outsider_audit,
    validate_outsider_audit,
)
from ai_scientist.env_utils import env_bool, env_str


def remove_accents_and_clean(s):
    # Normalize to separate accents
    nfkd_form = unicodedata.normalize("NFKD", s)
    # Remove non-ASCII characters
    ascii_str = nfkd_form.encode("ASCII", "ignore").decode("ascii")
    # Remove anything but letters, digits, underscores, colons, dashes, @, {, }, and commas
    ascii_str = re.sub(r"[^a-zA-Z0-9:_@\{\},-]+", "", ascii_str)
    # Convert to lowercase
    ascii_str = ascii_str.lower()
    return ascii_str


def _extract_latex_from_response(text: str) -> tuple[str | None, str]:
    patterns: list[tuple[str, str]] = [
        ("```latex```", r"```latex(.*?)```"),
        ("```tex```", r"```tex(.*?)```"),
        ("```fence```", r"```(.*?)```"),
    ]
    for label, pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            if "\\documentclass" in extracted and "\\end{document}" not in extracted:
                extracted = extracted + "\n\\end{document}\n"
                return extracted, label + " + appended \\end{document}"
            return extracted, label

    # Tolerate unterminated fences (common in streaming / partial responses).
    unterminated_patterns: list[tuple[str, str]] = [
        ("```latex (unterminated)```", r"```latex(.*)$"),
        ("```tex (unterminated)```", r"```tex(.*)$"),
        ("```fence (unterminated)```", r"```(.*)$"),
    ]
    for label, pattern in unterminated_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            if "\\documentclass" in extracted and "\\end{document}" not in extracted:
                extracted = extracted + "\n\\end{document}\n"
                return extracted, label + " + appended \\end{document}"
            return extracted, label

    doc_start = text.find("\\documentclass")
    doc_end = text.rfind("\\end{document}")
    if doc_start >= 0 and doc_end >= 0 and doc_end > doc_start:
        return text[doc_start : doc_end + len("\\end{document}")].strip(), "documentclass"
    if doc_start >= 0 and doc_end < 0:
        extracted = text[doc_start:].strip()
        extracted = extracted + "\n\\end{document}\n"
        return extracted, "documentclass (unterminated) + appended \\end{document}"
    return None, "none"


def _extract_patch_from_response(text: str) -> dict[str, Any] | None:
    json_output = extract_json_between_markers(text)
    if not isinstance(json_output, dict):
        return None
    return json_output


def _maybe_backanchor_numeric_literals(
    latex_text: str,
    *,
    symbolic_facts: bool,
    store,
    param_store,
) -> str:
    if not symbolic_facts:
        return latex_text
    if store is None or param_store is None:
        return latex_text
    if not env_bool("AI_SCIENTIST_NUMERIC_BACKANCHOR", False):
        return latex_text

    from ai_scientist.reliable.numeric_backanchor import backanchor_numeric_literals

    result = backanchor_numeric_literals(
        latex_text,
        store=store,
        param_store=param_store,
    )
    if result.replacements:
        summary = ", ".join(f"{k!r}x{v}" for k, v in sorted(result.replacements.items()))
        print(f"[writeup] auto-anchored numeric literals to placeholders: {summary}")
    return result.latex_text


def _sanitize_template_tex_for_pdflatex(cwd: str) -> None:
    try:
        tex_path = osp.join(cwd, "template.tex")
        if ensure_tex_has_end_document(tex_path):
            print("[latex] appended missing \\end{document} to template.tex")
        if ensure_tex_uses_references_bibliography(tex_path):
            print("[latex] normalized bibliography database to references.bib")
        references_bib_path = osp.join(cwd, "references.bib")
        if osp.exists(references_bib_path):
            os.remove(references_bib_path)
            print("[latex] removed stale references.bib to force regeneration from filecontents")
        report = sanitize_tex_file_for_pdflatex(tex_path)
        if not report.changed:
            return
        changed_keys = [
            f"U+{ord(ch):04X}x{count}" for ch, count in report.replacements.items()
        ]
        print(
            "[latex] sanitized template.tex for pdflatex unicode compatibility: "
            + ", ".join(changed_keys)
        )
        if report.remaining_non_ascii:
            remaining = ", ".join(
                sorted({f"U+{ord(ch):04X}" for ch in report.remaining_non_ascii})
            )
            print(f"[latex] warning: template.tex still contains non-ascii: {remaining}")
    except Exception:
        print("EXCEPTION in compile_latex while sanitizing template.tex:")
        print(traceback.format_exc())


def _run_latex_command(command: list[str], cwd: str, timeout: int) -> None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            print(
                f"EXCEPTION in compile_latex: command failed (returncode={result.returncode}): "
                f"{' '.join(command)}"
            )
        print("Standard Output:\n", result.stdout)
        print("Standard Error:\n", result.stderr)
    except subprocess.TimeoutExpired:
        print(f"EXCEPTION in compile_latex: LaTeX timed out after {timeout} seconds.")
        print(traceback.format_exc())
    except subprocess.CalledProcessError:
        print(
            f"EXCEPTION in compile_latex: Error running command {' '.join(command)}"
        )
        print(traceback.format_exc())


def compile_latex(cwd, pdf_file, timeout=30):
    print("GENERATING LATEX")
    ensure_latex_template_assets(cwd)
    _sanitize_template_tex_for_pdflatex(cwd)

    commands = [
        ["pdflatex", "-interaction=nonstopmode", "template.tex"],
        ["bibtex", "template"],
        ["pdflatex", "-interaction=nonstopmode", "template.tex"],
        ["pdflatex", "-interaction=nonstopmode", "template.tex"],
    ]

    for command in commands:
        _run_latex_command(command, cwd=cwd, timeout=timeout)

    print("FINISHED GENERATING LATEX")

    try:
        shutil.move(osp.join(cwd, "template.pdf"), pdf_file)
    except FileNotFoundError:
        print("Failed to rename PDF.")
        print("EXCEPTION in compile_latex while moving PDF:")
        print(traceback.format_exc())


def is_header_or_footer(line):
    """
    Returns True if the line is likely a header or footer.
    Filters out:
      - Lines that are too short (< 4 characters after stripping).
      - Lines that are only digits.
      - Lines starting with known phrases (e.g., "Under review").
      - Lines that consist solely of capital letters and spaces.
    """
    line_stripped = line.strip()
    if len(line_stripped) < 1:
        return True

    header_footer_patterns = [
        r"^\d+$",  # Only digits (e.g., page numbers like "000", "001", etc.)
        r"^Under review",  # Lines starting with "Under review"
    ]
    for pattern in header_footer_patterns:
        if re.match(pattern, line_stripped):
            return True
    return False


def clean_lines(content):
    """
    Given raw text content, split it into lines and remove lines that are
    likely headers/footers or otherwise not part of the main content.
    """
    lines = content.splitlines()
    # Keep only lines that are not detected as headers/footers.
    return [line for line in lines if not is_header_or_footer(line)]


def detect_references_position_clean(pdf_file):
    """
    Locate the first occurrence of the word "References" (or variations like
    "R EFERENCES") within the cleaned content extracted from the PDF.
    Uses pdftotext with layout preservation and cleans the extracted text.

    Returns a tuple (ref_page, ref_line) if found (with ref_line counting only
    the cleaned lines), otherwise None.
    """
    if not osp.exists(pdf_file):
        return None

    # Compile a regex pattern to match "REFERENCES" even if there are extra spaces
    # between letters (and do a case-insensitive match).
    pattern = re.compile(r"\bR\s*E\s*F\s*E\s*R\s*E\s*N\s*C\s*E\s*S\b", re.IGNORECASE)

    # Loop through pages (limit to 50 pages by default)
    for page in range(1, 51):
        temp_dir = tempfile.mkdtemp()
        page_txt = osp.join(temp_dir, f"page_{page}.txt")
        try:
            subprocess.run(
                [
                    "pdftotext",
                    "-layout",
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                    "-q",
                    pdf_file,
                    page_txt,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if not osp.exists(page_txt):
                shutil.rmtree(temp_dir)
                break
            try:
                with open(page_txt, "r", encoding="utf-8", errors="ignore") as fp:
                    content = fp.read()
            except Exception as e:
                print(f"Error reading page {page}: {e}")
                print(traceback.format_exc())
                shutil.rmtree(temp_dir)
                continue
            finally:
                shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Error running pdftotext for page {page}: {e}")
            print(traceback.format_exc())
            shutil.rmtree(temp_dir)
            continue

        # Clean the lines before searching for "References"
        cleaned = clean_lines(content)
        for idx, line in enumerate(cleaned):
            if pattern.search(line):
                # Found "References" on this page at cleaned line number idx+1
                return (page, idx + 1)
    return None


def extract_page_line_counts(pdf_file, first_page, last_page):
    """
    Extract the number of cleaned text lines for each page from first_page to last_page.
    This uses pdftotext with layout preservation and the clean_lines helper.
    Returns a dictionary {page_number: number_of_cleaned_lines}.
    Pages for which extraction fails are omitted.
    """
    page_lines = {}
    for page in range(first_page, last_page + 1):
        temp_dir = tempfile.mkdtemp()
        page_txt = osp.join(temp_dir, f"page_{page}.txt")
        try:
            subprocess.run(
                [
                    "pdftotext",
                    "-layout",
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                    "-q",
                    pdf_file,
                    page_txt,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if not osp.exists(page_txt):
                shutil.rmtree(temp_dir)
                break
            try:
                with open(page_txt, "r", encoding="utf-8", errors="ignore") as fp:
                    content = fp.read()
            except Exception as e:
                print(f"Error reading page {page}: {e}")
                print(traceback.format_exc())
                shutil.rmtree(temp_dir)
                continue
            finally:
                shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Error running pdftotext for page {page}: {e}")
            print(traceback.format_exc())
            shutil.rmtree(temp_dir)
            continue
        # Clean the extracted text and count the number of remaining lines.
        cleaned = clean_lines(content)
        page_lines[page] = len(cleaned)
    return page_lines


def check_page_limit(pdf_file, page_limit=4, timeout=30):
    """
    Compile the LaTeX project in a temporary folder, then determine where the
    "References" section begins using cleaned text extraction. Next, count the
    number of cleaned text lines used before the word "References" and compare that
    to the total number of cleaned lines available in the allowed number of pages (page_limit).

    Returns a dictionary with:
      - 'ref_page': page number where "References" was found (or None)
      - 'ref_line': cleaned line number within that page (or None)
      - 'used_lines': number of cleaned lines used for main content (before "References")
      - 'allowed_lines': total number of cleaned text lines available in pages 1..page_limit
      - 'excess': if used_lines > allowed_lines (number of lines over the limit),
      - 'available': if used_lines < allowed_lines (number of lines still available)

    If compilation or extraction fails, returns None.
    """
    try:
        # Ensure the PDF was produced
        if not osp.exists(pdf_file):
            return None

        # Locate the first occurrence of "References" using the cleaned extraction
        ref_pos = detect_references_position_clean(pdf_file)
        if ref_pos is None:
            # If "References" isn't found, assume no reference section exists.
            return None
        ref_page, ref_line = ref_pos

        # Determine up to which page we need to extract cleaned line counts:
        max_page_to_extract = max(page_limit, ref_page)
        page_line_counts = extract_page_line_counts(pdf_file, 1, max_page_to_extract)
        if not page_line_counts:
            return None

        # Compute total cleaned lines available in the allowed pages (pages 1 to page_limit)
        allowed_lines = sum(
            page_line_counts.get(page, 0) for page in range(1, page_limit + 1)
        )

        # Compute cleaned lines used before "References":
        used_lines = 0
        # Sum full pages before the reference page
        for page in range(1, ref_page):
            used_lines += page_line_counts.get(page, 0)
        # Add lines from the reference page up to (but not including) the line where "References" appears
        used_lines += ref_line - 1

        result = {
            "ref_page": ref_page,
            "ref_line": ref_line,
            "used_lines": used_lines,
            "allowed_lines": allowed_lines,
        }
        if used_lines > allowed_lines:
            result["excess"] = used_lines - allowed_lines
        else:
            result["available"] = allowed_lines - used_lines
        return result

    except Exception as e:
        print(f"Error checking page limit: {e}")
        print(traceback.format_exc())
        return None


def get_reflection_page_info(reflection_pdf, page_limit):
    info = check_page_limit(reflection_pdf, page_limit)
    if info is not None:
        if "excess" in info:
            reflection_page_info = (
                f"\nCurrently, 'References' begins on page {info['ref_page']}, approximately on line {info['ref_line']}. "
                f"The main text (before the references) uses {info['used_lines']} lines, which exceeds the allowed {info['allowed_lines']} lines for a {page_limit}-page limit by {info['excess']} lines. "
                f"DO NOT USE MORE THAN {page_limit} PAGES FOR THE MAIN TEXT. Please reduce the text or resize the plot to meet the page limit. "
                f"Consider grouping plots together to make the paper more concise. "
                f"Papers often look more professional if the main text is just under {page_limit} pages in length.\n"
            )
        elif "available" in info:
            reflection_page_info = (
                f"\nCurrently, 'References' begins on page {info['ref_page']}, approximately on line {info['ref_line']}. "
                f"The main text (before the references) uses {info['used_lines']} lines, leaving {info['available']} lines available out of the allowed {info['allowed_lines']} lines (which corresponds to {page_limit} pages). "
                f"DO NOT USE MORE THAN {page_limit} PAGES FOR THE MAIN TEXT. You can add up to {info['available']} lines if needed, "
                f"but papers often look more professional if the main text is just under {page_limit} pages in length.\n"
            )
        else:
            # Fallback in case the info dictionary doesn't contain 'excess' or 'available'
            reflection_page_info = (
                f"\nCurrently, 'References' begins on page {info.get('ref_page', '?')}, approximately on line {info.get('ref_line', '?')}. "
                f"The page limit is {page_limit} pages for the main text before the references. "
                f"DO NOT USE MORE THAN {page_limit} PAGES FOR THE MAIN TEXT. Adjust your content accordingly.\n"
            )
    else:
        reflection_page_info = (
            "\nCould not detect 'References' page (compilation or detection failed).\n"
        )

    return reflection_page_info


def get_citation_addition(
    client, model, context, current_round, total_rounds, idea_text
):
    report, citations = context
    msg_history = []
    citation_system_msg_template = """You are an ambitious AI researcher who is looking to publish a paper to a workshop at ICLR 2025 that explores real-world pitfalls, failures, and challenges in deep learning.
You have already completed the experiments and now you are looking to collect citations to related papers.
This phase focuses on collecting references and annotating them to be integrated later.
Collected citations will be added to a references.bib file.

Reasons to reference papers include:
1. Summarizing Research: Cite sources when summarizing the existing literature.
2. Using Specific Concepts: Provide citations when discussing specific theories or concepts.
3. Datasets, models, and optimizers: Cite the creators of datasets, models, and optimizers.
4. Comparing Findings: Cite relevant studies when comparing or contrasting different findings.
5. Highlighting Research Gaps: Cite previous research when pointing out gaps your study addresses.
6. Using Established Methods: Cite the creators of methodologies you employ.
7. Supporting Arguments: Cite sources that back up your conclusions and arguments.
8. Suggesting Future Research: Reference studies related to proposed future research directions.

Ensure sufficient cites will be collected for all of these categories, and no categories are missed.
You will be given access to the Semantic Scholar API; only add citations that you have found using the API.
Aim to discuss a broad range of relevant papers, not just the most popular ones.
Make sure not to copy verbatim from prior literature to avoid plagiarism.
You will have {total_rounds} rounds to add to the references but do not need to use them all.

DO NOT ADD A CITATION THAT ALREADY EXISTS!"""

    citation_first_prompt_template = """Round {current_round}/{total_rounds}:

You planned and executed the following idea:
```markdown
{Idea}
```

You produced the following report:
```markdown
{report}
```

Your current list of citations is:
```
{citations}
```

Identify the most important citation that you still need to add, and the query to find the paper.

Respond in the following format:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, first briefly reason and identify which citations are missing.
If no more citations are needed, add "No more citations needed" to your thoughts.
Do not add "No more citations needed" if you are adding citations this round.

In <JSON>, respond in JSON format with the following fields:
- "Description": The purpose of the desired citation and a brief description of what you are looking for.
- "Query": The search query to find the paper (e.g., attention is all you need).
This JSON will be automatically parsed, so ensure the format is precise."""

    citation_second_prompt_template = """Search has recovered the following articles:

{papers}

Respond in the following format:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, briefly reason over the search results and identify which citation(s) best fit your paper.
If none are appropriate or would contribute significantly to the write-up, add "Do not add any" to your thoughts.
Do not select papers that are already in the `references.bib` file, or if the same citation exists under a different name.

In <JSON>, respond in JSON format with the following fields:
- "Selected": A list of integer indices for the selected papers, for example [0, 1]. Do not use quotes for the indices, e.g. "['0', '1']" is invalid.
- "Description": Update the previous description of the citation(s) with the additional context. This should be a brief description of the work(s), their relevance, and where in a paper these should be cited.
This JSON will be automatically parsed, so ensure the format is precise."""

    try:
        text, msg_history = get_response_from_llm(
            prompt=citation_first_prompt_template.format(
                current_round=current_round + 1,
                total_rounds=total_rounds,
                Idea=idea_text,
                report=report,
                citations=citations,
            ),
            client=client,
            model=model,
            system_message=citation_system_msg_template.format(
                total_rounds=total_rounds
            ),
            msg_history=msg_history,
            print_debug=False,
        )
        if "No more citations needed" in text:
            print("No more citations needed.")
            return None, True

        json_output = extract_json_between_markers(text)
        assert json_output is not None, "Failed to extract JSON from LLM output"
        query = json_output["Query"]
        papers = search_for_papers(query, result_limit=5)
    except Exception:
        print("EXCEPTION in get_citation_addition (initial search):")
        print(traceback.format_exc())
        return None, False

    if papers is None:
        print("No papers found.")
        return None, False

    paper_strings = []
    for i, paper in enumerate(papers):
        paper_strings.append(
            "{i}: {title}. {authors}. {venue}, {year}.\nAbstract: {abstract}".format(
                i=i,
                title=paper["title"],
                authors=paper["authors"],
                venue=paper["venue"],
                year=paper["year"],
                abstract=paper["abstract"],
            )
        )
    papers_str = "\n\n".join(paper_strings)

    try:
        text, msg_history = get_response_from_llm(
            prompt=citation_second_prompt_template.format(
                papers=papers_str,
                current_round=current_round + 1,
                total_rounds=total_rounds,
            ),
            client=client,
            model=model,
            system_message=citation_system_msg_template.format(
                total_rounds=total_rounds
            ),
            msg_history=msg_history,
            print_debug=False,
        )
        if "Do not add any" in text:
            print("Do not add any.")
            return None, False

        json_output = extract_json_between_markers(text)
        assert json_output is not None, "Failed to extract JSON from LLM output"
        desc_raw = json_output.get("Description", json_output.get("description", ""))
        desc = str(desc_raw).strip()
        selected_raw = json_output.get("Selected", json_output.get("selected"))
        if selected_raw is None:
            print("[citations] selection JSON missing 'Selected'; skipping.")
            return None, False
        if isinstance(selected_raw, list):
            selected_indices = [int(x) for x in selected_raw]
        else:
            selected_indices = [
                int(x) for x in re.findall(r"-?\d+", str(selected_raw))
            ]

        if not selected_indices:
            return None, False

        assert all([0 <= i < len(papers) for i in selected_indices]), "Invalid paper index"
        bibtexs = [papers[i]["citationStyles"]["bibtex"] for i in selected_indices]

        cleaned_bibtexs = []
        for bibtex in bibtexs:
            newline_index = bibtex.find("\n")
            cite_key_line = bibtex[:newline_index]
            cite_key_line = remove_accents_and_clean(cite_key_line)
            cleaned_bibtexs.append(cite_key_line + bibtex[newline_index:])
        bibtexs = cleaned_bibtexs

        bibtex_string = "\n".join(bibtexs)

    except Exception:
        print("EXCEPTION in get_citation_addition (selecting papers):")
        print(traceback.format_exc())
        return None, False

    references_format = """% {description}
{bibtex}"""

    references_prompt = references_format.format(bibtex=bibtex_string, description=desc)
    return references_prompt, False


writeup_system_message_template = """You are an ambitious AI researcher who is looking to publish a paper to the "I Can't Believe It's Not Better" (ICBINB) Workshop at ICLR 2025.
This workshop aims to highlight real-world pitfalls, challenges, and negative or inconclusive results in deep learning, encouraging open discussion.
You must accurately represent the results of the experiments.
The main paper is limited to {page_limit} pages in single-column format, not counting references. In general, try to use the available space and include all relevant information.
DO NOT USE MORE THAN {page_limit} PAGES FOR THE MAIN TEXT.
MINIMIZE THE USAGE OF ITEMIZE OR ENUMERATE. ONLY USE THEM IF THEY ARE ABSOLUTELY NECESSARY AND CONTAIN SUBSTANTIAL INFORMATION.
Ensure that the tables and figures are correctly placed in a reasonable location and format.

- Do not change the overall style which is mandated by the conference. Keep to the current method of including the references.bib file.
- Do not remove the \\graphicspath directive or no figures will be found.
- Do not add `Acknowledgements` section to the paper.

Here are some tips for each section of the paper:

- **Title**:
  - Title should be catchy and informative. It should give a good idea of what the paper is about.
  - Try to keep it under 2 lines.

- **Abstract**:
  - Brief summary highlighting the nature of the challenge or pitfall explored.
  - Concise motivation of why this matters for real-world deployment.
  - This should be one continuous paragraph.

- **Introduction**:
  - Overview of the issue or challenge being explored.
  - Clearly state why this problem is important, especially for practical or real-world contexts.
  - Summarize your contributions or findings: they may include negative results, real-world pitfalls, unexpected behaviors, or partial improvements.

- **Related Work**:
  - Cite relevant papers or approaches that have tackled similar issues or have encountered similar pitfalls.
  - Compare and contrast with your own findings.

- **Background** (optional):
  - Provide necessary technical or domain-specific background if needed.

- **Method / Problem Discussion**:
  - Detail the problem context or the method if it is relevant to highlight the challenges faced.
  - If results are not strictly an improvement, discuss partial successes or lessons learned.

- **Experiments** (if applicable):
  - Present results truthfully according to the data you have. Negative, unexpected, or inconclusive findings are valid contributions for this workshop.
  - Include figures, tables, or real-world examples that illustrate the pitfalls.
  - Include up to 4 figures in the main text. All other figures should be in the appendix.

- **Conclusion**:
  - Summarize the main lessons learned or contributions.
  - Suggest next steps or future directions, highlighting how these insights can help the community avoid or overcome similar issues.

- **Appendix**:
  - Place for supplementary material that did not fit in the main paper.
  - Add more information and details (hyperparameters, algorithms, etc.) in the supplementary material.
  - Add more plots and tables in the supplementary material. Make sure that this information is not already covered in the main paper.
  - When checking for duplicate figures, be sure to also review their descriptions to catch cases where different figures convey the same information. For example, one figure might present aggregated training accuracy as a single line plot with a shaded standard deviation (e.g., aggregated_training_accuracy.png), while another (per_seed_training_accuracy.png) shows the same data as three separate line plots.

Ensure you are always writing good compilable LaTeX code. Common mistakes that should be fixed include:
- LaTeX syntax errors (unenclosed math, unmatched braces, etc.).
- Duplicate figure labels or references.
- Unescaped special characters: & % $ # _ {{ }} ~ ^ \\
- Proper table/figure closure.
- Do not hallucinate new citations or any results not in the logs.

Ensure proper citation usage:
- Always include references within \\begin{{filecontents}}{{references.bib}} ... \\end{{filecontents}}, even if they haven't changed from the previous round.
- Use citations from the provided references.bib content.
- Each section (especially Related Work) should have multiple citations.

When returning final code, place it in fenced triple backticks with 'latex' syntax highlighting.
"""

writeup_prompt = """{facts_instructions}
{facts_priority_block}
{remediation_instructions}

Your goal is to write up the following idea:

```markdown
{idea_text}
```

We have the following canonical writeup context pack (JSON):
```json
{writeup_context_pack}
```

{plot_generation_notes}

Produce the final version of the LaTeX manuscript now, ensuring the paper is coherent, concise, and reports results accurately.
Return the entire file in full.
This must be an acceptable complete LaTeX writeup, suitable for a 4-page single-column workshop paper.
Make sure to use the citations from the references.bib file.

Please provide the updated LaTeX code for 'template.tex', wrapped in triple backticks
with "latex" syntax highlighting, like so:

```latex
<UPDATED LATEX CODE>
```
"""


def load_idea_text(base_folder):
    """
    Load the idea text from the base folder.
    """
    idea_text = ""
    research_idea_path = osp.join(base_folder, "research_idea.md")
    if osp.exists(research_idea_path):
        with open(research_idea_path, "r") as f_idea:
            idea_text = f_idea.read()
    else:
        idea_md_path = osp.join(base_folder, "idea.md")
        if osp.exists(idea_md_path):
            with open(idea_md_path, "r") as f_idea:
                idea_text = f_idea.read()
    return idea_text


def load_exp_summaries(base_folder):
    """
    Load the experiment summaries from the base folder.
    """
    summary_files = [
        ("logs/0-run/baseline_summary.json", "BASELINE_SUMMARY"),
        ("logs/0-run/research_summary.json", "RESEARCH_SUMMARY"),
        ("logs/0-run/ablation_summary.json", "ABLATION_SUMMARY"),
    ]
    loaded_summaries = {}
    for fname, key in summary_files:
        path = osp.join(base_folder, fname)
        if osp.exists(path):
            try:
                with open(path, "r") as f:
                    loaded_summaries[key] = json.load(f)
            except json.JSONDecodeError:
                print(
                    f"Warning: {fname} is not valid JSON. Using empty data for {key}."
                )
                loaded_summaries[key] = {}
        else:
            loaded_summaries[key] = {}
    return loaded_summaries


def filter_experiment_summaries(exp_summaries, step_name):
    if step_name == "citation_gathering":
        node_keys_to_keep = {
            "overall_plan",
            "analysis",
            "metric",
            "vlm_feedback_summary",
        }
    elif step_name == "writeup":
        node_keys_to_keep = {
            "overall_plan",
            "analysis",
            "metric",
            "code",
            "plot_analyses",
            "vlm_feedback_summary",
        }
    elif step_name == "plot_aggregation":
        node_keys_to_keep = {
            "overall_plan",
            "analysis",
            "plot_plan",
            "plot_code",
            "plot_analyses",
            "vlm_feedback_summary",
            "exp_results_npy_files",
        }
    else:
        raise ValueError(f"Invalid step name: {step_name}")

    filtered_summaries = {}
    for stage_name in exp_summaries.keys():
        if stage_name in {"BASELINE_SUMMARY", "RESEARCH_SUMMARY"}:
            filtered_summaries[stage_name] = {}
            for key in exp_summaries[stage_name].keys():
                if key in {"best node"}:
                    filtered_summaries[stage_name][key] = {}
                    for node_key in exp_summaries[stage_name][key].keys():
                        if node_key in node_keys_to_keep:
                            filtered_summaries[stage_name][key][node_key] = (
                                exp_summaries[stage_name][key][node_key]
                            )
        elif stage_name == "ABLATION_SUMMARY" and step_name == "plot_aggregation":
            filtered_summaries[stage_name] = {}
            for ablation_summary in exp_summaries[stage_name]:
                filtered_summaries[stage_name][ablation_summary["ablation_name"]] = {}
                for node_key in ablation_summary.keys():
                    if node_key in node_keys_to_keep:
                        filtered_summaries[stage_name][
                            ablation_summary["ablation_name"]
                        ][node_key] = ablation_summary[node_key]
    return filtered_summaries


def gather_citations(base_folder, num_cite_rounds=20, small_model="deepseek-chat"):
    """
    Gather citations for a paper, with ability to resume from previous progress.

    Args:
        base_folder: Path to project folder
        num_cite_rounds: Maximum number of citation gathering rounds
        small_model: Model to use for citation collection
        resume: Whether to try to resume from previous progress

    Returns:
        str: The gathered citations text, or None if failed
    """

    # Paths for storing progress
    citations_cache_path = osp.join(base_folder, "cached_citations.bib")
    progress_path = osp.join(base_folder, "citations_progress.json")

    # Initialize or load progress
    current_round = 0
    citations_text = ""

    if osp.exists(citations_cache_path) and osp.exists(progress_path):
        try:
            with open(citations_cache_path, "r") as f:
                citations_text = f.read()
            citations_text, bibtex_replacements = sanitize_bibtex_text_for_pdflatex(citations_text)
            if bibtex_replacements:
                print(
                    "[citations] sanitized cached citations for pdflatex: "
                    + ", ".join(f"{k}x{v}" for k, v in sorted(bibtex_replacements.items()))
                )
                with open(citations_cache_path, "w") as f:
                    f.write(citations_text)
            with open(progress_path, "r") as f:
                progress = json.load(f)
                current_round = progress.get("completed_rounds", 0)
            print(f"Resuming citation gathering from round {current_round}")
        except Exception as e:
            print(f"Error loading cached citations: {e}")
            print("Starting fresh")
            current_round = 0
            citations_text = ""

    try:
        # Load idea text and summaries
        idea_text = load_idea_text(base_folder)
        exp_summaries = load_exp_summaries(base_folder)
        filtered_summaries = filter_experiment_summaries(
            exp_summaries, step_name="citation_gathering"
        )
        filtered_summaries_str = json.dumps(filtered_summaries, indent=2)

        # Run small model for citation additions
        client, client_model = create_client(small_model)

        for round_idx in range(current_round, num_cite_rounds):
            try:
                print(f"[citations] round {round_idx + 1}/{num_cite_rounds}")
                context_for_citation = (filtered_summaries_str, citations_text)
                addition, done = get_citation_addition(
                    client,
                    client_model,
                    context_for_citation,
                    round_idx,
                    num_cite_rounds,
                    idea_text,
                )

                if done:
                    # Save final state before exiting
                    with open(citations_cache_path, "w") as f:
                        f.write(citations_text)
                    with open(progress_path, "w") as f:
                        json.dump(
                            {"completed_rounds": round_idx + 1, "status": "completed"},
                            f,
                        )
                    break

                if addition is not None:
                    addition, _ = sanitize_bibtex_text_for_pdflatex(addition)
                    # Simple check to avoid duplicating the same title
                    title_match = re.search(r" title = {(.*?)}", addition)
                    if title_match:
                        new_title = title_match.group(1).lower()
                        existing_titles = re.findall(
                            r" title = {(.*?)}", citations_text
                        )
                        existing_titles = [t.lower() for t in existing_titles]
                        if new_title not in existing_titles:
                            citations_text += "\n" + addition
                            # Save progress after each successful addition
                            with open(citations_cache_path, "w") as f:
                                f.write(citations_text)
                            with open(progress_path, "w") as f:
                                json.dump(
                                    {
                                        "completed_rounds": round_idx + 1,
                                        "status": "in_progress",
                                    },
                                    f,
                                )
                        else:
                            print("[citations] skipped duplicate title; no update applied.")
                            with open(citations_cache_path, "w") as f:
                                f.write(citations_text)
                            with open(progress_path, "w") as f:
                                json.dump(
                                    {
                                        "completed_rounds": round_idx + 1,
                                        "status": "in_progress",
                                        "last_round": "duplicate_title",
                                    },
                                    f,
                                )
                    else:
                        citations_text += "\n" + addition
                        with open(citations_cache_path, "w") as f:
                            f.write(citations_text)
                        with open(progress_path, "w") as f:
                            json.dump(
                                {
                                    "completed_rounds": round_idx + 1,
                                    "status": "in_progress",
                                    "last_round": "added_no_title_field",
                                },
                                f,
                            )
                else:
                    # No citations added in this round. Still persist progress so resume does not
                    # repeatedly restart from the same round.
                    print("[citations] no suitable papers selected; continuing.")
                    with open(citations_cache_path, "w") as f:
                        f.write(citations_text)
                    with open(progress_path, "w") as f:
                        json.dump(
                            {
                                "completed_rounds": round_idx + 1,
                                "status": "in_progress",
                                "last_round": "no_addition",
                            },
                            f,
                        )

            except Exception as e:
                print(f"Error in citation round {round_idx}: {e}")
                print(traceback.format_exc())
                # Save progress even if there's an error
                with open(citations_cache_path, "w") as f:
                    f.write(citations_text)
                with open(progress_path, "w") as f:
                    json.dump({"completed_rounds": round_idx, "status": "error"}, f)
                continue

        return citations_text if citations_text else None

    except Exception:
        print("EXCEPTION in gather_citations:")
        print(traceback.format_exc())
        return citations_text if citations_text else None


def perform_writeup(
    base_folder,
    citations_text=None,
    no_writing=False,
    num_cite_rounds=20,
    small_model="deepseek-chat",
    big_model="deepseek-chat",
    n_writeup_reflections=3,
    page_limit=4,
    symbolic_facts: bool = False,
    remediation_context: dict[str, object] | None = None,
):
    pdf_file = osp.join(base_folder, f"{osp.basename(base_folder)}.pdf")
    latex_folder = osp.join(base_folder, "latex")
    writeup_file = osp.join(latex_folder, "template.tex")
    used_facts_path = osp.join(latex_folder, "used_facts.json")
    retry_target = remediation_retry_target(remediation_context)
    reuse_existing_writeup = (
        symbolic_facts and retry_target == "writeup" and has_validated_writeup(latex_folder)
    )
    reuse_symbolic_artifacts = (
        symbolic_facts
        and should_reuse_symbolic_writeup_artifacts(remediation_context)
        and osp.exists(writeup_file)
        and osp.exists(used_facts_path)
    )

    if not reuse_existing_writeup and not reuse_symbolic_artifacts:
        if osp.exists(latex_folder):
            shutil.rmtree(latex_folder)
        if osp.exists(pdf_file):
            os.remove(pdf_file)
        for old_pdf in os.listdir(base_folder):
            if old_pdf.endswith(".pdf") and "reflection" in old_pdf:
                os.remove(osp.join(base_folder, old_pdf))

    try:
        idea_text = load_idea_text(base_folder)
        facts_instructions = ""
        facts_index = ""
        params_index = ""
        param_store = None
        summaries_for_prompt = {}
        remediation_instructions = build_remediation_prompt_block(remediation_context)
        fact_store_path = osp.join(base_folder, "logs", "0-run", "fact_store.json")
        param_store_path = osp.join(base_folder, "logs", "0-run", "param_store.json")
        if symbolic_facts:
            store, writeup_summary_symbolic = ensure_symbolic_writeup_inputs(base_folder)
            param_store = ensure_param_store_for_run(base_folder)
            if not store.facts:
                raise RuntimeError(
                    "Symbolic-facts writeup requested but FactStore is empty. "
                    "Ensure baseline_summary.json and research_summary.json exist under logs/0-run."
                )

            facts_index = facts_index_for_prompt(store)
            params_index = params_index_for_prompt(param_store)
            facts_instructions = build_writeup_mode_instructions(symbolic_facts)
            summaries_for_prompt = {
                "WRITEUP_SYMBOLIC_SUMMARY": writeup_summary_symbolic,
            }
        else:
            exp_summaries = load_exp_summaries(base_folder)
            filtered_summaries_for_writeup = filter_experiment_summaries(
                exp_summaries, step_name="writeup"
            )
            facts_instructions = build_writeup_mode_instructions(symbolic_facts)
            summaries_for_prompt = filtered_summaries_for_writeup
        # Prepare a new fresh latex folder
        if not osp.exists(osp.join(latex_folder, "template.tex")):
            shutil.copytree(
                "ai_scientist/blank_icbinb_latex", latex_folder, dirs_exist_ok=True
            )
        if reuse_existing_writeup and restore_validated_writeup(latex_folder, writeup_file):
            print("[writeup] restored last validated symbolic draft for retry")

        writeup_file = osp.join(latex_folder, "template.tex")
        with open(writeup_file, "r") as f:
            writeup_text = f.read()

        # Gather plot filenames from figures/ folder
        figures_dir = osp.join(base_folder, "figures")
        plot_names = []
        if osp.exists(figures_dir):
            for fplot in os.listdir(figures_dir):
                if fplot.lower().endswith(".png"):
                    plot_names.append(fplot)

        manifest = None
        manifest_summary_str = "{}"
        if symbolic_facts:
            manifest = ensure_artifact_manifest(base_folder=base_folder, store=store)
            manifest_summary_str = json.dumps(
                artifact_manifest_summary(manifest),
                indent=2,
            )

        if no_writing:
            if symbolic_facts:
                compile_symbolic_latex_project(
                    latex_folder=latex_folder,
                    pdf_file=pdf_file,
                    fact_store_path=fact_store_path,
                    param_store_path=param_store_path,
                    rendered_tex_artifact_path=osp.join(latex_folder, "template.rendered.tex"),
                    used_facts_artifact_path=osp.join(latex_folder, "used_facts.json"),
                )
            else:
                compile_latex(latex_folder, pdf_file)
            return osp.exists(pdf_file)

        # If no citations provided, try to load from cache first
        if citations_text is None:
            citations_cache_path = osp.join(base_folder, "cached_citations.bib")
            if osp.exists(citations_cache_path):
                try:
                    with open(citations_cache_path, "r") as f:
                        citations_text = f.read()
                    print("Loaded citations from cache")
                except Exception as e:
                    print(f"Error loading cached citations: {e}")
                    citations_text = None

            # If still no citations, gather them
            if not citations_text:
                citations_text = gather_citations(
                    base_folder, num_cite_rounds, small_model
                )
                if citations_text is None:
                    print("Warning: Citation gathering failed")
                    citations_text = ""

        # Insert citations into template.tex
        if citations_text and not reuse_existing_writeup:
            citations_text, bibtex_replacements = sanitize_bibtex_text_for_pdflatex(citations_text)
            if bibtex_replacements:
                print(
                    "[citations] sanitized citations for pdflatex before embedding: "
                    + ", ".join(f"{k}x{v}" for k, v in sorted(bibtex_replacements.items()))
                )
            with open(writeup_file, "r") as f:
                content = f.read()
            pattern_end = r"\end{filecontents}"
            content = content.replace(pattern_end, f"\n{citations_text}{pattern_end}")
            with open(writeup_file, "w") as f:
                f.write(content)

        desc_map = {}
        plot_generation_notes = ""
        vlm_client = None
        vlm_model = None
        if n_writeup_reflections > 0 or not symbolic_facts:
            vlm_client, vlm_model = create_vlm_client(small_model)
        if not symbolic_facts:
            aggregator_path = osp.join(base_folder, "auto_plot_aggregator.py")
            aggregator_code = "No aggregator script found."
            if osp.exists(aggregator_path):
                with open(aggregator_path, "r") as fa:
                    aggregator_code = fa.read()
            try:
                for pf in plot_names:
                    ppath = osp.join(figures_dir, pf)
                    if not osp.exists(ppath):
                        continue
                    img_dict = {
                        "images": [ppath],
                        "caption": "No direct caption",
                    }
                    review_data = generate_vlm_img_review(img_dict, vlm_model, vlm_client)
                    if review_data:
                        desc_map[pf] = review_data.get(
                            "Img_description", "No description found"
                        )
                    else:
                        desc_map[pf] = "No description found"

            except Exception:
                print("EXCEPTION in VLM figure description generation:")
                print(traceback.format_exc())
                desc_map = {}
            plot_generation_notes = (
                "We also have a script used to produce the final plots "
                "(use this to see how the plots are generated and what names are used in the legend):\n"
                "```python\n"
                f"{aggregator_code}\n"
                "```\n"
                "Please also consider which plots can naturally be grouped together as subfigures."
            )

        big_model_system_message = writeup_system_message_template.format(
            page_limit=page_limit
        )
        if symbolic_facts:
            big_model_system_message = (
                big_model_system_message + "\n\n" + facts_instructions
            )
        big_client, big_client_model = create_client(big_model)
        with open(writeup_file, "r") as f:
            writeup_text = f.read()
        writeup_context_pack = build_writeup_context_pack(
            symbolic_facts=symbolic_facts,
            summaries_for_prompt=summaries_for_prompt,
            facts_index=facts_index,
            params_index=params_index,
            artifact_manifest_summary=json.loads(manifest_summary_str),
            current_latex=writeup_text,
            plot_names=plot_names,
            plot_descriptions=desc_map,
        )
        save_context_pack(
            osp.join(latex_folder, "writeup_context_pack.json"),
            writeup_context_pack,
        )

        facts_priority_block = ""
        if symbolic_facts:
            preferred = writeup_context_pack.get("preferred_fact_refs")
            if isinstance(preferred, list) and preferred:
                lines = ["PRIORITY FACT KEYS (must use at least 1 in main text):"]
                for item in preferred[:8]:
                    if not isinstance(item, dict):
                        continue
                    key = str(item.get("key") or "").strip()
                    meaning = str(item.get("meaning") or "").strip().replace("\n", " ")
                    if not key or not meaning:
                        continue
                    lines.append(f"- {key}: {meaning}")
                facts_priority_block = "\n".join(lines)

        combined_prompt = writeup_prompt.format(
            idea_text=idea_text,
            writeup_context_pack=json.dumps(writeup_context_pack, indent=2),
            facts_instructions=facts_instructions,
            facts_priority_block=facts_priority_block,
            remediation_instructions=remediation_instructions,
            plot_generation_notes=plot_generation_notes,
        )

        writeup_client = big_client
        writeup_client_model = big_client_model
        if symbolic_facts and remediation_context is not None and retry_target == "writeup":
            writeup_client, writeup_client_model = create_client(small_model)
            print(f"[writeup] remediation retry using repair model: {small_model}")

        response, msg_history = get_response_from_llm(
            prompt=combined_prompt,
            client=writeup_client,
            model=writeup_client_model,
            system_message=big_model_system_message,
            print_debug=False,
        )

        updated_latex_code, extract_kind = _extract_latex_from_response(response)
        if updated_latex_code is None:
            raise SymbolicLatexError(
                "LLM response did not contain a complete ```latex``` block."
            )
        if extract_kind != "```latex```":
            print(f"[writeup] extracted LaTeX via {extract_kind} (expected ```latex``` fence)")
        updated_latex_code, fixup_report = fixup_latex_before_validation(
            updated_latex_code,
            scaffold_tex=writeup_text,
        )
        if fixup_report.scaffold_preserved:
            print("[writeup] preserved canonical LaTeX scaffold before validation")
        if fixup_report.embedded_label_fixes:
            print(
                "[writeup] normalized malformed includegraphics blocks before validation: "
                f"{fixup_report.embedded_label_fixes}"
            )
        if fixup_report.bibliography_fixed:
            print("[writeup] normalized \\bibliography database to references before validation")
        if fixup_report.end_document_appended:
            print("[writeup] appended missing \\end{document} before validation")
        updated_latex_code = _maybe_backanchor_numeric_literals(
            updated_latex_code,
            symbolic_facts=symbolic_facts,
            store=store if symbolic_facts else None,
            param_store=param_store if symbolic_facts else None,
        )
        with open(writeup_file, "w") as f:
            f.write(updated_latex_code)
        validate_generated_writeup(
            updated_latex_code,
            symbolic_facts=symbolic_facts,
            store=store if symbolic_facts else None,
            param_store=param_store if symbolic_facts else None,
            artifact_manifest=manifest,
        )
        current_latex = updated_latex_code

        if n_writeup_reflections <= 0:
            if symbolic_facts:
                compile_symbolic_latex_project(
                    latex_folder=latex_folder,
                    pdf_file=pdf_file,
                    fact_store_path=fact_store_path,
                    param_store_path=param_store_path,
                    rendered_tex_artifact_path=osp.join(
                        latex_folder, "template.rendered.tex"
                    ),
                    used_facts_artifact_path=osp.join(latex_folder, "used_facts.json"),
                )
                save_validated_writeup(latex_folder, updated_latex_code)
            else:
                compile_latex(latex_folder, pdf_file)
        final_pdf_path = pdf_file

        # Multiple reflection loops on the final LaTeX
        reflection_aborted = False
        for i in range(n_writeup_reflections):
            with open(writeup_file, "r") as f:
                current_latex = f.read()

            # Check for unused or invalid figure references
            referenced_figs_temp = re.findall(
                r"\\includegraphics(?:\[[^\]]*\])?{([^}]+)}", current_latex
            )
            used_figs = set(os.path.basename(fig) for fig in referenced_figs_temp)
            all_figs = set(plot_names)
            unused_figs = all_figs - used_figs
            invalid_figs = used_figs - all_figs

            # Save PDF with reflection trial number
            reflection_pdf = osp.join(
                base_folder, f"{osp.basename(base_folder)}_reflection{i+1}.pdf"
            )
            # Compile current version before reflection
            print(f"[green]Compiling PDF for reflection {i+1}...[/green]")
            if symbolic_facts:
                compile_symbolic_latex_project(
                    latex_folder=latex_folder,
                    pdf_file=reflection_pdf,
                    fact_store_path=fact_store_path,
                    param_store_path=param_store_path,
                    rendered_tex_artifact_path=osp.join(latex_folder, "template.rendered.tex"),
                    used_facts_artifact_path=osp.join(latex_folder, "used_facts.json"),
                )
                save_validated_writeup(latex_folder, current_latex)
            else:
                compile_latex(latex_folder, reflection_pdf)
            final_pdf_path = reflection_pdf

            review_img_cap_ref = perform_imgs_cap_ref_review(
                vlm_client, vlm_model, reflection_pdf
            )

            # Detect duplicate figures between main text and appendix
            analysis_duplicate_figs = detect_duplicate_figures(
                vlm_client, vlm_model, reflection_pdf
            )
            print(analysis_duplicate_figs)

            # Get reflection_page_info
            reflection_page_info = get_reflection_page_info(reflection_pdf, page_limit)

            check_output = os.popen(  # TODO: should prob use subprocess instead
                f"chktex {writeup_file} -q -n2 -n24 -n13 -n1"
            ).read()

            reflection_guard_block = build_reflection_guard_block(
                symbolic_facts=symbolic_facts,
                remediation_instructions=remediation_instructions,
            )

            patch_contract = (
                "PATCH MODE (symbolic-facts reflection):\n"
                "- Do NOT return full template.tex; do NOT touch the scaffold before \\end{filecontents}.\n"
                "- Return a JSON object with a single key: ops.\n"
                "- If no changes are needed, return: {\"ops\": []}\n"
                "- Allowed ops:\n"
                "  - replace_abstract: {\"op\":\"replace_abstract\",\"text\":\"...\"}\n"
                "  - replace_section: {\"op\":\"replace_section\",\"section_label\":\"sec:intro\",\"text\":\"...\"}\n"
                "    (or use section_title instead of section_label)\n"
                "  - replace_between: {\"op\":\"replace_between\",\"start\":\"...\",\"end\":\"...\",\"text\":\"...\"}\n"
                "  - insert_after: {\"op\":\"insert_after\",\"anchor\":\"...\",\"text\":\"...\"}\n"
                "  - delete_between: {\"op\":\"delete_between\",\"start\":\"...\",\"end\":\"...\"}\n"
                "- Hard rules:\n"
                "  - Do not remove any existing \\fact{...} or \\param{...} placeholders.\n"
                "  - Do not introduce numeric literals; keep results symbolic.\n"
                "  - Keep LaTeX compilable.\n"
                "Return format:\n"
                "RESPONSE:\n"
                "```json\n"
                "{\"ops\": [...]} \n"
                "```\n"
            )

            current_editable = ""
            if symbolic_facts:
                current_editable = editable_suffix(current_latex)

            reflection_prompt = (
                f"""
Now let's reflect and identify any issues (including but not limited to):
1) Are there any LaTeX syntax errors or style violations we can fix? Refer to the chktex output below.
2) Is the writing clear, and scientifically rigorous for a workshop focusing on real-world pitfalls?
3) Have we included all relevant details from the summaries without hallucinating?
4) Are there short sections (one or two sentences) that could be combined into a single paragraph?
5) Can we use more information and details (hyperparameters, unused figures, etc.) in the supplementary material? Only add information that is not already covered in the main paper.
6) The following figures are available in the folder but not used in the LaTeX: {sorted(unused_figs)}
7) The following figure references in the LaTeX do not match any actual file: {sorted(invalid_figs)}
{reflection_page_info}
chktex results:
```
{check_output}
```
8) Issues identified in the VLM reviews of the images, their captions, and related text discussions. Ensure each caption clearly matches its image content and that there is substantial discussion of each figure in the text.
VLM reviews:
```
{review_img_cap_ref}
```

9) Duplicate figures between main text and appendix. Make sure to remove the duplicate figures from the appendix.
```
{analysis_duplicate_figs}
```

            {reflection_guard_block}
"""
                + (
                    f"\n{patch_contract}\n\nCURRENT EDITABLE LaTeX (after \\end{{filecontents}}):\n```latex\n{current_editable}\n```\n"
                    if symbolic_facts
                    else (
                        "\n"
                        "Please provide a revised complete LaTeX in triple backticks, or repeat the same if no changes are needed.\n"
                        "Return the entire file in full. This must be an acceptable complete LaTeX writeup.\n"
                        "Do not hallucinate any details!\n"
                        "Ensure proper citation usage:\n"
                        "- Always include references within \\\\begin{filecontents}{references.bib} ... \\\\end{filecontents}, even if they haven't changed from the previous round.\n"
                        "- Use citations from the provided references.bib content.\n"
                    )
                )
            )

            reflection_response, msg_history = get_response_from_llm(
                prompt=reflection_prompt,
                client=big_client,
                model=big_client_model,
                system_message=big_model_system_message,
                msg_history=msg_history[-1:],
                print_debug=False,
            )

            if symbolic_facts:
                patch_json = _extract_patch_from_response(reflection_response)
                if patch_json is None:
                    print(f"No valid JSON patch found in reflection step {i+1}.")
                    break
                ops = patch_json.get("ops")
                if not ops:
                    print(f"No changes in reflection step {i+1}.")
                    break
                try:
                    patch_result = apply_latex_patch_ops(current_latex, patch_json)
                    final_text, fixup_report = fixup_latex_before_validation(
                        patch_result.latex_text,
                        scaffold_tex=current_latex,
                    )
                    if fixup_report.scaffold_preserved:
                        print(
                            "[writeup] preserved canonical LaTeX scaffold after reflection patch"
                        )
                    if fixup_report.embedded_label_fixes:
                        print(
                            "[writeup] normalized malformed includegraphics blocks after reflection patch: "
                            f"{fixup_report.embedded_label_fixes}"
                        )
                    final_text = _maybe_backanchor_numeric_literals(
                        final_text,
                        symbolic_facts=symbolic_facts,
                        store=store,
                        param_store=param_store,
                    )
                    validate_generated_writeup(
                        final_text,
                        symbolic_facts=symbolic_facts,
                        store=store if symbolic_facts else None,
                        param_store=param_store if symbolic_facts else None,
                        artifact_manifest=manifest,
                    )
                    with open(writeup_file, "w") as fo:
                        fo.write(final_text)
                    compile_symbolic_latex_project(
                        latex_folder=latex_folder,
                        pdf_file=reflection_pdf,
                        fact_store_path=fact_store_path,
                        param_store_path=param_store_path,
                        rendered_tex_artifact_path=osp.join(
                            latex_folder, "template.rendered.tex"
                        ),
                        used_facts_artifact_path=osp.join(latex_folder, "used_facts.json"),
                    )
                    save_validated_writeup(latex_folder, final_text)
                    current_latex = final_text
                    final_pdf_path = reflection_pdf
                except Exception as exc:
                    with open(writeup_file, "w") as fo:
                        fo.write(current_latex)
                    print(
                        f"[writeup] discarded reflection {i+1} patch and kept previous validated draft: {exc}"
                    )
                    reflection_aborted = True
                    break
            else:
                reflection_code_match = re.search(
                    r"```latex(.*?)```", reflection_response, re.DOTALL
                )
                if reflection_code_match:
                    reflected_latex_code = reflection_code_match.group(1).strip()
                    final_text, normalization = normalize_generated_latex_draft(
                        reflected_latex_code,
                        scaffold_tex=current_latex,
                    )
                    if final_text != current_latex:
                        if normalization["scaffold_preserved"]:
                            print(
                                "[writeup] preserved canonical LaTeX scaffold after reflection"
                            )
                        if normalization["embedded_label_fixes"]:
                            print(
                                "[writeup] normalized malformed includegraphics blocks after reflection: "
                                f"{normalization['embedded_label_fixes']}"
                            )
                        try:
                            validate_generated_writeup(
                                final_text,
                                symbolic_facts=symbolic_facts,
                                store=store if symbolic_facts else None,
                                param_store=param_store if symbolic_facts else None,
                                artifact_manifest=manifest,
                            )
                            with open(writeup_file, "w") as fo:
                                fo.write(final_text)
                            if symbolic_facts:
                                compile_symbolic_latex_project(
                                    latex_folder=latex_folder,
                                    pdf_file=reflection_pdf,
                                    fact_store_path=fact_store_path,
                                    param_store_path=param_store_path,
                                    rendered_tex_artifact_path=osp.join(
                                        latex_folder, "template.rendered.tex"
                                    ),
                                    used_facts_artifact_path=osp.join(latex_folder, "used_facts.json"),
                                )
                                save_validated_writeup(latex_folder, final_text)
                            else:
                                compile_latex(latex_folder, reflection_pdf)
                            current_latex = final_text
                            final_pdf_path = reflection_pdf
                        except Exception as exc:
                            with open(writeup_file, "w") as fo:
                                fo.write(current_latex)
                            print(
                                f"[writeup] discarded reflection {i+1} revision and kept previous validated draft: {exc}"
                            )
                            reflection_aborted = True
                            break
                    else:
                        print(f"No changes in reflection step {i+1}.")
                        break
                else:
                    print(f"No valid LaTeX code block found in reflection step {i+1}.")
                    break
            # Get new reflection_page_info
            reflection_page_info = get_reflection_page_info(reflection_pdf, page_limit)
            review_img_selection = perform_imgs_cap_ref_review_selection(
                vlm_client, vlm_model, reflection_pdf, reflection_page_info
            )
            current_editable = ""
            if symbolic_facts:
                current_editable = editable_suffix(current_latex)

            img_reflection_prompt = (
                f"""Now let's reflect on
The following figures are currently used in the paper: {sorted(used_figs)}
The following figures are available in the folder but not used in the LaTeX: {sorted(unused_figs)}

{reflection_page_info}

The following is the VLM review on figures:

{review_img_selection}

Please review the figures and make the following changes:
1. For figures that do not add significant value to the paper, move them to the appendix
2. For figures that are not very informative or do not effectively communicate meaningful patterns, remove them entirely
3. For figures that does not contain subfigures and present sparse information, consider combining them with other related figures
4. Update all relevant text discussions to reflect any changes in figure placement or combinations
5. Enhance the scientific analysis of the remaining figures in the text - provide detailed, insightful discussions of their significance and findings

Please ensure all changes maintain scientific rigor and improve the paper's clarity and impact.
Be more aggressive with figure selection - move more figures to the appendix or group them together with other figures if the page limit is already exceeded.
{reflection_guard_block}

If you believe you are done with reflection, simply say: "I am done".
"""
                + (
                    f"\n{patch_contract}\n\nCURRENT EDITABLE LaTeX (after \\end{{filecontents}}):\n```latex\n{current_editable}\n```\n"
                    if symbolic_facts
                    else "If you make changes, return the full revised LaTeX in triple backticks."
                )
            )
            reflection_response, msg_history = get_response_from_llm(
                prompt=img_reflection_prompt,
                client=big_client,
                model=big_client_model,
                system_message=big_model_system_message,
                msg_history=msg_history[-1:],
                print_debug=False,
            )

            if "I am done" in reflection_response:
                print(
                    "LLM indicated it is done with reflections. Exiting reflection loop."
                )
                break

            if symbolic_facts:
                patch_json = _extract_patch_from_response(reflection_response)
                if patch_json is None:
                    print(f"No valid JSON patch found in figure reflection step {i+1}.")
                    break
                ops = patch_json.get("ops")
                if not ops:
                    print(f"No changes in reflection step {i+1}.")
                    break
                try:
                    patch_result = apply_latex_patch_ops(current_latex, patch_json)
                    final_text, fixup_report = fixup_latex_before_validation(
                        patch_result.latex_text,
                        scaffold_tex=current_latex,
                    )
                    if fixup_report.scaffold_preserved:
                        print(
                            "[writeup] preserved canonical LaTeX scaffold after figure reflection patch"
                        )
                    if fixup_report.embedded_label_fixes:
                        print(
                            "[writeup] normalized malformed includegraphics blocks after figure reflection patch: "
                            f"{fixup_report.embedded_label_fixes}"
                        )
                    final_text = _maybe_backanchor_numeric_literals(
                        final_text,
                        symbolic_facts=symbolic_facts,
                        store=store,
                        param_store=param_store,
                    )
                    validate_generated_writeup(
                        final_text,
                        symbolic_facts=symbolic_facts,
                        store=store if symbolic_facts else None,
                        param_store=param_store if symbolic_facts else None,
                        artifact_manifest=manifest,
                    )
                    with open(writeup_file, "w") as fo:
                        fo.write(final_text)
                    compile_symbolic_latex_project(
                        latex_folder=latex_folder,
                        pdf_file=reflection_pdf,
                        fact_store_path=fact_store_path,
                        param_store_path=param_store_path,
                        rendered_tex_artifact_path=osp.join(
                            latex_folder, "template.rendered.tex"
                        ),
                        used_facts_artifact_path=osp.join(latex_folder, "used_facts.json"),
                    )
                    save_validated_writeup(latex_folder, final_text)
                    current_latex = final_text
                    final_pdf_path = reflection_pdf
                except Exception as exc:
                    with open(writeup_file, "w") as fo:
                        fo.write(current_latex)
                    print(
                        f"[writeup] discarded figure reflection {i+1} patch and kept previous validated draft: {exc}"
                    )
                    reflection_aborted = True
                    break
            else:
                reflection_code_match = re.search(
                    r"```latex(.*?)```", reflection_response, re.DOTALL
                )
                if reflection_code_match:
                    reflected_latex_code = reflection_code_match.group(1).strip()
                    final_text, normalization = normalize_generated_latex_draft(
                        reflected_latex_code,
                        scaffold_tex=current_latex,
                    )
                    if final_text != current_latex:
                        if normalization["scaffold_preserved"]:
                            print(
                                "[writeup] preserved canonical LaTeX scaffold after figure reflection"
                            )
                        if normalization["embedded_label_fixes"]:
                            print(
                                "[writeup] normalized malformed includegraphics blocks after figure reflection: "
                                f"{normalization['embedded_label_fixes']}"
                            )
                        try:
                            validate_generated_writeup(
                                final_text,
                                symbolic_facts=symbolic_facts,
                                store=store if symbolic_facts else None,
                                param_store=param_store if symbolic_facts else None,
                                artifact_manifest=manifest,
                            )
                            with open(writeup_file, "w") as fo:
                                fo.write(final_text)
                            if symbolic_facts:
                                compile_symbolic_latex_project(
                                    latex_folder=latex_folder,
                                    pdf_file=reflection_pdf,
                                    fact_store_path=fact_store_path,
                                    param_store_path=param_store_path,
                                    rendered_tex_artifact_path=osp.join(
                                        latex_folder, "template.rendered.tex"
                                    ),
                                    used_facts_artifact_path=osp.join(latex_folder, "used_facts.json"),
                                )
                                save_validated_writeup(latex_folder, final_text)
                            else:
                                compile_latex(latex_folder, reflection_pdf)
                            current_latex = final_text
                            final_pdf_path = reflection_pdf
                        except Exception as exc:
                            with open(writeup_file, "w") as fo:
                                fo.write(current_latex)
                            print(
                                f"[writeup] discarded figure reflection {i+1} revision and kept previous validated draft: {exc}"
                            )
                            reflection_aborted = True
                            break
                    else:
                        print(f"No changes in reflection step {i+1}.")
                        break
                else:
                    print(f"No valid LaTeX code block found in reflection step {i+1}.")
                    break

        if n_writeup_reflections > 0 and not reflection_aborted:
            reflection_page_info = get_reflection_page_info(reflection_pdf, page_limit)

            current_editable = ""
            if symbolic_facts:
                current_editable = editable_suffix(current_latex)

            final_reflection_prompt = (
                f"""{reflection_page_info}
USE MINIMAL EDITS TO OPTIMIZE THE PAGE LIMIT USAGE.
{build_reflection_guard_block(symbolic_facts=symbolic_facts, remediation_instructions=remediation_instructions)}
"""
                + (
                    f"\n{patch_contract}\n\nCURRENT EDITABLE LaTeX (after \\end{{filecontents}}):\n```latex\n{current_editable}\n```\n"
                    if symbolic_facts
                    else "If you make changes, return the full revised LaTeX in triple backticks. Otherwise repeat the current full LaTeX."
                )
            )
            reflection_response, msg_history = get_response_from_llm(
                prompt=final_reflection_prompt,
                client=big_client,
                model=big_client_model,
                system_message=big_model_system_message,
                msg_history=msg_history[-1:],
                print_debug=False,
            )

            reflection_pdf = osp.join(
                base_folder, f"{osp.basename(base_folder)}_reflection_final_page_limit.pdf"
            )
            print(f"[green]Compiling PDF for reflection final page limit...[/green]")
            print(f"reflection step {i+1}")

            if symbolic_facts:
                patch_json = _extract_patch_from_response(reflection_response)
                if patch_json is None:
                    print("No valid JSON patch found in page-limit reflection step.")
                else:
                    ops = patch_json.get("ops")
                    if not ops:
                        print("No changes in reflection page step.")
                    else:
                        try:
                            patch_result = apply_latex_patch_ops(current_latex, patch_json)
                            final_text, fixup_report = fixup_latex_before_validation(
                                patch_result.latex_text,
                                scaffold_tex=current_latex,
                            )
                            if fixup_report.scaffold_preserved:
                                print(
                                    "[writeup] preserved canonical LaTeX scaffold after page-limit reflection patch"
                                )
                            if fixup_report.embedded_label_fixes:
                                print(
                                    "[writeup] normalized malformed includegraphics blocks after page-limit reflection patch: "
                                    f"{fixup_report.embedded_label_fixes}"
                                )
                            final_text = _maybe_backanchor_numeric_literals(
                                final_text,
                                symbolic_facts=symbolic_facts,
                                store=store,
                                param_store=param_store,
                            )
                            validate_generated_writeup(
                                final_text,
                                symbolic_facts=symbolic_facts,
                                store=store if symbolic_facts else None,
                                param_store=param_store if symbolic_facts else None,
                                artifact_manifest=manifest,
                            )
                            with open(writeup_file, "w") as fo:
                                fo.write(final_text)
                            compile_symbolic_latex_project(
                                latex_folder=latex_folder,
                                pdf_file=reflection_pdf,
                                fact_store_path=fact_store_path,
                                param_store_path=param_store_path,
                                rendered_tex_artifact_path=osp.join(
                                    latex_folder, "template.rendered.tex"
                                ),
                                used_facts_artifact_path=osp.join(
                                    latex_folder, "used_facts.json"
                                ),
                            )
                            save_validated_writeup(latex_folder, final_text)
                            current_latex = final_text
                            final_pdf_path = reflection_pdf
                        except Exception as exc:
                            with open(writeup_file, "w") as fo:
                                fo.write(current_latex)
                            print(
                                "[writeup] discarded page-limit reflection patch and kept previous "
                                f"validated draft: {exc}"
                            )
            else:
                reflection_code_match = re.search(
                    r"```latex(.*?)```", reflection_response, re.DOTALL
                )
                if reflection_code_match:
                    reflected_latex_code = reflection_code_match.group(1).strip()
                    final_text, normalization = normalize_generated_latex_draft(
                        reflected_latex_code,
                        scaffold_tex=current_latex,
                    )
                    if final_text != current_latex:
                        if normalization["scaffold_preserved"]:
                            print(
                                "[writeup] preserved canonical LaTeX scaffold after page-limit reflection"
                            )
                        if normalization["embedded_label_fixes"]:
                            print(
                                "[writeup] normalized malformed includegraphics blocks after page-limit reflection: "
                                f"{normalization['embedded_label_fixes']}"
                            )
                        try:
                            validate_generated_writeup(
                                final_text,
                                symbolic_facts=symbolic_facts,
                                store=store if symbolic_facts else None,
                                param_store=param_store if symbolic_facts else None,
                                artifact_manifest=manifest,
                            )
                            with open(writeup_file, "w") as fo:
                                fo.write(final_text)
                            if symbolic_facts:
                                compile_symbolic_latex_project(
                                    latex_folder=latex_folder,
                                    pdf_file=reflection_pdf,
                                    fact_store_path=fact_store_path,
                                    param_store_path=param_store_path,
                                    rendered_tex_artifact_path=osp.join(
                                        latex_folder, "template.rendered.tex"
                                    ),
                                    used_facts_artifact_path=osp.join(
                                        latex_folder, "used_facts.json"
                                    ),
                                )
                                save_validated_writeup(latex_folder, final_text)
                            else:
                                compile_latex(latex_folder, reflection_pdf)
                            current_latex = final_text
                            final_pdf_path = reflection_pdf
                        except Exception as exc:
                            with open(writeup_file, "w") as fo:
                                fo.write(current_latex)
                            print(
                                "[writeup] discarded page-limit reflection revision and kept previous "
                                f"validated draft: {exc}"
                            )
                    else:
                        print(f"No changes in reflection page step.")

        if symbolic_facts:
            used_facts_path = osp.join(latex_folder, "used_facts.json")
            if not osp.exists(used_facts_path):
                raise RuntimeError(
                    "Symbolic-facts writeup expected used_facts.json but it was not generated."
                )
            print("[claims] loading used_facts and symbolic_tex...")
            with open(used_facts_path, "r", encoding="utf-8") as f:
                used_facts = json.load(f)
            with open(writeup_file, "r", encoding="utf-8") as f:
                symbolic_tex = f.read()
            manifest = ensure_artifact_manifest(
                base_folder=base_folder,
                store=store,
                tex_path=writeup_file,
            )
            print("[claims] building claim context pack...")
            claim_context_pack = build_claim_context_pack(
                symbolic_tex=symbolic_tex,
                used_facts=used_facts,
                artifact_manifest_summary=artifact_manifest_summary(manifest),
            )
            save_context_pack(
                osp.join(latex_folder, "claim_context_pack.json"),
                claim_context_pack,
            )
            print("[claims] generating claim ledger...")
            ledger = generate_claim_ledger(
                symbolic_tex=symbolic_tex,
                used_facts=used_facts,
                client=big_client,
                model=big_client_model,
                artifact_manifest_summary=artifact_manifest_summary(manifest),
                context_pack=claim_context_pack,
                remediation_instructions=remediation_instructions,
            )
            used_keys = used_facts.get("used_keys")
            used_keys = used_keys if isinstance(used_keys, list) else []
            validate_claim_ledger(
                ledger,
                store=store,
                used_keys=used_keys,
                artifact_manifest=manifest,
            )
            claim_ledger_path = osp.join(latex_folder, "claim_ledger.json")
            save_claim_ledger(claim_ledger_path, ledger)
            print(f"[claims] wrote claim ledger: {claim_ledger_path}")
            claim_trace_index = build_claim_trace_index(
                ledger=ledger,
                symbolic_tex=symbolic_tex,
            )
            validate_used_fact_traceability(
                claim_trace_index,
                used_keys=used_keys,
            )
            claim_trace_path = osp.join(latex_folder, "claim_trace_index.json")
            save_claim_trace_index(claim_trace_path, claim_trace_index)
            print(f"[claims] wrote claim trace index: {claim_trace_path}")

            if env_bool("AI_SCIENTIST_OUTSIDER_AUDIT", True):
                audit_model_name = env_str("AI_SCIENTIST_MODEL_OUTSIDER_AUDIT", small_model)
                print(f"[audit] creating audit client: model={audit_model_name}")
                audit_client, audit_model = create_client(audit_model_name)
                print("[audit] building outsider audit inputs...")
                audit_inputs = build_outsider_audit_inputs(
                    symbolic_tex=symbolic_tex,
                    used_facts=used_facts,
                    store=store,
                    claim_ledger=ledger,
                    artifact_manifest_summary=artifact_manifest_summary(manifest),
                )
                save_context_pack(
                    osp.join(latex_folder, "audit_context_pack.json"),
                    build_outsider_audit_context_pack(audit_inputs),
                )
                print("[audit] generating outsider audit...")
                audit = generate_outsider_audit(
                    inputs=audit_inputs,
                    client=audit_client,
                    model=audit_model,
                )
                validate_outsider_audit(audit)
                audit_path = osp.join(latex_folder, "outsider_audit.json")
                save_outsider_audit(audit_path, audit)
                print(f"[audit] wrote outsider audit: {audit_path}")

        return osp.exists(final_pdf_path)

    except Exception as exc:
        trace_text = traceback.format_exc()
        report = classify_remediation_failure(
            phase="perform_icbinb_writeup",
            exc=exc,
            traceback_text=trace_text,
        )
        save_remediation_report(
            osp.join(latex_folder, "remediation_failure.json"),
            report,
        )
        print("EXCEPTION in perform_writeup:")
        print(trace_text)
        print_remediation_report(report)
        return False


if __name__ == "__main__":
    from ai_scientist.env_utils import env_int, env_str, load_env

    load_env()
    parser = argparse.ArgumentParser(description="Perform writeup for a project")
    parser.add_argument("--folder", type=str, help="Project folder", required=True)
    parser.add_argument("--no-writing", action="store_true", help="Only generate")
    parser.add_argument(
        "--num-cite-rounds",
        type=int,
        default=env_int("AI_SCIENTIST_NUM_CITE_ROUNDS", 20),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=env_str("AI_SCIENTIST_MODEL_CITATION", "deepseek-chat"),
        help="Model to use for citation collection (small model). Supports arbitrary OpenAI-compatible model names.",
    )
    parser.add_argument(
        "--big-model",
        type=str,
        default=env_str("AI_SCIENTIST_MODEL_WRITEUP", "deepseek-chat"),
        help="Model to use for final writeup (big model). Supports arbitrary OpenAI-compatible model names.",
    )
    parser.add_argument(
        "--writeup-reflections",
        type=int,
        default=env_int("AI_SCIENTIST_WRITEUP_REFLECTIONS", 3),
        help="Number of reflection steps for the final LaTeX writeup.",
    )
    parser.add_argument(
        "--page-limit",
        type=int,
        default=env_int("AI_SCIENTIST_WRITEUP_PAGE_LIMIT", 4),
        help="Target page limit for the main paper (excluding references).",
    )
    args = parser.parse_args()

    try:
        success = perform_writeup(
            base_folder=args.folder,
            no_writing=args.no_writing,
            num_cite_rounds=args.num_cite_rounds,
            small_model=args.model,
            big_model=args.big_model,
            n_writeup_reflections=args.writeup_reflections,
            page_limit=args.page_limit,
        )
        if not success:
            print("Writeup process did not complete successfully.")
    except Exception:
        print("EXCEPTION in main:")
        print(traceback.format_exc())
