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
from typing import Any

from ai_scientist.llm import (
    get_response_from_llm,
    extract_json_between_markers,
    create_client,
)

from ai_scientist.tools.literature_search import search_for_papers

from ai_scientist.perform_vlm_review import generate_vlm_img_review
from ai_scientist.vlm import create_client as create_vlm_client
from ai_scientist.latex_sanitize import (
    ensure_tex_has_end_document,
    ensure_tex_uses_references_bibliography,
    sanitize_bibtex_text_for_pdflatex,
    sanitize_tex_file_for_pdflatex,
)

from ai_scientist.reliable.facts import facts_index_for_prompt
from ai_scientist.reliable.latex_fixup import fixup_latex_before_validation
from ai_scientist.reliable.latex_patch import apply_latex_patch_ops
from ai_scientist.reliable.latex_scaffold import editable_suffix
from ai_scientist.reliable.latex_compile import (
    compile_symbolic_latex_project,
    ensure_latex_template_assets,
)
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
    remediation_retry_target,
    save_remediation_report,
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
from ai_scientist.env_utils import env_bool, env_str, env_str_optional


def remove_accents_and_clean(s):
    # print("Original:", s)
    # Normalize to separate accents
    nfkd_form = unicodedata.normalize("NFKD", s)
    # Remove non-ASCII characters
    ascii_str = nfkd_form.encode("ASCII", "ignore").decode("ascii")
    # Remove anything but letters, digits, underscores, colons, dashes, @, {, }, and now commas
    ascii_str = re.sub(r"[^a-zA-Z0-9:_@\{\},-]+", "", ascii_str)
    # Convert to lowercase
    ascii_str = ascii_str.lower()
    # print("Cleaned: ", ascii_str)
    return ascii_str


def _extract_patch_from_response(text: str) -> dict[str, Any] | None:
    json_output = extract_json_between_markers(text)
    if not isinstance(json_output, dict):
        return None
    return json_output


def _maybe_backanchor_numeric_literals(
    latex_text: str,
    *,
    symbolic_facts: bool,
    store: Any,
    param_store: Any,
) -> str:
    if not symbolic_facts:
        return latex_text
    if store is None or param_store is None:
        return latex_text
    if not env_bool("AI_SCIENTIST_NUMERIC_BACKANCHOR", False):
        return latex_text

    from ai_scientist.reliable.numeric_backanchor import backanchor_numeric_literals

    try:
        result = backanchor_numeric_literals(
            latex_text,
            store=store,
            param_store=param_store,
        )
    except SymbolicLatexError as exc:
        print(f"[writeup] numeric backanchor skipped: {exc}")
        return latex_text
    if result.replacements:
        summary = ", ".join(
            f"{k!r}x{v}" for k, v in sorted(result.replacements.items())
        )
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


def detect_pages_before_impact(latex_folder, timeout=30):
    """
    Temporarily copy the latex folder, compile, and detect on which page
    the phrase "Impact Statement" appears.
    Returns a tuple (page_number, line_number) if found, otherwise None.
    """
    temp_dir = osp.join(latex_folder, f"_temp_compile_{uuid.uuid4().hex}")
    try:
        shutil.copytree(latex_folder, temp_dir, dirs_exist_ok=True)

        # Compile in the temp folder
        commands = [
            ["pdflatex", "-interaction=nonstopmode", "template.tex"],
            ["bibtex", "template"],
            ["pdflatex", "-interaction=nonstopmode", "template.tex"],
            ["pdflatex", "-interaction=nonstopmode", "template.tex"],
        ]
        for command in commands:
            try:
                subprocess.run(
                    command,
                    cwd=temp_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                return None

        temp_pdf_file = osp.join(temp_dir, "template.pdf")
        if not osp.exists(temp_pdf_file):
            return None

        # Try page-by-page extraction to detect "Impact Statement"
        for i in range(1, 51):
            page_txt = osp.join(temp_dir, f"page_{i}.txt")
            subprocess.run(
                [
                    "pdftotext",
                    "-f",
                    str(i),
                    "-l",
                    str(i),
                    "-q",
                    temp_pdf_file,
                    page_txt,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if not osp.exists(page_txt):
                break
            with open(page_txt, "r", encoding="utf-8", errors="ignore") as fp:
                page_content = fp.read()
            lines = page_content.split("\n")
            for idx, line in enumerate(lines):
                if "Impact Statement" in line:
                    return (i, idx + 1)
        return None
    except Exception:
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def get_citation_addition(
    client, model, context, current_round, total_rounds, idea_text
):
    report, citations = context
    msg_history = []
    citation_system_msg_template = """You are an ambitious AI researcher who is looking to publish a paper to a top-tier ML conference that will contribute significantly to the field.
You have already completed the experiments and now you are looking to collect citations to related papers.
This phase focuses on collecting references and annotating them to be integrated later.
Collected citations will be added to a references.bib file.

Reasons to reference papers include:
1. Summarizing Research: Cite sources when summarizing the existing literature.
2. Using Specific Concepts or Data: Provide citations when discussing specific theories, models, or data.
3. Comparing Findings: Cite relevant studies when comparing or contrasting different findings.
4. Highlighting Research Gaps: Cite previous research when pointing out gaps your survey addresses.
5. Using Established Methods: Cite the creators of methodologies you employ in your survey.
6. Supporting Arguments: Cite sources that back up your conclusions and arguments.
7. Suggesting Future Research: Reference studies related to proposed future research directions.

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

In <THOUGHT>, first briefly reason over the search results and identify which citation(s) best fit your paper.
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
        papers = search_for_papers(query)
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
        desc = json_output["Description"]
        selected_papers = str(json_output["Selected"])

        if selected_papers != "[]":
            selected_indices = []
            for x in selected_papers.strip("[]").split(","):
                x_str = x.strip().strip('"').strip("'")
                if x_str:
                    selected_indices.append(int(x_str))
            assert all(
                [0 <= i < len(papers) for i in selected_indices]
            ), "Invalid paper index"
            bibtexs = [papers[i]["citationStyles"]["bibtex"] for i in selected_indices]

            cleaned_bibtexs = []
            for bibtex in bibtexs:
                newline_index = bibtex.find("\n")
                cite_key_line = bibtex[:newline_index]
                cite_key_line = remove_accents_and_clean(cite_key_line)
                cleaned_bibtexs.append(cite_key_line + bibtex[newline_index:])
            bibtexs = cleaned_bibtexs

            bibtex_string = "\n".join(bibtexs)
        else:
            return None, False

    except Exception:
        print("EXCEPTION in get_citation_addition (selecting papers):")
        print(traceback.format_exc())
        return None, False

    references_format = """% {description}
{bibtex}"""

    references_prompt = references_format.format(bibtex=bibtex_string, description=desc)
    return references_prompt, False


# Using a template string to allow injection of the {page_limit} argument
writeup_system_message_template = """You are an ambitious AI researcher who is looking to publish a paper that will contribute significantly to the field.
Ensure that the paper is scientifically accurate, objective, and truthful. Accurately report the experimental results, even if they are negative or inconclusive.
You are planning to submit to a top-tier ML conference, which has guidelines:
- The main paper is limited to {page_limit} pages, including all figures and tables, but excluding references, the impact statement, and optional appendices. In general, try to use the available space and include all relevant information.
- The main paper should be double-column format, while the appendices can be in single-column format. When in double column format, make sure that tables and figures are correctly placed.
- Do not change the overall style which is mandated by the conference. Keep to the current method of including the references.bib file.
- Do not remove the \\graphicspath directive or no figures will be found.

Here are some tips for each section of the paper:

- **Title**:
  - Title should be catchy and informative. It should give a good idea of what the paper is about.
  - Try to keep it under 2 lines.

- **Abstract**:
  - TL;DR of the paper.
  - What are we trying to do and why is it relevant?
  - Make sure the abstract reads smoothly and is well-motivated. This should be one continuous paragraph.

- **Introduction**:
  - Longer version of the Abstract, i.e., an overview of the entire paper.
  - Provide context to the study and explain its relevance.
  - If results are inconclusive or negative, present them frankly; if they are positive, you may highlight how the approach effectively addresses the research question or problem.
  - Summarize your contributions, highlighting pertinent findings, insights, or proposed methods.

- **Related Work**:
  - Academic siblings of our work, i.e., alternative attempts in literature at trying to address the same or similar problems.
  - Compare and contrast their approach with yours, noting key differences or similarities.
  - Ensure proper citations are provided.

- **Background**:
  - Present foundational concepts or prior work needed to understand your method.
  - This should include necessary definitions, the problem setting, or relevant theoretical constructs.

- **Method**:
  - Clearly detail what you propose to do and why. If your study aims to address certain hypotheses, describe them and how your method is constructed to test them.
  - If results are negative or inconclusive, you may suggest improvements or discuss possible causes.

- **Experimental Setup**:
  - Explain how you tested your method or hypothesis.
  - Describe necessary details such as data, environment, and baselines, but omit hardware details unless explicitly mentioned.

- **Experiments**:
  - Present the results truthfully according to the data you have. If outcomes are not as expected, discuss it transparently.
  - Include comparisons to baselines if available, and only include analyses supported by genuine data.
  - Try to include all relevant plots and tables. Consider combining multiple plots into one figure if they are related.

- **Conclusion**:
  - Summarize the entire paper, including key strengths or findings.
  - If results are strong, highlight how they might address the research problem.
  - If results are negative or inconclusive, highlight potential improvements or reasons and propose future directions.

- **Appendix**:
  - Place for supplementary material that did not fit in the main paper.

Ensure you are always writing good compilable LaTeX code. Common mistakes that should be fixed include:
- LaTeX syntax errors (unenclosed math, unmatched braces, etc.).
- Duplicate figure labels or references.
- Unescaped special characters: & % $ # _ {{ }} ~ ^ \\
- Proper table/figure closure.
- Do not hallucinate new citations or any results not in the logs.

When returning final code, place it in fenced triple backticks with 'latex' syntax highlighting.
"""

writeup_prompt = """Your goal is to write up the following idea:

```markdown
{idea_text}
```

We have the following canonical writeup context pack (JSON):
```json
{writeup_context_pack}
```

{facts_instructions}
{remediation_instructions}

{plot_generation_notes}

Produce the final version of the LaTeX manuscript now, ensuring the paper is coherent, concise, and reports results accurately.
Return the entire file in full.
This must be an acceptable complete LaTeX writeup.

Please provide the updated LaTeX code for 'template.tex', wrapped in triple backticks
with "latex" syntax highlighting, like so:

```latex
<UPDATED LATEX CODE>
```
"""


def perform_writeup(
    base_folder,
    no_writing=False,
    num_cite_rounds=20,
    small_model: str | None = None,
    big_model: str | None = None,
    n_writeup_reflections=3,
    page_limit=8,
    symbolic_facts: bool = False,
    remediation_context: dict[str, object] | None = None,
):
    if small_model is None:
        small_model = env_str_optional("AI_SCIENTIST_MODEL_WRITEUP_SMALL")
    if big_model is None:
        big_model = env_str_optional("AI_SCIENTIST_MODEL_WRITEUP")
    if small_model is None or big_model is None:
        raise RuntimeError(
            "Missing writeup model configuration. Set these in your .env file "
            "(or pass explicit arguments): AI_SCIENTIST_MODEL_WRITEUP_SMALL, AI_SCIENTIST_MODEL_WRITEUP."
        )
    compile_attempt = 0
    base_pdf_file = osp.join(base_folder, f"{osp.basename(base_folder)}")
    latex_folder = osp.join(base_folder, "latex")
    writeup_file = osp.join(latex_folder, "template.tex")
    retry_target = remediation_retry_target(remediation_context)
    reuse_existing_writeup = (
        symbolic_facts and retry_target == "writeup" and has_validated_writeup(latex_folder)
    )

    # Cleanup any previous latex folder and pdf
    if not reuse_existing_writeup and osp.exists(latex_folder):
        shutil.rmtree(latex_folder)
    # if osp.exists(pdf_file):
    #     os.remove(pdf_file)

    try:
        # Load idea text
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

        facts_instructions = build_writeup_mode_instructions(symbolic_facts)
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
            summaries_for_prompt = {
                "WRITEUP_SYMBOLIC_SUMMARY": writeup_summary_symbolic,
            }
        else:
            # Load summaries for non-symbolic prompting only.
            summary_files = [
                ("logs/0-run/draft_summary.json", "DRAFT_SUMMARY"),
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
            summaries_for_prompt = loaded_summaries

        # Prepare a new fresh latex folder
        if not osp.exists(osp.join(latex_folder, "template.tex")):
            shutil.copytree(
                "ai_scientist/blank_icml_latex", latex_folder, dirs_exist_ok=True
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
                    pdf_file=base_pdf_file + ".pdf",
                    fact_store_path=fact_store_path,
                    param_store_path=param_store_path,
                    rendered_tex_artifact_path=osp.join(latex_folder, "template.rendered.tex"),
                    used_facts_artifact_path=osp.join(latex_folder, "used_facts.json"),
                )
            else:
                compile_latex(latex_folder, base_pdf_file + ".pdf")
            return osp.exists(base_pdf_file + ".pdf")

        # Run small model for citation additions
        client, client_model = create_client(small_model)
        for round_idx in range(num_cite_rounds):
            with open(writeup_file, "r") as f:
                writeup_text = f.read()
            try:
                references_bib = re.search(
                    r"\\begin{filecontents}{references.bib}(.*?)\\end{filecontents}",
                    writeup_text,
                    re.DOTALL,
                )
                if references_bib is None:
                    raise ValueError("No references.bib found in template.tex")
                citations_text = references_bib.group(1)
                context_for_citation = (combined_summaries_str, citations_text)

                addition, done = get_citation_addition(
                    client,
                    client_model,
                    context_for_citation,
                    round_idx,
                    num_cite_rounds,
                    idea_text,
                )
                if done:
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
                            pattern_end = r"\end{filecontents}"
                            revised = writeup_text.replace(
                                pattern_end, f"\n{addition}{pattern_end}"
                            )
                            with open(writeup_file, "w") as fo:
                                fo.write(revised)
            except Exception:
                print("EXCEPTION in perform_writeup (citation round):")
                print(traceback.format_exc())
                continue

        desc_map = {}
        plot_generation_notes = ""
        if not symbolic_facts:
            aggregator_path = osp.join(base_folder, "auto_plot_aggregator.py")
            aggregator_code = "No aggregator script found."
            if osp.exists(aggregator_path):
                with open(aggregator_path, "r") as fa:
                    aggregator_code = fa.read()
            try:
                vlm_client, vlm_model = create_vlm_client(small_model)
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
                "Please also consider which plots should naturally be grouped together as subfigures."
            )

        # Construct final prompt for big model, placing the figure descriptions alongside the plot list
        big_model_system_message = writeup_system_message_template.format(
            page_limit=page_limit
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

        combined_prompt = writeup_prompt.format(
            idea_text=idea_text,
            writeup_context_pack=json.dumps(writeup_context_pack, indent=2),
            facts_instructions=facts_instructions,
            remediation_instructions=remediation_instructions,
            plot_generation_notes=plot_generation_notes,
        )

        response, msg_history = get_response_from_llm(
            prompt=combined_prompt,
            client=big_client,
            model=big_client_model,
            system_message=big_model_system_message,
            print_debug=False,
        )

        latex_code_match = re.search(r"```latex(.*?)```", response, re.DOTALL)
        if not latex_code_match:
            raise SymbolicLatexError(
                "LLM response did not contain a complete ```latex``` block."
            )
        updated_latex_code = latex_code_match.group(1).strip()
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
            param_store=param_store,
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

        if n_writeup_reflections <= 0:
            compile_pdf = base_pdf_file + f"_{compile_attempt}.pdf"
            if symbolic_facts:
                compile_symbolic_latex_project(
                    latex_folder=latex_folder,
                    pdf_file=compile_pdf,
                    fact_store_path=fact_store_path,
                    param_store_path=param_store_path,
                    rendered_tex_artifact_path=osp.join(
                        latex_folder, "template.rendered.tex"
                    ),
                    used_facts_artifact_path=osp.join(latex_folder, "used_facts.json"),
                )
                save_validated_writeup(latex_folder, updated_latex_code)
            else:
                compile_latex(latex_folder, compile_pdf)
            compile_attempt += 1

        # Multiple reflection loops on the final LaTeX
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

            # Compile current version before reflection
            compile_pdf = base_pdf_file + f"_{compile_attempt}.pdf"
            if symbolic_facts:
                compile_symbolic_latex_project(
                    latex_folder=latex_folder,
                    pdf_file=compile_pdf,
                    fact_store_path=fact_store_path,
                    param_store_path=param_store_path,
                    rendered_tex_artifact_path=osp.join(latex_folder, "template.rendered.tex"),
                    used_facts_artifact_path=osp.join(latex_folder, "used_facts.json"),
                )
                save_validated_writeup(latex_folder, current_latex)
            else:
                compile_latex(latex_folder, compile_pdf)
            compile_attempt += 1
            print(f"Compiled {base_pdf_file}_{compile_attempt}.pdf")

            # Detect where "Impact Statement" appears
            impact_loc = detect_pages_before_impact(latex_folder)
            if impact_loc is not None:
                page_num, line_num = impact_loc
                reflection_page_info = (
                    f"\nCurrently, 'Impact Statement' begins on page {page_num}, approximately on line {line_num}. "
                    f"The page limit is {page_limit}, which is before the Impact Statement. "
                    f"Papers often look more professional if the main text is near or just under {page_limit} pages in length.\n"
                )
            else:
                reflection_page_info = "\nCould not detect 'Impact Statement' page (compilation or detection failed).\n"

            check_output = os.popen(
                f"chktex {writeup_file} -q -n2 -n24 -n13 -n1"
            ).read()

            reflection_guard_block = build_reflection_guard_block(
                symbolic_facts=symbolic_facts,
                remediation_instructions=remediation_instructions,
            )
            patch_contract = ""
            current_editable = ""
            if symbolic_facts:
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
                current_editable = editable_suffix(current_latex)

            reflection_prompt = (
                f"""
Now let's reflect and identify any issues (including but not limited to):
1) Are there any LaTeX syntax errors or style violations we can fix? Refer to the chktex output below.
2) Is the writing clear, and scientifically rigorous?
3) Have we included all relevant details from the summaries without hallucinating?
4) The following figures are available in the folder but not used in the LaTeX: {sorted(unused_figs)}
5) The following figure references in the LaTeX do not match any actual file: {sorted(invalid_figs)}
{reflection_page_info}
chktex results:
```
{check_output}
```

{reflection_guard_block}
"""
                + (
                    f"\n{patch_contract}\n\nCURRENT EDITABLE LaTeX (after \\end{{filecontents}}):\n```latex\n{current_editable}\n```\n"
                    if symbolic_facts
                    else (
                        "\nPlease provide a revised complete LaTeX in triple backticks, or repeat the same if no changes are needed.\n"
                        "Return the entire file in full. This must be an acceptable complete LaTeX writeup.\n"
                        "Do not hallucinate any details!\n"
                    )
                )
                + '\nIf you believe you are done, simply say: "I am done".\n'
            )

            reflection_response, msg_history = get_response_from_llm(
                prompt=reflection_prompt,
                client=big_client,
                model=big_client_model,
                system_message=big_model_system_message,
                msg_history=msg_history,
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
                    print(f"No valid JSON patch found in reflection step {i+1}.")
                    break
                ops = patch_json.get("ops")
                if not ops:
                    print(f"No changes in reflection step {i+1}.")
                    break
                try:
                    patch_result = apply_latex_patch_ops(current_latex, patch_json)
                    final_text, _ = fixup_latex_before_validation(
                        patch_result.latex_text,
                        scaffold_tex=current_latex,
                    )
                    final_text = _maybe_backanchor_numeric_literals(
                        final_text,
                        symbolic_facts=symbolic_facts,
                        store=store if symbolic_facts else None,
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
                    compile_pdf = base_pdf_file + f"_{compile_attempt}.pdf"
                    compile_symbolic_latex_project(
                        latex_folder=latex_folder,
                        pdf_file=compile_pdf,
                        fact_store_path=fact_store_path,
                        param_store_path=param_store_path,
                        rendered_tex_artifact_path=osp.join(
                            latex_folder, "template.rendered.tex"
                        ),
                        used_facts_artifact_path=osp.join(latex_folder, "used_facts.json"),
                    )
                    save_validated_writeup(latex_folder, final_text)
                    compile_attempt += 1
                    print(f"Compiled {base_pdf_file}_{compile_attempt}.pdf")
                    current_latex = final_text
                except Exception as exc:
                    with open(writeup_file, "w") as fo:
                        fo.write(current_latex)
                    print(
                        f"[writeup] discarded reflection {i+1} patch and kept previous validated draft: {exc}"
                    )
                    break
            else:
                reflection_code_match = re.search(
                    r"```latex(.*?)```", reflection_response, re.DOTALL
                )
                if not reflection_code_match:
                    print(f"No valid LaTeX code block found in reflection step {i+1}.")
                    break
                reflected_latex_code = reflection_code_match.group(1).strip()
                if reflected_latex_code == current_latex:
                    print(f"No changes in reflection step {i+1}.")
                    break
                final_text, _ = fixup_latex_before_validation(
                    reflected_latex_code,
                    scaffold_tex=current_latex,
                )
                with open(writeup_file, "w") as fo:
                    fo.write(final_text)
                validate_generated_writeup(
                    final_text,
                    symbolic_facts=symbolic_facts,
                    store=store if symbolic_facts else None,
                    param_store=param_store if symbolic_facts else None,
                    artifact_manifest=manifest,
                )
                compile_pdf = base_pdf_file + f"_{compile_attempt}.pdf"
                compile_latex(latex_folder, compile_pdf)
                compile_attempt += 1
                print(f"Compiled {base_pdf_file}_{compile_attempt}.pdf")

        final_pdf = base_pdf_file + f"_{compile_attempt-1}.pdf"
        if symbolic_facts:
            used_facts_path = osp.join(latex_folder, "used_facts.json")
            if not osp.exists(used_facts_path):
                raise RuntimeError(
                    "Symbolic-facts writeup expected used_facts.json but it was not generated."
                )
            with open(used_facts_path, "r", encoding="utf-8") as f:
                used_facts = json.load(f)
            with open(writeup_file, "r", encoding="utf-8") as f:
                symbolic_tex = f.read()
            manifest = ensure_artifact_manifest(
                base_folder=base_folder,
                store=store,
                tex_path=writeup_file,
            )
            claim_context_pack = build_claim_context_pack(
                symbolic_tex=symbolic_tex,
                used_facts=used_facts,
                artifact_manifest_summary=artifact_manifest_summary(manifest),
            )
            save_context_pack(
                osp.join(latex_folder, "claim_context_pack.json"),
                claim_context_pack,
            )

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
                audit_client, audit_model = create_client(audit_model_name)
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
                audit = generate_outsider_audit(
                    inputs=audit_inputs,
                    client=audit_client,
                    model=audit_model,
                )
                validate_outsider_audit(audit)
                audit_path = osp.join(latex_folder, "outsider_audit.json")
                save_outsider_audit(audit_path, audit)
                print(f"[audit] wrote outsider audit: {audit_path}")

        return osp.exists(final_pdf)

    except Exception as exc:
        trace_text = traceback.format_exc()
        report = classify_remediation_failure(
            phase="perform_writeup",
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
    from ai_scientist.env_utils import env_int, env_str_optional, load_env

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
        default=env_str_optional("AI_SCIENTIST_MODEL_WRITEUP_SMALL"),
        help=(
            "Model to use for writeup support steps (includes citation collection in normal mode). "
            "Supports arbitrary OpenAI-compatible model names."
        ),
    )
    parser.add_argument(
        "--big-model",
        type=str,
        default=env_str_optional("AI_SCIENTIST_MODEL_WRITEUP"),
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
        default=env_int("AI_SCIENTIST_WRITEUP_PAGE_LIMIT", 8),
        help="Target page limit for the main paper (excluding references, impact statement, etc.)",
    )
    args = parser.parse_args()
    if args.model is None or args.big_model is None:
        raise RuntimeError(
            "Missing writeup model configuration. Set AI_SCIENTIST_MODEL_WRITEUP_SMALL and "
            "AI_SCIENTIST_MODEL_WRITEUP in your .env file (or pass --model/--big-model)."
        )

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
