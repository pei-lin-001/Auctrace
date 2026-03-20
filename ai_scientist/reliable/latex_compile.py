from __future__ import annotations

import os
import os.path as osp
import re
import shutil
import subprocess
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path

from ai_scientist.latex_sanitize import (
    ensure_tex_has_end_document,
    ensure_tex_uses_references_bibliography,
    sanitize_tex_file_for_pdflatex,
)

from .facts import FactStore
from .params import ParamStore
from .renderer import render_symbolic_latex


@dataclass(frozen=True)
class LatexCommandFailure(RuntimeError):
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    def __str__(self) -> str:
        cmd = " ".join(self.command)
        return (
            f"LaTeX command failed (returncode={self.returncode}): {cmd}\n"
            f"--- stdout ---\n{self.stdout}\n"
            f"--- stderr ---\n{self.stderr}\n"
        )


_UNDER_REVIEW_RE = re.compile(r"(?i)under\\s+review")


def ensure_latex_template_assets(cwd: str) -> None:
    """Ensure latex scaffold assets exist for pdflatex compilation.

    Older runs (or partially-copied latex folders) may be missing the conference/workshop
    style files. When that happens, LaTeX may silently fall back to a system-installed
    style which can inject review headers like "Under review ...".
    """

    tex_path = Path(cwd) / "template.tex"
    if not tex_path.exists():
        return

    try:
        tex = tex_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return

    repo_root = Path(__file__).resolve().parents[2]
    copied: list[str] = []

    if "iclr2025" in tex:
        src_dir = repo_root / "ai_scientist" / "blank_icbinb_latex"
        copied.extend(
            _sync_assets(
                src_dir=src_dir,
                dst_dir=Path(cwd),
                names=(
                    "iclr2025.sty",
                    "iclr2025.bst",
                    "iclr2025.bib",
                    "natbib.sty",
                    "fancyhdr.sty",
                    "math_commands.tex",
                ),
            )
        )

    if "icml2025" in tex:
        src_dir = repo_root / "ai_scientist" / "blank_icml_latex"
        copied.extend(
            _sync_assets(
                src_dir=src_dir,
                dst_dir=Path(cwd),
                names=(
                    "icml2025.sty",
                    "icml2025.bst",
                    "algorithm.sty",
                    "algorithmic.sty",
                ),
            )
        )

    if copied:
        copied_str = ", ".join(sorted(set(copied)))
        print(f"[latex] synced template assets: {copied_str}")


def _sync_assets(*, src_dir: Path, dst_dir: Path, names: tuple[str, ...]) -> list[str]:
    copied: list[str] = []
    for name in names:
        src_path = src_dir / name
        if not src_path.exists():
            continue

        dst_path = dst_dir / name
        if dst_path.exists():
            try:
                contents = dst_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                contents = ""
            if _UNDER_REVIEW_RE.search(contents):
                shutil.copy2(src_path, dst_path)
                copied.append(name)
            continue

        shutil.copy2(src_path, dst_path)
        copied.append(name)
    return copied


def _sanitize_template_tex_for_pdflatex(cwd: str) -> None:
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
    if report.changed:
        changed_keys = [f"U+{ord(ch):04X}x{count}" for ch, count in report.replacements.items()]
        print(
            "[latex] sanitized template.tex for pdflatex unicode compatibility: "
            + ", ".join(changed_keys)
        )
        if report.remaining_non_ascii:
            remaining = ", ".join(sorted({f"U+{ord(ch):04X}" for ch in report.remaining_non_ascii}))
            print(f"[latex] warning: template.tex still contains non-ascii: {remaining}")


def _run(command: list[str], *, cwd: str, timeout: int) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise LatexCommandFailure(
            command=command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )


def compile_latex_project(cwd: str, pdf_file: str, *, timeout: int = 30) -> None:
    print("[latex] compiling (strict mode)")
    ensure_latex_template_assets(cwd)
    _sanitize_template_tex_for_pdflatex(cwd)

    commands = [
        ["pdflatex", "-interaction=nonstopmode", "template.tex"],
        ["bibtex", "template"],
        ["pdflatex", "-interaction=nonstopmode", "template.tex"],
        ["pdflatex", "-interaction=nonstopmode", "template.tex"],
    ]
    for cmd in commands:
        _run(cmd, cwd=cwd, timeout=timeout)

    shutil.move(osp.join(cwd, "template.pdf"), pdf_file)
    print(f"[latex] wrote pdf: {pdf_file}")


def compile_symbolic_latex_project(
    *,
    latex_folder: str,
    pdf_file: str,
    fact_store_path: str,
    param_store_path: str | None = None,
    timeout: int = 30,
    rendered_tex_artifact_path: str | None = None,
    used_facts_artifact_path: str | None = None,
) -> None:
    """Compile a LaTeX project whose template.tex contains \\fact{key} placeholders.

    The compile happens in a temporary directory:
    - The symbolic template.tex is rendered into a numeric template.tex in temp dir.
    - The temp dir is compiled.
    - Optional artifacts (rendered tex + used facts) are written outside temp dir.
    """

    temp_dir = osp.join(
        osp.dirname(latex_folder),
        f"_temp_render_compile_{uuid.uuid4().hex}",
    )
    try:
        shutil.copytree(latex_folder, temp_dir, dirs_exist_ok=True)
        symbolic_path = osp.join(temp_dir, "template.tex")
        with open(symbolic_path, "r", encoding="utf-8") as f:
            symbolic_tex = f.read()
        store = FactStore.load_json(fact_store_path)
        param_store = ParamStore.load_json(param_store_path) if param_store_path else None
        rendered_tex, used = render_symbolic_latex(symbolic_tex, store, param_store)

        with open(symbolic_path, "w", encoding="utf-8") as f:
            f.write(rendered_tex)

        if rendered_tex_artifact_path:
            with open(rendered_tex_artifact_path, "w", encoding="utf-8") as f:
                f.write(rendered_tex)
        if used_facts_artifact_path:
            import json

            with open(used_facts_artifact_path, "w", encoding="utf-8") as f:
                json.dump(used, f, indent=2, ensure_ascii=True)

        compile_latex_project(temp_dir, pdf_file, timeout=timeout)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
