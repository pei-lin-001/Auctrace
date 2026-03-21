from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import traceback
import tempfile
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    try:
        mpl_dir = Path(tempfile.gettempdir()) / "mplconfig"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    except Exception:
        pass

from ai_scientist.llm import create_client
from ai_scientist.perform_icbinb_writeup import (
    gather_citations,
    perform_writeup as perform_icbinb_writeup,
)
from ai_scientist.perform_writeup import perform_writeup as perform_normal_writeup
from ai_scientist.perform_llm_review import load_paper, perform_review
from ai_scientist.perform_plotting import aggregate_plots
from ai_scientist.perform_vlm_review import perform_imgs_cap_ref_review
from ai_scientist.project_env import load_project_env
from ai_scientist.reliable.symbolic_postprocess import perform_symbolic_postprocess_retry
from ai_scientist.reliable.remediation import (
    RemediationDecision,
    classify_remediation_failure,
    load_remediation_report,
    print_remediation_report,
    remediation_retry_banner,
    save_remediation_report,
    should_retry_writeup,
    should_reuse_symbolic_writeup_artifacts,
)
from ai_scientist.treesearch.log_summarization import overall_summarize, save_overall_summaries
from ai_scientist.env_utils import env_bool, env_int, env_str, env_str_optional, load_env

DEFAULT_LOG_DIR = Path("logs") / "0-run"


def _parse_args() -> argparse.Namespace:
    load_env()
    parser = argparse.ArgumentParser(
        description=(
            "Postprocess an existing experiments/<timestamp>_<idea>_attempt_<id> folder "
            "to generate summaries, figures, paper PDF, and reviews without rerunning stages."
        )
    )
    parser.add_argument(
        "--idea-dir",
        required=True,
        help="Path to the existing experiment directory under experiments/.",
    )
    parser.add_argument(
        "--log-dir",
        default=str(DEFAULT_LOG_DIR),
        help="Relative path from idea-dir to log dir (default: logs/0-run).",
    )
    parser.add_argument(
        "--skip_summaries",
        action="store_true",
        help="Skip overall_summarize (useful for review-only runs).",
    )
    parser.add_argument(
        "--model_agg_plots",
        default=env_str_optional("AI_SCIENTIST_MODEL_AGG_PLOTS"),
        help="Model for plot aggregation script generation.",
    )
    parser.add_argument(
        "--model_citation",
        default=env_str_optional("AI_SCIENTIST_MODEL_CITATION"),
        help="Model for citation gathering.",
    )
    parser.add_argument(
        "--model_writeup",
        default=env_str_optional("AI_SCIENTIST_MODEL_WRITEUP"),
        help="Model for LaTeX writeup (text).",
    )
    parser.add_argument(
        "--model_writeup_small",
        default=env_str_optional("AI_SCIENTIST_MODEL_WRITEUP_SMALL"),
        help="Model for VLM steps inside writeup (must support images).",
    )
    parser.add_argument(
        "--model_review",
        default=env_str_optional("AI_SCIENTIST_MODEL_REVIEW"),
        help="Model for review (text + VLM review).",
    )
    parser.add_argument(
        "--num_cite_rounds",
        type=int,
        default=env_int("AI_SCIENTIST_NUM_CITE_ROUNDS", 20),
        help="Number of citation gathering rounds.",
    )
    parser.add_argument(
        "--writeup-type",
        choices=("icbinb", "normal"),
        default=env_str("AI_SCIENTIST_WRITEUP_TYPE", "icbinb"),
        help="Which writeup implementation to run (default: icbinb).",
    )
    parser.add_argument(
        "--writeup-symbolic-facts",
        action=argparse.BooleanOptionalAction,
        default=env_bool("AI_SCIENTIST_WRITEUP_SYMBOLIC_FACTS", False),
        help=(
            "If enabled, use symbolic fact variables in LaTeX writeup: LLM must write \\fact{key} placeholders, "
            "and a deterministic renderer fills numeric values before compilation."
        ),
    )
    parser.add_argument(
        "--writeup-retries",
        type=int,
        default=env_int("AI_SCIENTIST_WRITEUP_RETRIES", 3),
        help="Maximum number of targeted writeup retries after a blocking remediation failure.",
    )
    parser.add_argument(
        "--plot_reflections",
        type=int,
        default=env_int("AI_SCIENTIST_PLOT_REFLECTIONS", 5),
        help="Number of reflection loops for plot aggregation.",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Skip aggregate_plots (assumes figures/ already exist).",
    )
    parser.add_argument(
        "--skip_writeup",
        action="store_true",
        help="Skip perform_writeup (assumes PDF already exists).",
    )
    parser.add_argument(
        "--skip_review",
        action="store_true",
        help="Skip review generation.",
    )
    parser.add_argument(
        "--keep_experiment_results_copy",
        action="store_true",
        help="Keep idea-dir/experiment_results copy after plotting.",
    )
    return parser.parse_args()


def _load_manager(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _copy_experiment_results(idea_dir: Path, log_dir: Path) -> Path:
    src = log_dir / "experiment_results"
    if not src.exists():
        raise FileNotFoundError(f"Missing experiment results directory: {src}")
    dst = idea_dir / "experiment_results"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def _pick_review_pdf(idea_dir: Path) -> Path:
    base = idea_dir.name
    preferred = [
        idea_dir / f"{base}_reflection_final_page_limit_noheader.pdf",
        idea_dir / f"{base}_reflection_final_noheader.pdf",
        idea_dir / f"{base}_reflection_final_page_limit.pdf",
        idea_dir / f"{base}_reflection_final.pdf",
        idea_dir / f"{base}.pdf",
    ]
    for cand in preferred:
        if cand.exists():
            return cand

    reflection_pdfs = sorted(idea_dir.glob(f"{base}_reflection*.pdf"))
    if reflection_pdfs:
        return reflection_pdfs[-1]

    any_pdfs = sorted(idea_dir.glob("*.pdf"))
    if any_pdfs:
        return any_pdfs[0]

    raise FileNotFoundError(f"No PDF found under {idea_dir}")


def main() -> None:
    args = _parse_args()
    load_project_env()

    missing_model_vars: list[str] = []
    if not args.skip_plots and args.model_agg_plots is None:
        missing_model_vars.append("AI_SCIENTIST_MODEL_AGG_PLOTS (or --model_agg_plots)")
    if not args.skip_writeup:
        if args.model_writeup is None:
            missing_model_vars.append("AI_SCIENTIST_MODEL_WRITEUP (or --model_writeup)")
        if args.model_writeup_small is None:
            missing_model_vars.append(
                "AI_SCIENTIST_MODEL_WRITEUP_SMALL (or --model_writeup_small)"
            )
        if args.writeup_type == "icbinb" and args.model_citation is None:
            missing_model_vars.append("AI_SCIENTIST_MODEL_CITATION (or --model_citation)")
    if not args.skip_review and args.model_review is None:
        missing_model_vars.append("AI_SCIENTIST_MODEL_REVIEW (or --model_review)")
    if missing_model_vars:
        joined = "\n- ".join(missing_model_vars)
        raise RuntimeError(
            "Missing required model configuration. Set these in your .env file:\n"
            f"- {joined}"
        )

    idea_dir = Path(args.idea_dir).expanduser().resolve()
    try:
        if not idea_dir.exists():
            raise FileNotFoundError(f"idea-dir does not exist: {idea_dir}")

        log_dir = (idea_dir / args.log_dir).resolve()
        print(f"[postprocess] idea_dir={idea_dir}")
        print(f"[postprocess] log_dir={log_dir}")
        if not args.skip_summaries:
            manager_path = log_dir / "manager.pkl"
            if not manager_path.exists():
                raise FileNotFoundError(f"manager.pkl not found: {manager_path}")

            manager = _load_manager(manager_path)
            cfg = getattr(manager, "cfg", None)
            journals = getattr(manager, "journals", None)
            if not isinstance(journals, dict) or not journals:
                raise RuntimeError("Loaded manager.pkl but did not find a non-empty journals dict.")
            print(f"[postprocess] journals={len(journals)}")

            print("[postprocess] generating summaries...")
            draft_summary, baseline_summary, research_summary, ablation_summary = overall_summarize(
                journals.items(), cfg
            )
            save_overall_summaries(
                base_folder=str(idea_dir),
                log_dir=str(log_dir),
                draft_summary=draft_summary,
                baseline_summary=baseline_summary,
                research_summary=research_summary,
                ablation_summary=ablation_summary,
            )

        copied_results_dir = None
        if not args.skip_plots:
            print("[postprocess] preparing experiment_results for plot aggregation...")
            copied_results_dir = _copy_experiment_results(idea_dir, log_dir)
            print("[postprocess] aggregating plots...")
            aggregate_plots(
                base_folder=str(idea_dir),
                model=args.model_agg_plots,
                n_reflections=args.plot_reflections,
            )
            if copied_results_dir is not None and not args.keep_experiment_results_copy:
                shutil.rmtree(copied_results_dir)

        if not args.skip_writeup:
            citations_text = None
            if args.writeup_type == "icbinb":
                print("[postprocess] gathering citations...")
                citations_text = gather_citations(
                    str(idea_dir),
                    num_cite_rounds=args.num_cite_rounds,
                    small_model=args.model_citation,
                )
            print(f"[postprocess] generating paper ({args.writeup_type} writeup)...")
            ok = False
            remediation_context = None
            remediation_report_path = idea_dir / "latex" / "remediation_failure.json"
            for attempt in range(args.writeup_retries):
                print(
                    f"[postprocess] writeup attempt {attempt + 1}/{args.writeup_retries}"
                )
                if args.writeup_symbolic_facts and should_reuse_symbolic_writeup_artifacts(
                    remediation_context
                ):
                    ok = perform_symbolic_postprocess_retry(
                        base_folder=str(idea_dir),
                        small_model=args.model_writeup_small,
                        big_model=args.model_writeup,
                        remediation_context=remediation_context,
                    )
                else:
                    if args.writeup_type == "icbinb":
                        ok = perform_icbinb_writeup(
                            base_folder=str(idea_dir),
                            citations_text=citations_text,
                            small_model=args.model_writeup_small,
                            big_model=args.model_writeup,
                            page_limit=4,
                            symbolic_facts=args.writeup_symbolic_facts,
                            remediation_context=remediation_context,
                        )
                    else:
                        ok = perform_normal_writeup(
                            base_folder=str(idea_dir),
                            num_cite_rounds=args.num_cite_rounds,
                            small_model=args.model_writeup_small,
                            big_model=args.model_writeup,
                            page_limit=8,
                            symbolic_facts=args.writeup_symbolic_facts,
                            remediation_context=remediation_context,
                        )
                if ok:
                    break
                report = load_remediation_report(str(remediation_report_path))
                if not should_retry_writeup(report):
                    print(
                        "[postprocess] stopping writeup retries after non-retryable failure."
                    )
                    break
                if attempt + 1 >= args.writeup_retries:
                    print(
                        "[remediation] writeup retries exhausted after targeted repair attempts."
                    )
                    break
                print(remediation_retry_banner(report))
                remediation_context = report
            if not ok:
                raise RuntimeError(
                    f"perform_writeup ({args.writeup_type}) returned False (paper generation failed)."
                )

        if not args.skip_review:
            pdf_path = _pick_review_pdf(idea_dir)
            print(f"[postprocess] reviewing paper: {pdf_path}")
            paper_content = load_paper(str(pdf_path))
            client, client_model = create_client(args.model_review)
            review_text = perform_review(paper_content, client_model, client)
            review_img_cap_ref = perform_imgs_cap_ref_review(client, client_model, str(pdf_path))
            (idea_dir / "review_text.txt").write_text(json.dumps(review_text, indent=4), encoding="utf-8")
            with (idea_dir / "review_img_cap_ref.json").open("w", encoding="utf-8") as f:
                json.dump(review_img_cap_ref, f, indent=4)
            print("[postprocess] paper review completed.")

        print("[postprocess] done.")
    except Exception as exc:
        trace_text = traceback.format_exc()
        report: RemediationDecision
        is_writeup_abort = isinstance(exc, RuntimeError) and (
            "perform_writeup" in str(exc) and "returned False" in str(exc)
        )
        upstream_report = None
        if "idea_dir" in locals() and is_writeup_abort:
            upstream_report = load_remediation_report(
                str(Path(idea_dir) / "latex" / "remediation_failure.json")
            )
        if upstream_report is not None:
            report = RemediationDecision(
                schema=upstream_report.schema,
                phase="postprocess_existing_run",
                failure_code=upstream_report.failure_code,
                blocking=upstream_report.blocking,
                retryable=upstream_report.retryable,
                retry_target=upstream_report.retry_target,
                exception_type=upstream_report.exception_type,
                message=f"[upstream:{upstream_report.phase}] {upstream_report.message}",
                traceback=upstream_report.traceback,
            )
        else:
            report = classify_remediation_failure(
                phase="postprocess_existing_run",
                exc=exc,
                traceback_text=trace_text,
            )
        report_path = (
            idea_dir / "logs" / "0-run" / "postprocess_remediation_failure.json"
            if idea_dir.exists()
            else Path("postprocess_remediation_failure.json")
        )
        save_remediation_report(
            str(report_path),
            report,
        )
        print_remediation_report(report)
        raise


if __name__ == "__main__":
    main()
