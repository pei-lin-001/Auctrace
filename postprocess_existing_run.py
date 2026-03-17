from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any

from ai_scientist.llm import create_client
from ai_scientist.perform_icbinb_writeup import (
    gather_citations,
    perform_writeup as perform_icbinb_writeup,
)
from ai_scientist.perform_llm_review import load_paper, perform_review
from ai_scientist.perform_plotting import aggregate_plots
from ai_scientist.perform_vlm_review import perform_imgs_cap_ref_review
from ai_scientist.project_env import load_project_env
from ai_scientist.treesearch.log_summarization import overall_summarize
from ai_scientist.env_utils import env_int, env_str, load_env

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
        "--model_agg_plots",
        default=env_str("AI_SCIENTIST_MODEL_AGG_PLOTS", "openai/gpt-oss-120b"),
        help="Model for plot aggregation script generation.",
    )
    parser.add_argument(
        "--model_citation",
        default=env_str("AI_SCIENTIST_MODEL_CITATION", "openai/gpt-oss-120b"),
        help="Model for citation gathering.",
    )
    parser.add_argument(
        "--model_writeup",
        default=env_str("AI_SCIENTIST_MODEL_WRITEUP", "openai/gpt-oss-120b"),
        help="Model for LaTeX writeup (text).",
    )
    parser.add_argument(
        "--model_writeup_small",
        default=env_str("AI_SCIENTIST_MODEL_WRITEUP_SMALL", "gpt-5.4"),
        help="Model for VLM steps inside writeup (must support images).",
    )
    parser.add_argument(
        "--model_review",
        default=env_str("AI_SCIENTIST_MODEL_REVIEW", "gpt-5.4"),
        help="Model for review (text + VLM review).",
    )
    parser.add_argument(
        "--num_cite_rounds",
        type=int,
        default=env_int("AI_SCIENTIST_NUM_CITE_ROUNDS", 20),
        help="Number of citation gathering rounds.",
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

    idea_dir = Path(args.idea_dir).expanduser().resolve()
    if not idea_dir.exists():
        raise FileNotFoundError(f"idea-dir does not exist: {idea_dir}")

    log_dir = (idea_dir / args.log_dir).resolve()
    manager_path = log_dir / "manager.pkl"
    if not manager_path.exists():
        raise FileNotFoundError(f"manager.pkl not found: {manager_path}")

    manager = _load_manager(manager_path)
    cfg = getattr(manager, "cfg", None)
    journals = getattr(manager, "journals", None)
    if not isinstance(journals, dict) or not journals:
        raise RuntimeError("Loaded manager.pkl but did not find a non-empty journals dict.")

    print(f"[postprocess] idea_dir={idea_dir}")
    print(f"[postprocess] log_dir={log_dir}")
    print(f"[postprocess] journals={len(journals)}")

    print("[postprocess] generating summaries...")
    draft_summary, baseline_summary, research_summary, ablation_summary = overall_summarize(
        journals.items(), cfg
    )
    _write_json(log_dir / "draft_summary.json", draft_summary)
    _write_json(log_dir / "baseline_summary.json", baseline_summary)
    _write_json(log_dir / "research_summary.json", research_summary)
    _write_json(log_dir / "ablation_summary.json", ablation_summary)

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
        print("[postprocess] gathering citations...")
        citations_text = gather_citations(
            str(idea_dir),
            num_cite_rounds=args.num_cite_rounds,
            small_model=args.model_citation,
        )
        print("[postprocess] generating paper (icbinb writeup)...")
        ok = perform_icbinb_writeup(
            base_folder=str(idea_dir),
            citations_text=citations_text,
            small_model=args.model_writeup_small,
            big_model=args.model_writeup,
            page_limit=4,
        )
        if not ok:
            raise RuntimeError("perform_icbinb_writeup returned False (paper generation failed).")

    if not args.skip_review and not args.skip_writeup:
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


if __name__ == "__main__":
    main()
