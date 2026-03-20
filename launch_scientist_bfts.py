import os.path as osp
import json
import argparse
import shutil
import os
import re
import sys
from datetime import datetime
from ai_scientist.llm import create_client
from ai_scientist.network_env import disable_http_proxy
from ai_scientist.project_env import load_project_env
from omegaconf import OmegaConf

from contextlib import contextmanager
from ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager import (
    perform_experiments_bfts,
)
from ai_scientist.treesearch.resume import load_stage_checkpoint
from ai_scientist.treesearch.bfts_utils import (
    idea_to_markdown,
    edit_bfts_config_file,
)
from ai_scientist.perform_plotting import aggregate_plots
from ai_scientist.perform_writeup import perform_writeup
from ai_scientist.perform_icbinb_writeup import (
    perform_writeup as perform_icbinb_writeup,
    gather_citations,
)
from ai_scientist.perform_llm_review import perform_review, load_paper
from ai_scientist.perform_vlm_review import perform_imgs_cap_ref_review
from ai_scientist.reliable.remediation import (
    load_remediation_report,
    remediation_retry_banner,
    should_retry_writeup,
    should_reuse_symbolic_writeup_artifacts,
)
from ai_scientist.reliable.symbolic_postprocess import perform_symbolic_postprocess_retry
from ai_scientist.utils.token_tracker import token_tracker
from ai_scientist.env_utils import env_bool, env_int, env_str, load_env

try:
    import torch
except ImportError:
    torch = None


def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def save_token_tracker(idea_dir):
    with open(osp.join(idea_dir, "token_tracker.json"), "w") as f:
        json.dump(token_tracker.get_summary(), f)
    with open(osp.join(idea_dir, "token_tracker_interactions.json"), "w") as f:
        json.dump(token_tracker.get_interactions(), f)


def parse_arguments():
    load_env()
    parser = argparse.ArgumentParser(description="Run AI scientist experiments")
    parser.add_argument(
        "--writeup-type",
        type=str,
        default=env_str("AI_SCIENTIST_WRITEUP_TYPE", "icbinb"),
        choices=["normal", "icbinb"],
        help="Type of writeup to generate (normal=8 page, icbinb=4 page)",
    )
    parser.add_argument(
        "--load_ideas",
        type=str,
        default=env_str(
            "AI_SCIENTIST_IDEAS_PATH",
            "ai_scientist/ideas/i_cant_believe_its_not_better.json",
        ),
        help="Path to a JSON file containing pregenerated ideas",
    )
    parser.add_argument(
        "--load_code",
        action="store_true",
        help="If set, load a Python file with same name as ideas file but .py extension",
    )
    parser.add_argument(
        "--idea_idx",
        type=int,
        default=env_int("AI_SCIENTIST_IDEA_IDX", 0),
        help="Index of the idea to run",
    )
    parser.add_argument(
        "--add_dataset_ref",
        action="store_true",
        help="If set, add a HF dataset reference to the idea",
    )
    parser.add_argument(
        "--writeup-retries",
        type=int,
        default=env_int("AI_SCIENTIST_WRITEUP_RETRIES", 3),
        help="Number of writeup attempts to try",
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
        "--attempt_id",
        type=int,
        default=env_int("AI_SCIENTIST_ATTEMPT_ID", 0),
        help="Attempt ID, used to distinguish same idea in different attempts in parallel runs",
    )
    parser.add_argument(
        "--model_agg_plots",
        type=str,
        default=env_str("AI_SCIENTIST_MODEL_AGG_PLOTS", "minimax/MiniMax-M2.7"),
        help="Model to use for plot aggregation",
    )
    parser.add_argument(
        "--plot-reflections",
        type=int,
        default=env_int("AI_SCIENTIST_PLOT_REFLECTIONS", 5),
        help="Number of reflection loops for plot aggregation.",
    )
    parser.add_argument(
        "--model_writeup",
        type=str,
        default=env_str("AI_SCIENTIST_MODEL_WRITEUP", "minimax/MiniMax-M2.7"),
        help="Model to use for writeup",
    )
    parser.add_argument(
        "--model_citation",
        type=str,
        default=env_str("AI_SCIENTIST_MODEL_CITATION", "minimax/MiniMax-M2.7"),
        help="Model to use for citation gathering",
    )
    parser.add_argument(
        "--num_cite_rounds",
        type=int,
        default=env_int("AI_SCIENTIST_NUM_CITE_ROUNDS", 20),
        help="Number of citation rounds to perform",
    )
    parser.add_argument(
        "--model_writeup_small",
        type=str,
        default=env_str("AI_SCIENTIST_MODEL_WRITEUP_SMALL", "gpt-5.4"),
        help="Smaller model to use for writeup",
    )
    parser.add_argument(
        "--model_review",
        type=str,
        default=env_str("AI_SCIENTIST_MODEL_REVIEW", "gpt-5.4"),
        help="Model to use for review main text and captions",
    )
    parser.add_argument(
        "--skip_writeup",
        action="store_true",
        help="If set, skip the writeup process",
    )
    parser.add_argument(
        "--skip_review",
        action="store_true",
        help="If set, skip the review process",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=env_str("AI_SCIENTIST_BFTS_CONFIG_PATH", "bfts_config.yaml"),
        help="Path to the BFTS config file to use for this run.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help=(
            "Optional path to a previously saved stage checkpoint.pkl. "
            "The new run will start from the next main stage after that checkpoint."
        ),
    )
    parser.add_argument(
        "--disable-http-proxy",
        action="store_true",
        help=(
            "Disable HTTP proxy usage for the current process (clears *PROXY env vars and sets NO_PROXY='*'). "
            "Leave this off if your network requires a system proxy/VPN to reach Vast.ai."
        ),
    )
    return parser.parse_args()


def _normalize_resume_idea(checkpoint_path: str) -> dict:
    checkpoint = load_stage_checkpoint(checkpoint_path)
    task_desc = checkpoint["task_desc"]
    if isinstance(task_desc, str):
        task_desc = json.loads(task_desc)
    if not isinstance(task_desc, dict):
        raise RuntimeError(
            f"Resume checkpoint {checkpoint_path} does not contain a JSON idea/task description."
        )
    return task_desc


def _load_idea_from_args(args):
    if args.resume_checkpoint:
        idea = _normalize_resume_idea(args.resume_checkpoint)
        print(
            "Resume checkpoint provided; using the checkpoint task description "
            f"for idea metadata: {idea.get('Name', '<unknown>')}"
        )
        return idea

    with open(args.load_ideas, "r") as f:
        ideas = json.load(f)
    print(f"Loaded {len(ideas)} pregenerated ideas from {args.load_ideas}")
    return ideas[args.idea_idx]


def get_available_gpus(gpu_ids=None):
    if gpu_ids is not None:
        return [int(gpu_id) for gpu_id in gpu_ids.split(",")]
    if torch is None:
        raise RuntimeError(
            "Local GPU execution requires PyTorch to be installed in the launcher environment."
        )
    return list(range(torch.cuda.device_count()))


def describe_execution_backend(config_path="bfts_config.yaml"):
    try:
        cfg = OmegaConf.load(config_path)
    except Exception:
        return f"Using GPUs: {get_available_gpus()}"
    backend = cfg.get("exec", {}).get("backend", "local")
    if backend == "wsl_ssh":
        wsl_cfg = cfg.get("exec", {}).get("wsl_ssh", {})
        host = wsl_cfg.get("host", "unknown")
        user = wsl_cfg.get("user", "unknown")
        distro = wsl_cfg.get("wsl_distro", "WSL")
        return f"Using WSL SSH execution backend ({user}@{host}, distro={distro})"
    if backend != "vast":
        return f"Using GPUs: {get_available_gpus()}"
    search_cfg = cfg.get("exec", {}).get("vast", {}).get("search", {})
    requested = search_cfg.get("num_gpus", "unknown")
    return f"Using Vast.ai execution backend (requested GPUs: {requested})"


def find_pdf_path_for_review(idea_dir):
    pdf_files = [f for f in os.listdir(idea_dir) if f.endswith(".pdf")]
    if not pdf_files:
        return None
    reflection_pdfs = [f for f in pdf_files if "reflection" in f]
    pdf_path = None
    if reflection_pdfs:
        # First check if there's a final version
        final_pdfs = [f for f in reflection_pdfs if "final" in f.lower()]
        if final_pdfs:
            # Use the final version if available
            pdf_path = osp.join(idea_dir, final_pdfs[0])
        else:
            # Try to find numbered reflections
            reflection_nums = []
            for f in reflection_pdfs:
                match = re.search(r"reflection[_.]?(\d+)", f)
                if match:
                    reflection_nums.append((int(match.group(1)), f))

            if reflection_nums:
                # Get the file with the highest reflection number
                highest_reflection = max(reflection_nums, key=lambda x: x[0])
                pdf_path = osp.join(idea_dir, highest_reflection[1])
            else:
                # Fall back to the first reflection PDF if no numbers found
                pdf_path = osp.join(idea_dir, reflection_pdfs[0])
    if pdf_path is None:
        pdf_path = osp.join(idea_dir, pdf_files[0])
    return pdf_path


@contextmanager
def redirect_stdout_stderr_to_file(log_file_path):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log = open(log_file_path, "a")
    sys.stdout = log
    sys.stderr = log
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log.close()


if __name__ == "__main__":
    args = parse_arguments()
    load_project_env()
    if args.disable_http_proxy:
        disable_http_proxy()
    os.environ["AI_SCIENTIST_ROOT"] = os.path.dirname(os.path.abspath(__file__))
    print(f"Set AI_SCIENTIST_ROOT to {os.environ['AI_SCIENTIST_ROOT']}")

    # Check available GPUs and adjust parallel processes if necessary
    print(describe_execution_backend(args.config_path))

    idea = _load_idea_from_args(args)

    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    idea_dir = f"experiments/{date}_{idea['Name']}_attempt_{args.attempt_id}"
    print(f"Results will be saved in {idea_dir}")
    os.makedirs(idea_dir, exist_ok=True)

    # Convert idea json to markdown file
    idea_path_md = osp.join(idea_dir, "idea.md")

    # If load_code is True, get the Python file with same name as JSON
    code = None
    if args.load_code:
        code_path = args.load_ideas.rsplit(".", 1)[0] + ".py"
        if os.path.exists(code_path):
            with open(code_path, "r") as f:
                code = f.read()
        else:
            print(f"Warning: Code file {code_path} not found")
    else:
        code_path = None

    idea_to_markdown(idea, idea_path_md, code_path)

    dataset_ref_code = None
    if args.add_dataset_ref:
        dataset_ref_path = "hf_dataset_reference.py"
        if os.path.exists(dataset_ref_path):
            with open(dataset_ref_path, "r") as f:
                dataset_ref_code = f.read()
        else:
            print(f"Warning: Dataset reference file {dataset_ref_path} not found")
            dataset_ref_code = None

    if dataset_ref_code is not None and code is not None:
        added_code = dataset_ref_code + "\n" + code
    elif dataset_ref_code is not None and code is None:
        added_code = dataset_ref_code
    elif dataset_ref_code is None and code is not None:
        added_code = code
    else:
        added_code = None

    print(added_code)

    # Add code to idea json if it was loaded
    if added_code is not None:
        idea["Code"] = added_code

    # Store raw idea json
    idea_path_json = osp.join(idea_dir, "idea.json")
    with open(idea_path_json, "w") as f:
        json.dump(idea, f, indent=4)

    config_path = args.config_path
    idea_config_path = edit_bfts_config_file(
        config_path,
        idea_dir,
        idea_path_json,
    )

    perform_experiments_bfts(
        idea_config_path,
        resume_checkpoint_path=args.resume_checkpoint,
    )
    experiment_results_dir = osp.join(idea_dir, "logs/0-run/experiment_results")
    if os.path.exists(experiment_results_dir):
        shutil.copytree(
            experiment_results_dir,
            osp.join(idea_dir, "experiment_results"),
            dirs_exist_ok=True,
        )

    aggregate_plots(
        base_folder=idea_dir,
        model=args.model_agg_plots,
        n_reflections=args.plot_reflections,
    )

    shutil.rmtree(osp.join(idea_dir, "experiment_results"))

    save_token_tracker(idea_dir)

    if not args.skip_writeup:
        writeup_success = False
        remediation_context = None
        remediation_report_path = osp.join(idea_dir, "latex", "remediation_failure.json")
        citations_text = gather_citations(
            idea_dir,
            num_cite_rounds=args.num_cite_rounds,
            small_model=args.model_citation,
        )
        for attempt in range(args.writeup_retries):
            print(f"Writeup attempt {attempt+1} of {args.writeup_retries}")
            if args.writeup_symbolic_facts and should_reuse_symbolic_writeup_artifacts(
                remediation_context
            ):
                writeup_success = perform_symbolic_postprocess_retry(
                    base_folder=idea_dir,
                    small_model=args.model_writeup_small,
                    big_model=args.model_writeup,
                    remediation_context=remediation_context,
                )
            elif args.writeup_type == "normal":
                writeup_success = perform_writeup(
                    base_folder=idea_dir,
                    small_model=args.model_writeup_small,
                    big_model=args.model_writeup,
                    page_limit=8,
                    symbolic_facts=args.writeup_symbolic_facts,
                    remediation_context=remediation_context,
                )
            else:
                writeup_success = perform_icbinb_writeup(
                    base_folder=idea_dir,
                    small_model=args.model_writeup_small,
                    big_model=args.model_writeup,
                    page_limit=4,
                    citations_text=citations_text,
                    symbolic_facts=args.writeup_symbolic_facts,
                    remediation_context=remediation_context,
                )
            if writeup_success:
                break
            report = load_remediation_report(remediation_report_path)
            if not should_retry_writeup(report):
                print("[remediation] stopping writeup retries after non-retryable failure.")
                break
            if attempt + 1 >= args.writeup_retries:
                print("[remediation] writeup retries exhausted after targeted repair attempts.")
                break
            print(remediation_retry_banner(report))
            remediation_context = report

        if not writeup_success:
            print("Writeup process did not complete successfully after all retries.")

    save_token_tracker(idea_dir)

    if not args.skip_review and not args.skip_writeup:
        # Perform paper review if the paper exists
        pdf_path = find_pdf_path_for_review(idea_dir)
        if pdf_path and os.path.exists(pdf_path):
            print("Paper found at: ", pdf_path)
            paper_content = load_paper(pdf_path)
            client, client_model = create_client(args.model_review)
            review_text = perform_review(paper_content, client_model, client)
            review_img_cap_ref = perform_imgs_cap_ref_review(
                client, client_model, pdf_path
            )
            with open(osp.join(idea_dir, "review_text.txt"), "w") as f:
                f.write(json.dumps(review_text, indent=4))
            with open(osp.join(idea_dir, "review_img_cap_ref.json"), "w") as f:
                json.dump(review_img_cap_ref, f, indent=4)
            print("Paper review completed.")
        else:
            print("No paper PDF found; skipping review.")

    print("Start cleaning up processes")
    # Kill all mp and torch processes associated with this experiment
    import psutil
    import signal

    # Get the current process and all its children
    current_process = psutil.Process()
    children = current_process.children(recursive=True)

    # First try graceful termination
    for child in children:
        try:
            child.send_signal(signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Wait briefly for processes to terminate
    gone, alive = psutil.wait_procs(children, timeout=3)

    # If any processes remain, force kill them
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Additional cleanup: find any orphaned processes containing specific keywords
    keywords = ["python", "torch", "mp", "bfts", "experiment"]
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            # Check both process name and command line arguments
            cmdline = " ".join(proc.cmdline()).lower()
            if any(keyword in cmdline for keyword in keywords):
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=3)
                if proc.is_running():
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            continue

    # Finally, terminate the current process
    # current_process.send_signal(signal.SIGTERM)
    # try:
    #     current_process.wait(timeout=3)
    # except psutil.TimeoutExpired:
    #     current_process.kill()

    # exit the program
    sys.exit(0)
