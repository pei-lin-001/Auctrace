<div align="center">
  <a href="https://github.com/SakanaAI/AI-Scientist_v2/blob/main/docs/logo_v1.jpg">
    <img src="docs/logo_v1.png" width="215" alt="AI Scientist v2 Logo" />
  </a>
  <h1>
    <b>The AI Scientist-v2: Workshop-Level Automated</b><br>
    <b>Scientific Discovery via Agentic Tree Search</b>
  </h1>
</div>

<p align="center">
  📚 <a href="https://pub.sakana.ai/ai-scientist-v2/paper">[Paper]</a> |
  📝 <a href="https://sakana.ai/ai-scientist-first-publication/"> [Blog Post]</a> |
  📂 <a href="https://github.com/SakanaAI/AI-Scientist-ICLR2025-Workshop-Experiment"> [ICLR2025 Workshop Experiment]</a>
</p>

Fully autonomous scientific research systems are becoming increasingly capable, with AI playing a pivotal role in transforming how scientific discoveries are made.
We are excited to introduce The AI Scientist-v2, a generalized end-to-end agentic system that has generated the first workshop paper written entirely by AI and accepted through peer review.

This system autonomously generates hypotheses, runs experiments, analyzes data, and writes scientific manuscripts. Unlike [its predecessor (AI Scientist-v1)](https://github.com/SakanaAI/AI-Scientist), the AI Scientist-v2 removes reliance on human-authored templates, generalizes across Machine Learning (ML) domains, and employs a progressive agentic tree search, guided by an experiment manager agent.

> **Note:**
> The AI Scientist-v2 doesn’t necessarily produce better papers than v1, especially when a strong starting template is available. v1 follows well-defined templates, leading to high success rates, while v2 takes a broader, more exploratory approach with lower success rates. v1 works best for tasks with clear objectives and a solid foundation, whereas v2 is designed for open-ended scientific exploration.

> **Caution!**
> This codebase will execute Large Language Model (LLM)-written code. There are various risks and challenges associated with this autonomy, including the potential use of dangerous packages, uncontrolled web access, and the possibility of spawning unintended processes. Ensure that you run this within a controlled sandbox environment (e.g., a Docker container). Use at your own discretion.

## Table of Contents

1.  [Requirements](#requirements)
    *   [Installation](#installation)
    *   [Supported Models and API Keys](#supported-models-and-api-keys)
2.  [Generate Research Ideas](#generate-research-ideas)
3.  [Run AI Scientist-v2 Paper Generation Experiments](#run-ai-scientist-v2-paper-generation-experiments)
4.  [Citing The AI Scientist-v2](#citing-the-ai-scientist-v2)
5.  [Frequently Asked Questions](#frequently-asked-questions)
6.  [Acknowledgement](#acknowledgement)

## Requirements

This code is designed to run on Linux with NVIDIA GPUs using CUDA and PyTorch.

### Installation

```bash
# Create a new conda environment
conda create -n ai_scientist python=3.11
conda activate ai_scientist

# Install PyTorch with CUDA support (adjust pytorch-cuda version for your setup)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install PDF and LaTeX tools
conda install anaconda::poppler
conda install conda-forge::chktex

# Install Python package requirements
pip install -r requirements.txt
```

Installation usually takes no more than one hour.

### Supported Models and API Keys

#### OpenAI Models

By default, the system uses the `OPENAI_API_KEY` environment variable for OpenAI models.

#### OpenAI-compatible Models

The repository now supports **arbitrary OpenAI-compatible model names** across the v2 experiment path, ideation, plotting, writeup, LLM review, and VLM review.

Set either:

```bash
export OPENAI_BASE_URL="https://your-compatible-endpoint/v1"
export OPENAI_API_KEY="YOUR_COMPATIBLE_KEY"
```

`OPENAI_API_BASE` is also accepted as an alias for `OPENAI_BASE_URL`.

or the explicit aliases:

```bash
export OPENAI_COMPATIBLE_BASE_URL="https://your-compatible-endpoint/v1"
export OPENAI_COMPATIBLE_API_KEY="YOUR_COMPATIBLE_KEY"
```

The repository also reads a local `.env` file at startup if present. To unify the token budget for the configured OpenAI-compatible models to 128k, set:

```bash
AI_SCIENTIST_CONTEXT_TOKENS=131072
```

This repository's request-side generation cap is controlled separately:

```bash
AI_SCIENTIST_MAX_OUTPUT_TOKENS=8192
```

When a compatible base URL is configured, plain model names such as `Qwen/Qwen3-32B` or `meta-llama/llama-3.1-70b-instruct` are sent through that OpenAI-compatible endpoint. This repository no longer maintains provider-specific SDK routes; if you want to use non-OpenAI models, expose them via your OpenAI-compatible gateway.

For the v2 tree-search path, each stage config can now declare an explicit `fallback_model`. When the primary OpenAI-compatible model exhausts the SDK's retry budget on retryable transport/provider errors, the request is retried once on the configured fallback model and the switch is logged explicitly instead of looping forever on the original model.

Example:

```yaml
agent:
  code:
    model: MiniMax-M2.5
    fallback_model: nvidia/nemotron-3-super-120b-a12b
```

#### Semantic Scholar API (Literature Search)

Our code can optionally use a Semantic Scholar API Key (`S2_API_KEY`) for higher throughput during literature search [if you have one](https://www.semanticscholar.org/product/api). This is used during both the ideation and paper writing stages. The system should work without it, though you might encounter rate limits or reduced novelty checking during ideation. If you experience issues with Semantic Scholar, you can skip the citation phase during paper generation.

#### Setting API Keys

Ensure you provide the necessary API keys as environment variables for the models you intend to use. For example:
```bash
export OPENAI_API_KEY="YOUR_OPENAI_KEY_HERE"
export OPENAI_BASE_URL="https://your-compatible-endpoint/v1"  # optional, for OpenAI-compatible APIs
export S2_API_KEY="YOUR_S2_KEY_HERE"
```

## Generate Research Ideas

Before running the full AI Scientist-v2 experiment pipeline, you first use the `ai_scientist/perform_ideation_temp_free.py` script to generate potential research ideas. This script uses an LLM to brainstorm and refine ideas based on a high-level topic description you provide, interacting with tools like Semantic Scholar to check for novelty.

1.  **Prepare a Topic Description:** Create a Markdown file (e.g., `my_research_topic.md`) describing the research area or theme you want the AI to explore. This file should contain sections like `Title`, `Keywords`, `TL;DR`, and `Abstract` to define the scope of the research. Refer to the example file `ai_scientist/ideas/i_cant_believe_its_not_better.md` for the expected structure and content format. Place your file in a location accessible by the script (e.g., the `ai_scientist/ideas/` directory).

2.  **Run the Ideation Script:** Execute the script from the main project directory, pointing it to your topic description file and specifying the desired LLM.

    ```bash
    python ai_scientist/perform_ideation_temp_free.py \
     --workshop-file "ai_scientist/ideas/my_research_topic.md" \
     --model gpt-4o-2024-05-13 \
     --max-num-generations 20 \
     --num-reflections 5
    ```
    *   `--workshop-file`: Path to your topic description Markdown file.
    *   `--model`: The LLM to use for generating ideas. This can now be any supported native model or any OpenAI-compatible model name when `OPENAI_BASE_URL` is configured.
    *   `--max-num-generations`: How many distinct research ideas to attempt generating.
    *   `--num-reflections`: How many refinement steps the LLM should perform for each idea.

3.  **Output:** The script will generate a JSON file named after your input Markdown file (e.g., `ai_scientist/ideas/my_research_topic.json`). This file will contain a list of structured research ideas, including hypotheses, proposed experiments, and related work analysis.

4.  **Proceed to Experiments:** Once you have the generated JSON file containing research ideas, you can proceed to the next section to run the experiments.

This ideation step guides the AI Scientist towards specific areas of interest and produces concrete research directions to be tested in the main experimental pipeline.

## Run AI Scientist-v2 Paper Generation Experiments

Using the JSON file generated in the previous ideation step, you can now launch the main AI Scientist-v2 pipeline. This involves running experiments via agentic tree search, analyzing results, and generating a paper draft.

Specify the models used for the write-up and review phases via command-line arguments.
The configuration for the best-first tree search (BFTS) is located in `bfts_config.yaml`. Adjust parameters in this file as needed.

All OpenAI-style stages in this repository now accept arbitrary model strings when `OPENAI_BASE_URL` or `OPENAI_COMPATIBLE_BASE_URL` is configured, so you are no longer limited to the previously hardcoded model list for ideation or writeup.

Key tree search configuration parameters in `bfts_config.yaml`:

-   `agent` config:
    -   Set `num_workers` (number of parallel exploration paths) and `steps` (maximum number of nodes to explore). For example, if `num_workers=3` and `steps=21`, the tree search will explore up to 21 nodes, expanding 3 nodes concurrently at each step.
    -   `num_seeds`: Should generally be the same as `num_workers` if `num_workers` is less than 3. Otherwise, set `num_seeds` to 3.
    -   Note: Other agent parameters like `k_fold_validation`, `expose_prediction`, and `data_preview` are not used in the current version.
-   `search` config:
    -   `max_debug_depth`: The maximum number of times the agent will attempt to debug a failing node before abandoning that search path.
    -   `debug_prob`: The probability of attempting to debug a failing node.
    -   `num_drafts`: The number of initial root nodes (i.e., the number of independent trees to grow) during Stage 1.

Example command to run AI-Scientist-v2 using a generated idea file (e.g., `my_research_topic.json`). Please review `bfts_config.yaml` for detailed tree search parameters. Do not set `load_code` if you do not want to initialize experimentation with a code snippet.

```bash
python launch_scientist_bfts.py \
 --load_ideas "ai_scientist/ideas/my_research_topic.json" \
 --load_code \
 --add_dataset_ref \
 --model_writeup o1-preview-2024-09-12 \
 --model_citation gpt-4o-2024-11-20 \
 --model_review gpt-4o-2024-11-20 \
 --model_agg_plots o3-mini-2025-01-31 \
 --num_cite_rounds 20
```

Once the initial experimental stage is complete, you will find a timestamped log folder inside the `experiments/` directory. Navigate to `experiments/"timestamp_ideaname"/logs/0-run/` within that folder to find the tree visualization file `unified_tree_viz.html`.
After all experiment stages are complete, the writeup stage begins. The writeup stage typically takes about 20 to 30 minutes in total. Once it finishes, you should see `timestamp_ideaname.pdf` in the `timestamp_ideaname` folder.
For this example run, all stages typically finish within several hours.

If a previous run already completed an earlier main stage and wrote a stage checkpoint, you can start a **new** run from the next main stage with:

```bash
python launch_scientist_bfts.py \
 --config-path ".codex-tasks/my-run/formal_bfts_config.yaml" \
 --resume-checkpoint "experiments/<previous_run>/logs/0-run/stage_2_baseline_tuning_1_first_attempt/checkpoint.pkl"
```

If `--resume-checkpoint` is provided, the launcher now uses the checkpoint's saved task description as the source of truth for idea metadata. This avoids accidentally resuming `failure_budget_controller` under an unrelated `--load_ideas/--idea_idx` combination.

This does not pretend to restore the killed process in place. Instead, it loads the completed stage state, creates a fresh experiment directory, and continues from the next main stage.

## Run Experiments on Vast.ai

This repository now includes a **Vast.ai execution backend** for the v2 tree-search pipeline. It is intended to replace the assumption that experiments must run on the local machine's CUDA-visible GPUs.

The integration uses Vast.ai's official instance APIs to:

- search rentable offers,
- create or reuse an instance,
- attach a real local SSH public key to the rented instance,
- verify that the instance is not only `running` but also passes a real SSH health probe,
- execute experiment workers remotely over SSH,
- automatically destroy the rented instance when the run finishes,
- and replace bad or unresponsive low-cost offers instead of silently hanging forever.

### Required environment variables

```bash
export VAST_API_KEY="YOUR_VAST_API_KEY"
export VAST_SSH_PRIVATE_KEY_PATH="$HOME/.ssh/id_ed25519"
export VAST_SSH_PUBLIC_KEY_PATH="$HOME/.ssh/id_ed25519.pub"
```

The launcher and Vast backend also read these keys from the repository-local `.env` file if present, using the same `os.environ.setdefault(...)` precedence as the OpenAI-compatible path.

If the environment variables are omitted, the backend will explicitly look for a local keypair in `~/.ssh/id_ed25519(.pub)` and then `~/.ssh/id_rsa(.pub)`. Vast.ai account-side key labels alone are not enough for unattended remote execution; the backend must have access to a real local public key to attach and, unless you rely on `ssh-agent`, the matching private key for the health probe and remote execution.

### Enable the backend

Edit [bfts_config.yaml](./bfts_config.yaml) and set:

```yaml
exec:
  backend: vast
```

Key Vast.ai options live under `exec.vast`, including:

- `existing_instance_id`: reuse an existing instance instead of auto-renting
- `offer_id`: force a specific offer ID during debugging or deterministic testing
- `image`: Docker image used when creating the instance
- `runtype`: defaults to `ssh_direct`
- `max_provision_attempts`: how many low-cost offers to try before failing
- `instance_poll_interval`: polling interval while waiting for the instance to boot
- `ssh_probe_*`: retries and timeouts for the explicit SSH health check
- `search.*`: offer filtering such as `num_gpus`, `reliability_min`, and `direct_port_count_min`
- `install_project_requirements`: whether to upload the local `requirements.txt` and install it once per remote image hash
- `requirements_file`: local requirements file to bootstrap on the Vast instance
- `pip_install_timeout`: timeout for the remote dependency installation step
- `auto_install_missing_packages`: whether to catch remote `ModuleNotFoundError` / `ImportError`, install the missing package, and retry the same execution
- `max_auto_dependency_installs`: cap on automatic dependency installs per execution attempt
- `setup_commands`: optional one-time remote setup commands
- `auto_destroy`: whether to destroy the instance on cleanup

### Notes

- The remote execution path syncs each worker workspace to the Vast.ai instance, runs the generated code remotely, then syncs artifacts back to the local `experiments/` directory.
- By default the backend now bootstraps the remote Python environment from the repo's `requirements.txt` before the first worker runs. `setup_commands` remain available for extra system or project-specific setup.
- The backend treats the **lowest-cost healthy** offer as the target. If an offer boots into a broken host state, never exposes a usable SSH banner, or closes SSH during key exchange, that instance is destroyed and the next low-cost offer is tried explicitly.
- If a required Python package is still missing remotely, the Vast execution path now retries explicitly: it logs the missing module, installs the mapped pip package on the active instance, and reruns the same code. If installation fails, the error still surfaces explicitly and the run does not silently fall back to local execution.

## Citing The AI Scientist-v2

If you use **The AI Scientist-v2** in your research, please cite our work as follows:

```bibtex
@article{aiscientist_v2,
  title={The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search},
  author={Yamada, Yutaro and Lange, Robert Tjarko and Lu, Cong and Hu, Shengran and Lu, Chris and Foerster, Jakob and Clune, Jeff and Ha, David},
  journal={arXiv preprint arXiv:2504.08066},
  year={2025}
}
```

## Frequently Asked Questions

**Why wasn't a PDF or a review generated for my experiment?**

The AI Scientist-v2 completes experiments with a success rate that depends on the chosen foundation model, and the complexity of the idea. Higher success rates are generally observed when using powerful models like Claude 3.5 Sonnet for the experimentation phase.

**What is the estimated cost per experiment?**

The ideation step cost depends on the LLM used and the number of generations/reflections, but is generally low (a few dollars). For the main experiment pipeline, using Claude 3.5 Sonnet for the experimentation phase typically costs around $15–$20 per run. The subsequent writing phase adds approximately $5 when using the default models specified in the example command. Using GPT-4o for `model_citation` is recommended as it can help reduce writing costs.

**How do I run The AI Scientist-v2 for different subject fields?**

First, perform the [Generate Research Ideas](#generate-research-ideas) step. Create a new Markdown file describing your desired subject field or topic, following the structure of the example `ai_scientist/ideas/i_cant_believe_its_not_better.md`. Run the `perform_ideation_temp_free.py` script with this file to generate a corresponding JSON idea file. Then, proceed to the [Run AI Scientist-v2 Paper Generation Experiments](#run-ai-scientist-v2-paper-generation-experiments) step, using this JSON file with the `launch_scientist_bfts.py` script via the `--load_ideas` argument.

**What should I do if I have problems accessing the Semantic Scholar API?**

The Semantic Scholar API is used to assess the novelty of generated ideas and to gather citations during the paper write-up phase. If you don't have an API key, encounter rate limits, you may be able to skip these phases.

**I encountered a "CUDA Out of Memory" error. What can I do?**

This error typically occurs when the AI Scientist-v2 attempts to load or run a model that requires more GPU memory than available on your system. To resolve this, you can try updating your ideation prompt file (`ai_scientist/ideas/my_research_topic.md`) to suggest using smaller models for the experiments.

## Acknowledgement

The tree search component implemented within the `ai_scientist` directory is built on top of the [AIDE](https://github.com/WecoAI/aideml) project. We thank the AIDE developers for their valuable contributions and for making their work publicly available.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=SakanaAI/AI-Scientist-v2&type=Date)](https://star-history.com/#SakanaAI/AI-Scientist-v2&Date)

## ⚖️ License & Responsible Use

This project is licensed under **The AI Scientist Source Code License** (a derivative of the Responsible AI License). 

**Mandatory Disclosure:** By using this code, you are legally bound to clearly and prominently disclose the use of AI in any resulting scientific manuscripts or papers. 

We recommend the following attribution in your paper's Abstract or Methods section:
> "This manuscript was autonomously generated using [The AI Scientist](https://github.com/SakanaAI/AI-Scientist)."
