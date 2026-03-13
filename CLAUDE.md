# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **The AI Scientist** (SakanaAI) - a system for fully automated open-ended scientific discovery. It uses LLMs to autonomously generate research ideas, run experiments, write LaTeX papers, and perform peer review.

**Safety note:** This codebase executes LLM-written code via subprocess. Containerize and restrict web access when running.

## Key Commands

### Setup
```bash
conda create -n ai_scientist python=3.11 && conda activate ai_scientist
sudo apt-get install texlive-full
pip install -r requirements.txt
```

### Prepare Data (NanoGPT templates)
```bash
python data/enwik8/prepare.py
python data/shakespeare_char/prepare.py
python data/text8/prepare.py
```

### Run Baseline for a Template
```bash
cd templates/<template_name>
python experiment.py --out_dir run_0
python plot.py
```

### Run Full Pipeline
```bash
python launch_scientist.py \
  --model "claude-3-5-sonnet-20241022" \
  --experiment nanoGPT_lite \
  --num-ideas 2
```

Key flags: `--parallel N` (multi-GPU), `--gpus "0,1,2"`, `--improvement` (review-then-revise cycle), `--skip-idea-generation`, `--skip-novelty-check`, `--engine semanticscholar|openalex`.

### Standalone Review
```python
from ai_scientist.perform_review import load_paper, perform_review
from ai_scientist.llm import create_client

paper_text = load_paper("path/to/paper.pdf")
client, model = create_client("gpt-4o-2024-05-13")
review = perform_review(
    paper_text,
    model=model,
    client=client,
    num_reflections=5,
    num_fs_examples=1,
    num_reviews_ensemble=5,
    temperature=0.1,
)
```

## Architecture

### Pipeline (5 phases, orchestrated by `launch_scientist.py`)

```
Idea Generation → Novelty Check → Experiments → Paper Writeup → Peer Review (→ optional Improvement)
```

### Core Modules (`ai_scientist/`)

- **`llm.py`** - Multi-provider LLM abstraction. `create_client(model)` returns a client + model name pair. Supports Anthropic (direct/Bedrock/Vertex), OpenAI, DeepSeek, Google Gemini, OpenRouter. All API calls use `@backoff.on_exception` for retries. Key functions: `get_response_from_llm()`, `get_batch_responses_from_llm()`, `extract_json_between_markers()`.

- **`generate_ideas.py`** - Multi-turn idea generation with iterative reflection (default 3 rounds). Uses `seed_ideas.json` from the template for in-context learning. Each idea has: Name, Title, Experiment, Interestingness/Feasibility/Novelty scores. Novelty checking queries Semantic Scholar or OpenAlex APIs.

- **`perform_experiments.py`** - Uses [Aider](https://aider.chat) (`aider-chat` library) to LLM-edit `experiment.py`, then runs it as a subprocess. MAX_RUNS=5 per idea. Timeouts: 7200s per experiment run, 600s for plotting.

- **`perform_writeup.py`** - Generates LaTeX papers section-by-section with per-section prompts and iterative refinement. Manages citations via Semantic Scholar API. Validates references, figures, and runs `chktex` + `pdflatex` compilation. Few-shot examples in `fewshot_examples/`.

- **`perform_review.py`** - NeurIPS-style LLM review with ensemble voting (multiple reviewers). Outputs structured JSON with 15 rating fields. Supports improvement iteration based on identified weaknesses.

### Templates (`templates/`)

Each template is a self-contained research domain. Required structure:
- `experiment.py` - Must accept `--out_dir run_i`, save `final_info.json` with metrics
- `plot.py` - Generates visualizations from run results
- `prompt.json` - System message and task description
- `seed_ideas.json` - Example ideas for in-context learning
- `latex/template.tex` - Base paper template

Official templates: `nanoGPT`, `nanoGPT_lite`, `2d_diffusion`, `grokking`. All others are community contributions.

### Output Structure

Results go to `results/<experiment>/<timestamp>_<idea_name>/` containing modified code, run directories (`run_0` through `run_N`), `latex/` with generated paper, review files, and logs.

## Key Patterns

- **Message history accumulation**: Functions pass and return `msg_history` lists to maintain multi-turn LLM context across refinement rounds.
- **JSON extraction from LLM output**: `extract_json_between_markers()` parses structured data between `<JSON>` and `</JSON>` markers with regex fallback.
- **Aider integration**: Experiment code is modified by Aider's `Coder` class (diff edit format, no git, no streaming).
- **Parallel execution**: Worker pool with `multiprocessing.Queue`, one GPU per worker, 150s stagger between worker starts.

## Environment Variables

```
OPENAI_API_KEY          # OpenAI models
ANTHROPIC_API_KEY       # Anthropic direct
DEEPSEEK_API_KEY        # DeepSeek models
OPENROUTER_API_KEY      # OpenRouter (Llama)
GEMINI_API_KEY          # Google Gemini
S2_API_KEY              # Semantic Scholar (optional, for novelty check)
AWS_ACCESS_KEY_ID       # Bedrock
AWS_REGION_NAME         # Bedrock
ANTHROPIC_VERTEX_PROJECT_ID  # Vertex AI
```

## System Requirements

- Python 3.11, Linux with NVIDIA GPUs (CUDA), `pdflatex` and `chktex` (from `texlive-full`)
- Docker setup available in `experimental/Dockerfile`
