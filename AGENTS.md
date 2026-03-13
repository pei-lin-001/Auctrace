# Repository Guidelines

## Project Structure & Module Organization

`ai_scientist/` contains the core orchestration code for idea generation, experiment execution, write-up, review, and model clients. `launch_scientist.py` is the main CLI entrypoint. `templates/<template_name>/` holds runnable research templates; each template should keep `experiment.py`, `plot.py`, `prompt.json`, `seed_ideas.json`, and usually `latex/template.tex`. Use `data/` for dataset preparation scripts, `docs/` for README assets, and treat `example_papers/`, `review_ai_scientist/`, and `review_iclr_bench/` as reference artifacts and evaluation data rather than core library code.

## Build, Test, and Development Commands

Create a Python 3.11 environment first, then install dependencies:

```bash
pip install -r requirements.txt
```

Prepare baseline data or runs before end-to-end generation:

```bash
python data/enwik8/prepare.py
cd templates/nanoGPT && python experiment.py --out_dir run_0 && python plot.py
```

Run a small end-to-end smoke test from the repo root:

```bash
python launch_scientist.py --model "gpt-4o-2024-05-13" --experiment nanoGPT_lite --num-ideas 2
```

Review pipeline changes with:

```bash
cd review_iclr_bench && python iclr_analysis.py --num_reviews 500 --batch_size 100
```

LaTeX write-up generation requires `pdflatex` and `chktex` installed on the host.

## Coding Style & Naming Conventions

Use 4-space indentation and Pythonic `snake_case` for modules, functions, and variables; reserve `PascalCase` for classes only. Keep shared orchestration logic in `ai_scientist/` and template-specific logic inside the relevant `templates/` subdirectory. Preserve template directory names exactly as the CLI expects, including existing mixed styles such as `nanoGPT_lite`, `MACE`, and `earthquake-prediction`. Do not hardcode API keys or model credentials.

## Testing Guidelines

There is no top-level `tests/` suite today. Validate changes with the narrowest real workflow: rerun the touched template’s `experiment.py` and `plot.py`, use a small `launch_scientist.py` smoke run for orchestration changes, and rerun `review_iclr_bench/iclr_analysis.py` or the README review snippet for review changes. In PRs, record the exact command and the output artifact path you checked.

## Commit & Pull Request Guidelines

Recent history uses short imperative subjects such as `Add ...`, `Fix ...`, `Update ...`, and `Support ...`; follow that pattern and keep each commit scoped to one logical change. PRs should summarize the affected module or template, list new dependencies or API assumptions, and include concrete verification evidence. If you change generated figures, papers, or review outputs, attach representative artifact paths or screenshots. Link the related issue or prior template PR when relevant.

## Security & Runtime Notes

Load credentials from environment variables such as `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, or `GEMINI_API_KEY`. This repository executes LLM-written code, so prefer containerized or otherwise isolated runs; `experimental/Dockerfile` is the starting point.
